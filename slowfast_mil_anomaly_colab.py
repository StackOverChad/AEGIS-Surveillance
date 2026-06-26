# -*- coding: utf-8 -*-
"""
Smart Home Spatiotemporal Surveillance System - Model Training (Google Colab)
File: slowfast_mil_anomaly_colab.py

This script is designed to run on Google Colab to train a lightweight SlowFast PyTorch
network on spatial/temporal features of video frames using Multiple Instance Learning (MIL)
and Center Loss (for clustering normal features).

INSTRUCTIONS FOR GOOGLE COLAB:
1. Open Google Colab (https://colab.research.google.com).
2. Create a new notebook with GPU runtime: Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU (or other).
3. Upload this script or copy-paste this code into a cell.
4. Mount your Google Drive by uncommenting the drive mount cells or clicking the folder icon.
5. Install necessary packages if missing (PyTorch is installed by default on Colab).
6. Run the cells to train the model.
7. Once training is complete, the weights file 'slowfast_mil_anomaly.pt' will be saved.
8. Download 'slowfast_mil_anomaly.pt' and place it in your local backend:
   `e:\home surviellance\backend\models\slowfast_mil_anomaly.pt`
"""

import os
import time
import torch
import torch.nn as nn   
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. SlowFast Network Architecture
# ==========================================

class ConvBlock3d(nn.Module):
    """A helper 3D Convolutional Block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SlowFastAnomalyDetector(nn.Module):
    """
    A lightweight SlowFast Network adapted for resource-conscious local execution.
    - Input dimensions: Batch x Channels (3) x Temporal Frames (32) x Height (112) x Width (112)
    - Slow Pathway: Captures spatial features (4 frames, high channels).
    - Fast Pathway: Captures temporal features (32 frames, low channels).
    """
    def __init__(self, num_classes=1, feature_dim=128):
        super(SlowFastAnomalyDetector, self).__init__()
        
        # --- FAST PATHWAY (High temporal, low channels) ---
        # Input size: [B, 3, 32, 112, 112]
        self.fast_conv1 = ConvBlock3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.fast_res2 = nn.Sequential(
            ConvBlock3d(8, 16, kernel_size=(3, 3, 3), padding=1),
            ConvBlock3d(16, 16, kernel_size=(3, 3, 3), padding=1)
        )
        
        # --- SLOW PATHWAY (Low temporal, high channels) ---
        # Subsamples frames (every 8th frame, i.e., 4 frames)
        # Input size: [B, 3, 4, 112, 112]
        self.slow_conv1 = ConvBlock3d(3, 32, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Lateral connection projection (Fast pathway has 16 channels, downsampled temporally by 8, projected to 32)
        # Fast pathway output is fused into Slow pathway
        self.lateral_proj = nn.Conv3d(16, 32, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0))
        
        # Slow Res Block 2 (Slow Channels + Projected Fast Channels: 32 + 32 = 64)
        self.slow_res2 = nn.Sequential(
            ConvBlock3d(64, 64, kernel_size=(3, 3, 3), padding=1),
            ConvBlock3d(64, 64, kernel_size=(3, 3, 3), padding=1)
        )
        
        # --- HEAD / CLASSIFIER ---
        self.avgpool_slow = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool_fast = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Combined feature size: 64 (slow) + 16 (fast) = 80 channels
        self.fc_features = nn.Linear(80, feature_dim)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        
        # Final classification score
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: [B, 3, 32, 112, 112]
        # Fast Pathway processes all 32 frames
        fast_feat = self.fast_conv1(x)
        fast_feat = self.fast_pool1(fast_feat)
        fast_feat = self.fast_res2(fast_feat) # [B, 16, 32, 28, 28]
        
        # Slow Pathway processes 4 frames (every 8th frame)
        slow_x = x[:, :, ::8, :, :] # [B, 3, 4, 112, 112]
        slow_feat = self.slow_conv1(slow_x)
        slow_feat = self.slow_pool1(slow_feat) # [B, 32, 4, 28, 28]
        
        # Lateral fusion: Project fast pathway to slow pathway shape and add
        lateral_fast = self.lateral_proj(fast_feat) # [B, 32, 4, 28, 28]
        slow_feat = torch.cat([slow_feat, lateral_fast], dim=1) # [B, 64, 4, 28, 28]
        
        slow_feat = self.slow_res2(slow_feat) # [B, 64, 4, 28, 28]
        
        # Pooling
        slow_pool = self.avgpool_slow(slow_feat).view(slow_feat.size(0), -1) # [B, 64]
        fast_pool = self.avgpool_fast(fast_feat).view(fast_feat.size(0), -1) # [B, 16]
        
        # Concatenate pathways
        combined = torch.cat([slow_pool, fast_pool], dim=1) # [B, 80]
        
        # Extract deep features
        deep_features = self.relu_fc(self.fc_features(combined))
        
        # Classification
        out = self.dropout(deep_features)
        score = self.sigmoid(self.classifier(out))
        
        return score, deep_features

# ==========================================
# 2. Loss Functions: MIL Loss & Center Loss
# ==========================================

class MILLoss(nn.Module):
    """
    Multiple Instance Learning (MIL) ranking loss.
    Assumes a batch consists of anomalous (positive) and normal (negative) video bags.
    L = max(0, 1 - max(P_anomaly) + max(P_normal))
    """
    def __init__(self, margin=1.0):
        super(MILLoss, self).__init__()
        self.margin = margin

    def forward(self, positive_scores, negative_scores):
        # positive_scores: Scores of segments in an anomalous video [B, S]
        # negative_scores: Scores of segments in a normal video [B, S]
        
        # In MIL, the bag score is the maximum segment score
        max_pos, _ = torch.max(positive_scores, dim=1)
        max_neg, _ = torch.max(negative_scores, dim=1)
        
        # Ranking Loss: encourage max_pos > max_neg + margin
        loss = torch.clamp(self.margin - max_pos + max_neg, min=0.0)
        return torch.mean(loss)

class CenterLoss(nn.Module):
    """
    Center Loss.
    Encourages features of normal (negative class) frames to cluster tightly around a center.
    This helps the model represent 'normalcy' robustly on sparse training data.
    """
    def __init__(self, feature_dim=128, device='cuda'):
        super(CenterLoss, self).__init__()
        self.feature_dim = feature_dim
        self.device = device
        # Normal feature center (trainable parameter)
        self.center = nn.Parameter(torch.randn(1, feature_dim).to(device))

    def forward(self, features, labels):
        # features: [B, D] (deep features from model)
        # labels: [B] (0 for normal, 1 for anomaly)
        
        # We only compute center loss for the normal class (label == 0)
        normal_mask = (labels == 0).float().unsqueeze(1)
        normal_features = features * normal_mask
        
        # Number of normal samples in the batch
        num_normal = torch.sum(normal_mask).item()
        if num_normal == 0:
            return torch.tensor(0.0).to(self.device)
            
        # Distance of normal features to the center
        normal_center = self.center.expand(features.size(0), -1) * normal_mask
        loss = 0.5 * torch.sum((normal_features - normal_center) ** 2) / num_normal
        return loss

# ==========================================
# 3. UCF-Crime Video Dataset (Mock/Real Adapter)
# ==========================================

class UCFCrimeDataset(Dataset):
    """
    Dataset loader for UCF-Crime. 
    In Google Colab, it reads pre-extracted features or processes frames.
    For demonstration, this generates synthetic tensors that mimic the actual data.
    """
    def __init__(self, num_samples=100, num_frames=32, height=112, width=112, mode='train'):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.mode = mode
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random spatial-temporal frames (mimics a video clip batch)
        # Normal videos have label 0, anomalous videos have label 1
        label = 1 if idx < (self.num_samples // 2) else 0
        
        # Synthetic video tensor: [3 channels, 32 frames, 112 height, 112 width]
        video_tensor = torch.randn(3, self.num_frames, self.height, self.width)
        
        # If it's anomalous, let's inject a localized spike (pattern) in the last few frames
        if label == 1:
            video_tensor[:, 20:, :, :] += 2.0
            
        return video_tensor, label

# ==========================================
# 4. Training Loop (Executable on Colab)
# ==========================================

def train_slowfast_mil():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Hyperparameters
    epochs = 10
    batch_size = 4
    learning_rate = 0.001
    center_loss_weight = 0.1
    
    # Initialize Dataset and Dataloader
    # In practice on Colab, load actual UCF-Crime frames
    dataset = UCFCrimeDataset(num_samples=60, num_frames=32, height=112, width=112)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize Model, Losses, and Optimizer
    model = SlowFastAnomalyDetector().to(device)
    mil_loss_fn = MILLoss()
    center_loss_fn = CenterLoss(feature_dim=128, device=device)
    
    optimizer = optim.Adam(
        list(model.parameters()) + [center_loss_fn.center], 
        lr=learning_rate, 
        weight_decay=1e-5
    )
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            scores, features = model(videos) # scores: [B, 1], features: [B, 128]
            
            # --- MIL Ranking Loss Construction ---
            # To apply MIL loss, we separate positive (anomalous) bags and negative (normal) bags
            pos_mask = (labels == 1)
            neg_mask = (labels == 0)
            
            if torch.sum(pos_mask) > 0 and torch.sum(neg_mask) > 0:
                pos_scores = scores[pos_mask].view(1, -1) # Mock segments in positive bag
                neg_scores = scores[neg_mask].view(1, -1) # Mock segments in negative bag
                
                # Compute MIL Ranking Loss
                mil_loss = mil_loss_fn(pos_scores, neg_scores)
            else:
                # Fallback if batch doesn't contain both classes
                bce_loss_fn = nn.BCELoss()
                mil_loss = bce_loss_fn(scores, labels.float().unsqueeze(1))
            
            # Compute Center Loss
            center_loss = center_loss_fn(features, labels)
            
            # Combine losses
            total_loss = mil_loss + center_loss_weight * center_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Center: {center_loss_fn.center.mean().item():.4f} - Time: {elapsed:.2f}s")
        
    print("Training finished!")
    
    # Save the finalized model weights
    # For Colab, we save to local dir first, then optionally copy to Drive
    output_filename = "slowfast_mil_anomaly.pt"
    torch.save(model.state_dict(), output_filename)
    print(f"Model saved successfully to: {os.path.abspath(output_filename)}")
    
    # Google Drive export code template (commented out by default)
    print("\n--- GOOGLE DRIVE EXPORT INSTRUCTIONS ---")
    print("# To copy the model to your Google Drive, mount Drive in Colab and run:")
    print("# from google.colab import drive")
    print("# drive.mount('/content/drive')")
    print(f"# !cp {output_filename} '/content/drive/MyDrive/{output_filename}'")
    print("-----------------------------------------")

if __name__ == "__main__":
    train_slowfast_mil()
