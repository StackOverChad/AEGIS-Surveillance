import os
import time
import numpy as np

# Try to import torch and torchvision, fall back to Pure-NumPy if not available
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] PyTorch is not installed in the current environment. The system will fall back to a Pure-NumPy/OpenCV motion tracker to simulate SlowFast anomaly detection.")

# Model weights directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "slowfast_mil_anomaly.pt")

# ==========================================
# PyTorch Model Definition (Conditional)
# ==========================================
if TORCH_AVAILABLE:
    class ConvBlock3d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super(ConvBlock3d, self).__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class SlowFastAnomalyDetectorModel(nn.Module):
        """
        SlowFast PyTorch model definition matching slowfast_mil_anomaly_colab.py
        """
        def __init__(self, num_classes=1, feature_dim=128):
            super(SlowFastAnomalyDetectorModel, self).__init__()
            
            # --- FAST PATHWAY ---
            self.fast_conv1 = ConvBlock3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
            self.fast_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.fast_res2 = nn.Sequential(
                ConvBlock3d(8, 16, kernel_size=(3, 3, 3), padding=1),
                ConvBlock3d(16, 16, kernel_size=(3, 3, 3), padding=1)
            )
            
            # --- SLOW PATHWAY ---
            self.slow_conv1 = ConvBlock3d(3, 32, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
            self.slow_pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            
            self.lateral_proj = nn.Conv3d(16, 32, kernel_size=(5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0))
            
            self.slow_res2 = nn.Sequential(
                ConvBlock3d(64, 64, kernel_size=(3, 3, 3), padding=1),
                ConvBlock3d(64, 64, kernel_size=(3, 3, 3), padding=1)
            )
            
            # --- HEAD ---
            self.avgpool_slow = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.avgpool_fast = nn.AdaptiveAvgPool3d((1, 1, 1))
            
            self.fc_features = nn.Linear(80, feature_dim)
            self.relu_fc = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.5)
            self.classifier = nn.Linear(feature_dim, num_classes)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            fast_feat = self.fast_conv1(x)
            fast_feat = self.fast_pool1(fast_feat)
            fast_feat = self.fast_res2(fast_feat)
            
            slow_x = x[:, :, ::8, :, :]
            slow_feat = self.slow_conv1(slow_x)
            slow_feat = self.slow_pool1(slow_feat)
            
            lateral_fast = self.lateral_proj(fast_feat)
            slow_feat = torch.cat([slow_feat, lateral_fast], dim=1)
            
            slow_feat = self.slow_res2(slow_feat)
            
            slow_pool = self.avgpool_slow(slow_feat).view(slow_feat.size(0), -1)
            fast_pool = self.avgpool_fast(fast_feat).view(fast_feat.size(0), -1)
            
            combined = torch.cat([slow_pool, fast_pool], dim=1)
            deep_features = self.relu_fc(self.fc_features(combined))
            
            out = self.dropout(deep_features)
            score = self.sigmoid(self.classifier(out))
            return score, deep_features


# ==========================================
# Inference Pipeline Wrapper Class
# ==========================================

class ModelRunner:
    def __init__(self):
        self.is_mock = True
        self.model = None
        self.device = None
        self.last_raw_score = 0.0
        self.last_motion_score = 0.0
        self.last_motion_gate = 0.0
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cpu")
            try:
                # Try to load standard weights
                if os.path.exists(MODEL_PATH):
                    print(f"Loading PyTorch SlowFast weights from: {MODEL_PATH}", flush=True)
                    self.model = SlowFastAnomalyDetectorModel()
                    self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                    self.model.eval()
                    self.is_mock = False
                    print("SlowFast Model initialized successfully in inference mode on CPU.", flush=True)
                else:
                    print(f"[INFO] Weight file not found at {MODEL_PATH}. Initializing dummy PyTorch weights with motion-tracking heuristics.", flush=True)
                    self.model = SlowFastAnomalyDetectorModel()
                    self.model.eval()
            except Exception as e:
                print(f"[WARNING] Error loading PyTorch model: {e}. Falling back to motion simulation.", flush=True)

    def preprocess(self, frame, target_res=(112, 112)):
        """
        Downsamples frame to target_res, converts to RGB, and normalizes.
        Returns NumPy array if pure-numpy mode, or PyTorch tensor if torch is available.
        """
        import cv2
        # Resize to target resolution
        resized = cv2.resize(frame, target_res, interpolation=cv2.INTER_LINEAR)
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        if TORCH_AVAILABLE:
            # Normalize and convert to FloatTensor [Channels, Height, Width]
            tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
            # Normalize with ImageNet stats
            mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
            std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor
        else:
            # Normalize to 0-1 range
            normalized = rgb_frame.astype(np.float32) / 255.0
            return normalized

    def predict(self, raw_frames, preprocessed_buffer):
        """
        Runs inference over a 32-frame buffer.
        Arguments:
            raw_frames: A list of 32 raw BGR numpy arrays (used for motion heuristic and clip output).
            preprocessed_buffer: A list of 32 preprocessed frames (tensors or arrays).
        Returns:
            anomaly_score: float (0.0 to 1.0)
            inference_latency_ms: float
        """
        start_time = time.time()
        
        # Calculate base motion metric (useful for mock simulation or fallback)
        motion_score = self._calculate_motion_activity(raw_frames)
        self.last_motion_score = motion_score
        
        if self.is_mock or not TORCH_AVAILABLE:
            # Simulate CPU inference delay based on resolution and temporal buffer
            # Resolution factor (tensors are typically 112x112, larger resolutions scale compute)
            if preprocessed_buffer:
                shape = preprocessed_buffer[0].shape
                # Use shape elements: shape[-1] and shape[-2] (width and height)
                h, w = shape[-2], shape[-1]
            else:
                h, w = 112, 112
                
            res_factor = (h * w) / (112 * 112)
            time.sleep(0.04 * res_factor)  # Simulate 40ms * resolution factor CPU execution time
            
            # Anomaly score is motion-weighted, adding some natural noise
            noise = np.random.normal(0, 0.02)
            # Clip motion score to 0-1, scale it to map to anomaly likelihood
            anomaly_score = float(np.clip(0.05 + 0.94 * (motion_score - 0.004) / 0.005 + noise, 0.05, 0.99))
            self.last_raw_score = anomaly_score
            self.last_motion_gate = 1.0
            
            latency = (time.time() - start_time) * 1000.0
            return anomaly_score, latency
 
        # If we have a real model loaded and torch is active:
        try:
            with torch.no_grad():
                # Stack 32 frames along the temporal dimension: [3, 32, H, W]
                # Then add batch dimension: [1, 3, 32, H, W]
                video_tensor = torch.stack(preprocessed_buffer, dim=1).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs, _ = self.model(video_tensor)
                raw_score = float(outputs.item())
                self.last_raw_score = raw_score
                
                if self.is_mock:
                    # Blend with scaled motion score for demo responsiveness
                    scaled_motion = float(np.clip((motion_score - 0.004) / 0.005, 0.0, 1.0))
                    score = float(np.clip(raw_score * 0.3 + scaled_motion * 0.7, 0.05, 0.99))
                    motion_gate = 1.0
                else:
                    # Apply motion gating to suppress false positives when there is no movement on real weights.
                    motion_gate = float(np.clip((motion_score - 0.005) / 0.004, 0.0, 1.0))
                    
                    # The PyTorch model outputs ~0.0 for normal scenes.
                    # We blend in a tiny bit of the motion score (max 0.15) to keep the dashboard graph "alive" 
                    # and reflect general movement, while ensuring true anomalies (raw_score > 0.5) trigger.
                    alive_baseline = float(np.clip(motion_score * 0.5, 0.0, 0.15))
                    score = float(np.clip((raw_score * motion_gate) + alive_baseline, 0.0, 1.0))
                    
                self.last_motion_gate = motion_gate
                print(f"[AEGIS Inference] Raw Score: {raw_score:.3f} | Motion Score: {motion_score:.4f} | Gate: {motion_gate:.3f} | Final: {score:.3f}", flush=True)
                
                latency = (time.time() - start_time) * 1000.0
                return score, latency
                
        except Exception as e:
            # Fail-safe fallback to motion score
            scaled_motion = float(np.clip((motion_score - 0.004) / 0.005, 0.0, 1.0))
            anomaly_score = float(np.clip(0.05 + 0.94 * scaled_motion, 0.05, 0.99))
            self.last_raw_score = anomaly_score
            self.last_motion_gate = 1.0
            print(f"[AEGIS Inference ERROR] {e}. Falling back to motion score.", flush=True)
            latency = (time.time() - start_time) * 1000.0
            return anomaly_score, latency

    def _calculate_motion_activity(self, raw_frames):
        """
        Computes motion activity by comparing successive frames in the buffer.
        Returns the average fraction of pixels changing by more than a noise threshold.
        """
        if len(raw_frames) < 2:
            return 0.0
            
        import cv2
        diffs = []
        # Sample 5 frames spaced across the buffer to keep CPU impact low
        indices = np.linspace(0, len(raw_frames) - 2, 5, dtype=int)
        
        for idx in indices:
            f1 = raw_frames[idx]
            f2 = raw_frames[idx + 1]
            
            # Downsample frames heavily to 32x32 for extremely fast comparison
            gray1 = cv2.cvtColor(cv2.resize(f1, (32, 32)), cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(cv2.resize(f2, (32, 32)), cv2.COLOR_BGR2GRAY)
            
            diff = cv2.absdiff(gray1, gray2)
            # Filter out camera compression and sensor noise below 8 gray levels of change
            _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
            # Fraction of pixels changed
            change_fraction = np.mean(thresh) / 255.0
            diffs.append(change_fraction)
            
        avg_change = float(np.mean(diffs))
        return avg_change

if __name__ == "__main__":
    # Test runner logic
    import cv2
    print("Testing ModelRunner...")
    runner = ModelRunner()
    
    # Create mock frames
    mock_frames = [np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(32)]
    preprocessed = [runner.preprocess(f, (112, 112)) for f in mock_frames]
    
    score, latency = runner.predict(mock_frames, preprocessed)
    print(f"Prediction complete. Score: {score:.4f}, Latency: {latency:.2f}ms")
