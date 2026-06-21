// AEGIS Surveillance - Dashboard Orchestrator

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const resolutionSlider = document.getElementById("resolution-range");
    const fpsSlider = document.getElementById("fps-range");
    const intervalSlider = document.getElementById("interval-range");
    const thresholdSlider = document.getElementById("threshold-range");
    
    const valResolution = document.getElementById("val-resolution");
    const valFps = document.getElementById("val-fps");
    const valInterval = document.getElementById("val-interval");
    const valThreshold = document.getElementById("val-threshold");
    
    const metricCpu = document.getElementById("metric-cpu");
    const metricMemory = document.getElementById("metric-memory");
    const metricLatency = document.getElementById("metric-latency");
    const metricDropped = document.getElementById("metric-dropped");
    
    const streamImg = document.getElementById("live-stream-feed");
    const streamSourceSelect = document.getElementById("stream-source-select");
    const anomalyBadge = document.getElementById("anomaly-overlay-badge");
    const anomalyBadgeText = document.getElementById("anomaly-badge-text");
    
    const streamStatResolution = document.getElementById("stream-stat-resolution");
    const streamStatFps = document.getElementById("stream-stat-fps");
    const streamStatSource = document.getElementById("stream-stat-source");
    
    const triggerAnomalyBtn = document.getElementById("trigger-anomaly-btn");
    const clearAlertsBtn = document.getElementById("clear-alerts-btn");
    const alertsTbody = document.getElementById("alerts-tbody");
    
    // RTSP Inputs
    const rtspInputContainer = document.getElementById("rtsp-input-container");
    const rtspUrlInput = document.getElementById("rtsp-url-input");
    const rtspApplyBtn = document.getElementById("rtsp-apply-btn");
    
    // Video Modal Elements
    const videoModal = document.getElementById("video-modal");
    const modalVideoPlayer = document.getElementById("modal-video-player");
    const modalAlertSummary = document.getElementById("modal-alert-summary");
    const closeModalBtn = document.getElementById("close-modal-btn");
    
    // State Variables
    let chartInstance = null;
    const maxChartPoints = 40;
    let chartLabels = Array(maxChartPoints).fill("");
    let chartData = Array(maxChartPoints).fill(0.0);
    
    let pytorchActive = true; // Will check backend status
    let pollingInterval = null;
    let currentAnomalyThreshold = parseFloat(thresholdSlider.value);

    // ==========================================
    // 1. Chart.js Initialization
    // ==========================================
    function initChart() {
        const ctx = document.getElementById("anomalyChart").getContext("2d");
        
        // Custom gradient for filling under chart line
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, "rgba(99, 102, 241, 0.3)");
        gradient.addColorStop(1, "rgba(99, 102, 241, 0.0)");

        chartInstance = new Chart(ctx, {
            type: "line",
            data: {
                labels: chartLabels,
                datasets: [
                    {
                        label: "Anomaly Score",
                        data: chartData,
                        borderColor: "#3b82f6",
                        borderWidth: 3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        backgroundColor: gradient,
                        tension: 0.3
                    },
                    {
                        label: "Threshold",
                        data: Array(maxChartPoints).fill(currentAnomalyThreshold),
                        borderColor: "#ef4444",
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: "rgba(255, 255, 255, 0.02)"
                        },
                        ticks: {
                            display: false
                        }
                    },
                    y: {
                        min: 0.0,
                        max: 1.0,
                        grid: {
                            color: "rgba(255, 255, 255, 0.05)"
                        },
                        ticks: {
                            color: "#9ca3af",
                            font: {
                                family: "Outfit"
                            }
                        }
                    }
                }
            }
        });
    }

    function updateChart(newScore) {
        if (!chartInstance) return;
        
        // Push new value, shift array
        chartData.push(newScore);
        chartData.shift();
        
        // Update threshold lines in case it changed
        chartInstance.data.datasets[1].data = Array(maxChartPoints).fill(currentAnomalyThreshold);
        chartInstance.update("none"); // Update with no animation for speed
    }

    // ==========================================
    // 2. Interactive Simulator Calculations
    // ==========================================
    function recalculateSimulatorMetrics() {
        const resolution = parseInt(resolutionSlider.value);
        const fps = parseInt(fpsSlider.value);
        const interval = parseInt(intervalSlider.value);
        
        // Display values next to label
        valResolution.innerText = `${resolution} px`;
        valFps.innerText = `${fps} FPS`;
        valInterval.innerText = `${interval} ms`;
        valThreshold.innerText = currentAnomalyThreshold.toFixed(2);
        
        // Update stream footer pills
        streamStatResolution.innerText = `${resolution} × ${resolution}`;
        streamStatFps.innerText = `${(1000 / interval).toFixed(1)} FPS`;
        
        // A. CPU Load (%)
        // CPU load scales quadratically with resolution, linearly with fps, and inversely with interval.
        const resRatio = resolution / 112;
        const fpsRatio = fps / 30;
        const intervalRatio = 500 / interval;
        
        let cpuLoad = 2.5 + 4.5 * (resRatio * resRatio) * fpsRatio * intervalRatio;
        if (pytorchActive) {
            cpuLoad *= 1.8; // PyTorch inference has higher CPU load than numpy fallback
        }
        cpuLoad = Math.min(cpuLoad, 98.5); // Cap CPU at 98.5%
        metricCpu.innerText = `${cpuLoad.toFixed(1)}%`;
        
        // B. Memory Footprint (MB)
        // Basic app structures take ~180MB. PyTorch library adds ~150MB.
        // Cache queue of frames takes memory depending on resolution.
        let ramUsage = 180 + (pytorchActive ? 160 : 0);
        ramUsage += (resRatio * resRatio) * 14.5;
        metricMemory.innerText = `${Math.round(ramUsage)} MB`;
        
        // C. Local Latency (ms)
        // Image resizing and normal preprocessing + network read overhead
        let latency = 8.5 + 25.5 * (resRatio * resRatio);
        if (pytorchActive) {
            latency += 15.0; // PyTorch forward pass overhead
        }
        metricLatency.innerText = `${latency.toFixed(1)} ms`;
        
        // D. Dropped Frames (%)
        // Camera streaming frame rate vs model ingestion interval.
        // If camera feeds 30 FPS, but inference runs 2 times a second (1000/500),
        // we drop 28 frames.
        const inferenceFps = 1000 / interval;
        let droppedRate = 0;
        if (fps > inferenceFps) {
            droppedRate = ((fps - inferenceFps) / fps) * 100;
        }
        metricDropped.innerText = `${droppedRate.toFixed(0)}%`;
        
        // Push settings update to FastAPI
        saveSettingsToBackend();
    }

    function saveSettingsToBackend() {
        const payload = {
            resolution: parseInt(resolutionSlider.value),
            stream_fps: parseInt(fpsSlider.value),
            inference_interval: parseInt(intervalSlider.value),
            threshold: currentAnomalyThreshold,
            source: streamSourceSelect.value,
            rtsp_url: rtspUrlInput ? rtspUrlInput.value.trim() : ""
        };
        
        return fetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .catch(err => console.error("Error updating settings:", err));
    }


    // ==========================================
    // 3. Backend Polling & UI Rendering
    // ==========================================
    function startPolling() {
        if (pollingInterval) clearInterval(pollingInterval);
        
        pollingInterval = setInterval(() => {
            // Fetch live metrics
            fetch("/api/metrics")
                .then(res => res.json())
                .then(data => {
                    pytorchActive = data.pytorch_installed;
                    document.getElementById("model-mode-badge").innerText = data.pytorch_installed ? 
                        (data.model_mocked ? "SlowFast CPU (Mocked)" : "SlowFast CPU (Loaded)") : "Pure NumPy Fallback";
                    
                    // Update chart with newest anomaly score
                    updateChart(data.current_score);
                    
                    // If anomaly score is higher than threshold, update overlay badge
                    if (data.current_score >= currentAnomalyThreshold) {
                        anomalyBadge.className = "stream-overlay-badge anomaly-active";
                        anomalyBadgeText.innerText = "STATUS: ANOMALY DETECTED";
                        document.getElementById("stream-card").style.borderColor = "var(--neon-red)";
                        document.getElementById("stream-card").style.boxShadow = "0 8px 32px var(--neon-red-glow)";
                    } else {
                        anomalyBadge.className = "stream-overlay-badge";
                        anomalyBadgeText.innerText = "STATUS: SAFE";
                        document.getElementById("stream-card").style.borderColor = "var(--card-border)";
                        document.getElementById("stream-card").style.boxShadow = "0 8px 32px 0 rgba(0, 0, 0, 0.3)";
                    }
                })
                .catch(err => {
                    console.error("Error polling metrics:", err);
                    document.getElementById("system-status-text").innerText = "Pipe Connection Offline";
                    document.getElementById("system-status-text").parentElement.querySelector(".status-dot").style.backgroundColor = "var(--neon-red)";
                });
                
            // Fetch recent alerts
            fetch("/api/alerts")
                .then(res => res.json())
                .then(data => {
                    renderAlertsTable(data);
                })
                .catch(err => console.error("Error polling alerts:", err));
                
        }, 500);
    }

    function renderAlertsTable(alerts) {
        if (!alerts || alerts.length === 0) {
            alertsTbody.innerHTML = `
                <tr>
                    <td colspan="5" class="no-alerts-placeholder">
                        <i class="fa-solid fa-circle-info"></i> No anomalies detected yet. System secure.
                    </td>
                </tr>`;
            return;
        }
        
        let html = "";
        alerts.forEach(alert => {
            const formattedTime = new Date(alert.timestamp).toLocaleString();
            const scoreClass = alert.anomaly_score >= 0.85 ? "score-badge" : "score-badge warning";
            const statusClass = alert.status === "sent" ? "status-badge sent" : "status-badge logged";
            const statusIcon = alert.status === "sent" ? '<i class="fa-solid fa-paper-plane"></i> Sent' : '<i class="fa-solid fa-box-archive"></i> Logged';
            
            const hasClip = alert.clip_path ? true : false;
            const actionBtn = hasClip ? 
                `<button class="action-btn-circle play-clip-btn" data-id="${alert.id}" data-summary="${alert.llm_summary || ''}">
                    <i class="fa-solid fa-play"></i>
                 </button>` : 
                `<button class="action-btn-disabled" disabled>
                    <i class="fa-solid fa-ban"></i>
                 </button>`;
            
            html += `
                <tr>
                    <td><strong>${formattedTime}</strong></td>
                    <td><span class="${scoreClass}">${alert.anomaly_score.toFixed(3)}</span></td>
                    <td class="alert-summary-cell">${alert.llm_summary || "Analyzing emergency rules..."}</td>
                    <td><span class="${statusClass}">${statusIcon}</span></td>
                    <td>${actionBtn}</td>
                </tr>`;
        });
        
        alertsTbody.innerHTML = html;
        
        // Add event listeners to play buttons
        document.querySelectorAll(".play-clip-btn").forEach(btn => {
            btn.onclick = (e) => {
                const alertId = btn.getAttribute("data-id");
                const summary = btn.getAttribute("data-summary");
                openVideoModal(alertId, summary);
            };
        });
    }

    // ==========================================
    // 4. Modal Operations
    // ==========================================
    function openVideoModal(alertId, summary) {
        modalVideoPlayer.src = `/api/alerts/${alertId}/clip`;
        modalAlertSummary.innerText = summary || "Anomaly video clip metadata details.";
        videoModal.classList.remove("hidden");
    }

    function closeVideoModal() {
        modalVideoPlayer.src = "";
        videoModal.classList.add("hidden");
    }

    // ==========================================
    // 5. Event Listeners
    // ==========================================
    
    // Sliders
    resolutionSlider.oninput = recalculateSimulatorMetrics;
    fpsSlider.oninput = recalculateSimulatorMetrics;
    intervalSlider.oninput = recalculateSimulatorMetrics;
    
    thresholdSlider.oninput = (e) => {
        currentAnomalyThreshold = parseFloat(e.target.value);
        recalculateSimulatorMetrics();
    };

    // Source change
    streamSourceSelect.onchange = (e) => {
        const source = e.target.value;
        
        // Toggle RTSP input fields
        if (source === "rtsp") {
            rtspInputContainer.classList.remove("hidden");
            streamStatSource.innerText = "CCTV Feed";
        } else {
            rtspInputContainer.classList.add("hidden");
            streamStatSource.innerText = source === "webcam" ? "Webcam Input" : "Synthetic Cam";
            
            // Show loading spinner momentarily to feel realistic
            const loading = document.getElementById("loading-overlay");
            loading.classList.remove("hidden");
            
            saveSettingsToBackend().then(() => {
                setTimeout(() => {
                    // Update stream src to refresh stream
                    streamImg.src = `/api/stream?t=${Date.now()}`;
                    loading.classList.add("hidden");
                }, 600);
            });
        }
    };

    // Apply RTSP stream URL
    rtspApplyBtn.onclick = () => {
        const url = rtspUrlInput.value.trim();
        if (!url) {
            alert("Please enter a valid RTSP CCTV stream URL (e.g., rtsp://username:password@ip_address:port/path).");
            return;
        }
        
        const loading = document.getElementById("loading-overlay");
        loading.classList.remove("hidden");
        
        saveSettingsToBackend().then(() => {
            setTimeout(() => {
                // Update stream src to refresh stream
                streamImg.src = `/api/stream?t=${Date.now()}`;
                loading.classList.add("hidden");
            }, 800);
        });
    };

    // Action buttons
    triggerAnomalyBtn.onclick = () => {
        fetch("/api/inject_intrusion", { method: "POST" })
            .then(res => res.json())
            .then(data => {
                console.log("Anomaly intrusion injected:", data);
                // Temporarily flash button
                triggerAnomalyBtn.classList.remove("btn-danger-glow");
                triggerAnomalyBtn.style.backgroundColor = "var(--neon-green)";
                triggerAnomalyBtn.innerHTML = '<i class="fa-solid fa-circle-check"></i> Injection Done';
                setTimeout(() => {
                    triggerAnomalyBtn.classList.add("btn-danger-glow");
                    triggerAnomalyBtn.style.backgroundColor = "";
                    triggerAnomalyBtn.innerHTML = '<i class="fa-solid fa-burst"></i> Inject Intrusion';
                }, 1500);
            })
            .catch(err => console.error("Error injecting anomaly:", err));
    };

    clearAlertsBtn.onclick = () => {
        if (confirm("Are you sure you want to clear all security logs from the database?")) {
            fetch("/api/clear_alerts", { method: "POST" })
                .then(res => res.json())
                .then(data => {
                    console.log("Database alert logs cleared:", data);
                    renderAlertsTable([]);
                })
                .catch(err => console.error("Error clearing logs:", err));
        }
    };

    closeModalBtn.onclick = closeVideoModal;
    
    // Close modal when clicking outside content
    videoModal.onclick = (e) => {
        if (e.target === videoModal) {
            closeVideoModal();
        }
    };

    // ==========================================
    // 6. Application Startup
    // ==========================================
    initChart();
    recalculateSimulatorMetrics();
    startPolling();
});
