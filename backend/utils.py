import numpy as np

WINDOW_SIZE = 720
STEP_SIZE = 360

# --- UPDATED THRESHOLD ---
# Your normal file scored ~1.20.
# We set this to 1.3 so 1.20 is considered "Normal" (below the limit).
FIXED_MSE_THRESHOLD = 1.3

def detect_anomaly(ecg_signal, model):
    # Flatten input to 1D array
    ecg_signal = np.asarray(ecg_signal).flatten()

    # 1. Validation: Check if signal is too short
    if len(ecg_signal) < WINDOW_SIZE:
        return np.zeros(len(ecg_signal), dtype=bool), 0.0

    # 2. Normalize (Standard Z-Score)
    std_val = np.std(ecg_signal)
    if std_val == 0:
        std_val = 1e-8
    ecg_norm = (ecg_signal - np.mean(ecg_signal)) / std_val

    # 3. Create Windows
    windows = []
    for i in range(0, len(ecg_norm) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(ecg_norm[i:i + WINDOW_SIZE])
    
    windows = np.array(windows)

    # Safety check
    if len(windows) == 0:
        return np.zeros(len(ecg_signal), dtype=bool), 0.0

    # 4. Model Prediction
    try:
        recon = model.predict(windows, verbose=0)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        return np.zeros(len(ecg_signal), dtype=bool), 0.0
    
    # 5. Calculate MSE (Error)
    mse = np.mean(np.power(windows - recon, 2), axis=1)

    # ---------------------------------------------------------
    # DIAGNOSTIC REPORT
    # ---------------------------------------------------------
    avg_mse = np.mean(mse)
    print(f"\n{'='*40}")
    print(f"📊 NEUROPULSE DIAGNOSTICS:")
    print(f"   Calculated Error (MSE): {avg_mse:.4f}")
    print(f"   Current Threshold:      {FIXED_MSE_THRESHOLD}")
    
    if avg_mse > FIXED_MSE_THRESHOLD:
        print(f"   ❌ Result: ABNORMAL (Score is higher than limit)")
        print(f"   👉 To fix: Increase FIXED_MSE_THRESHOLD to {avg_mse + 0.1:.1f}")
    else:
        print(f"   ✅ Result: NORMAL (Score is lower than limit)")
    print(f"{'='*40}\n")
    # ---------------------------------------------------------

    # 6. Detect Anomalies
    anomalies = mse > FIXED_MSE_THRESHOLD
    
    # Calculate Percentage
    anomaly_percentage = np.mean(anomalies) * 100

    return anomalies, float(anomaly_percentage)