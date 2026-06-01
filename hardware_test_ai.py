import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 1. DATA ENGINEERING: SIMULATE FLEET-SCALE TEST DATA
def generate_manufacturing_data(num_samples=5000):
    print("[INFO] Ingesting fleet-scale hardware test records...")
    np.random.seed(42)
    
    temp = np.random.normal(45.0, 5.0, num_samples) # Operating Temp (C)
    voltage = np.random.normal(3.3, 0.1, num_samples) # Voltage (V)
    resistance = np.random.normal(10.0, 0.5, num_samples) # Resistance (Ohms)
    
    df = pd.DataFrame({
        'serial_number': [f"SN-{i:05d}" for i in range(num_samples)],
        'temp_c': temp,
        'voltage_v': voltage,
        'resistance_ohms': resistance
    })
    
    anomaly_indices = np.random.choice(num_samples, 20, replace=False)
    df.loc[anomaly_indices, 'temp_c'] = np.random.normal(80.0, 5.0, 20)
    df.loc[anomaly_indices, 'resistance_ohms'] = np.random.normal(13.0, 1.0, 20)
    
    return df

# 2. ML ANOMALY DETECTION (PREDICTIVE QUALITY)
def detect_anomalies(df):
    print("[INFO] Training ML Anomaly Detector (Isolation Forest)...")
    features = ['temp_c', 'voltage_v', 'resistance_ohms']
    
    # Train Isolation Forest
    model = IsolationForest(contamination=0.01, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[features])
    
    # -1 represents an anomaly, 1 represents normal
    failures = df[df['anomaly_score'] == -1].copy()
    return failures

# 3. AI / LLM ROOT CAUSE RECOMMENDATION SYSTEM
def agentic_rca_recommendation(row):
    """
    Simulates an LLM (e.g. Vertex AI / Claude) analyzing the specific 
    sensor values to recommend actionable troubleshooting steps.
    """
    if row['temp_c'] > 75 and row['resistance_ohms'] > 12:
        return "Thermal runaway detected correlated with resistance drift. Inspect SMT solder joints on power management IC for bridging or insufficient wetting."
    elif row['voltage_v'] < 3.1:
        return "Voltage drop anomaly. Check power rail capacitors for potential short or leakage."
    else:
        return "Multivariate variance detected. Run full diagnostic test suite."

def generate_insights(failures):
    print("[INFO] Flagging outliers and generating AI Root Cause recommendations...")
    print("-" * 50)
    
    failures['ai_recommendation'] = failures.apply(agentic_rca_recommendation, axis=1)
    
    for _, row in failures.head(3).iterrows():
        print(f"[ALERT] UNIT FAILED: {row['serial_number']}")
        print(f"Sensors: Temp={row['temp_c']:.1f}C | Voltage={row['voltage_v']:.2f}V | Resistance={row['resistance_ohms']:.1f}Ω")
        print(f"AI Recommendation: {row['ai_recommendation']}")
        print("-" * 50)

if __name__ == "__main__":
    hw_data = generate_manufacturing_data()
    failed_units = detect_anomalies(hw_data)
    generate_insights(failed_units)
