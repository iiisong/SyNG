from river.drift import ADWIN

def detect_drift(feature_name, train_data, test_data, **kwargs):
    # Step 4: Initialize Drift Detector
    drift_detector = ADWIN(
        delta=kwargs.get('delta', .001), 
        min_window_length=kwargs.get('min_window_length', 500), 
        grace_period=kwargs.get('grace_period', 1000))

    # Step 5: Track Drift and Store Detected Points
    drift_points = []  # To store the indices where drift is detected

    for value in train_data[feature_name]:
        drift_detector.update(value)
        
    # Step 6: Feed Synthetic Data into Drift Detector (Comparing Only to Synthetic Past)
    for i, value in enumerate(test_data[feature_name]):
        drift_detector.update(value)
        if drift_detector.drift_detected:
            print(f"🚨 Drift detected at sample {len(train_data) + i+1}, value: {value}, variance of {drift_detector.variance}, estimate of {drift_detector.estimation}")
            
            drift_points.append(i)
            
    return drift_points