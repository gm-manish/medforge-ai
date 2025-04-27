# utils.py

def preprocess_vitals(df):
    features = ['HeartRate', 'SysBP', 'DiaBP', 'MeanBP', 'Temperature', 'SpO2', 'RespirationRate', 'CVP']
    X = df[features]
    X = X.fillna(method='ffill').fillna(method='bfill')
    return X
