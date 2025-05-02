# utils.py
import streamlit as st

def preprocess_vitals(df):
    required_columns = ['systolic_bp', 'diastolic_bp', 'mean_bp']
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # continue if all columns are present
    X = df[required_columns]
    X = X.fillna(method='ffill').fillna(method='bfill')
    return X


