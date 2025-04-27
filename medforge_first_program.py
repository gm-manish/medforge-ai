import os
import pandas as pd
import re
import numpy as np
import random

# SEED THE RANDOMNESS
np.random.seed(42)
random.seed(42)

# Path to your unzipped folder (update this if needed!)
base_path = "/Users/manishghimire/Downloads/mimic2cdb"

# Prepare
all_patient_data = []

# Go through all folders
for folder_name in os.listdir(base_path):
    if folder_name.startswith('s') and os.path.isdir(os.path.join(base_path, folder_name)):
        folder_path = os.path.join(base_path, folder_name)
        txt_file = folder_name + ".txt"
        txt_path = os.path.join(folder_path, txt_file)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    # Only parse lines starting with [timestamp] ch 
                    if "ch" in line:
                        match = re.search(r'id=(\d+).*?v1=([\d\.]+).*?u1=([a-zA-Z%]+)?', line)
                        if match:
                            id_val = match.group(1)
                            value = match.group(2)
                            unit = match.group(3) if match.group(3) else None
                            
                            # Optional: Capture v2 if available (diastolic BP maybe)
                            match_v2 = re.search(r'v2=([\d\.]+)', line)
                            v2_value = match_v2.group(1) if match_v2 else None

                            all_patient_data.append({
                                "patient_id": folder_name,
                                "file": txt_file,
                                "id": id_val,
                                "v1": value,
                                "u1": unit,
                                "v2": v2_value
                            })

# Final DataFrame
full_df = pd.DataFrame(all_patient_data)

# Save it!
full_df.to_csv("parsed_icu_patients.csv", index=False)
print("✅ Done parsing all patients. Saved as parsed_icu_patients.csv!")

# --------------------------
# MedForge ICU Data Mapper
# --------------------------
#%%
import pandas as pd

# Load your parsed CSV
input_csv = '/Users/manishghimire/parsed_icu_patients.csv'   
output_csv = '/Users/manishghimire/Documents/medforge_streamlit/mapped_patient_data.csv'   

# Read the CSV
df = pd.read_csv(input_csv)

# Define the ID-to-Concept dictionary
id_to_concept = {
    51: "Systolic BP (mmHg)",
    52: "Diastolic BP (mmHg)",
    55: "Mean Arterial Pressure (MAP, mmHg)",
    211: "Heart Rate (BPM)",
    646: "Oxygen Saturation (SpO2, %)",
    742: "Body Weight (kg)",
    69: "Height (meters)",
    762: "Body Weight (kg)",
    920: "Height (inches)",
    618: "Respiratory Rate (Breaths per minute)",
    676: "Temperature (Celsius)",
    679: "Temperature (Fahrenheit)",
    113: "Central Venous Pressure (CVP, mmHg)",
    492: "Pulmonary Artery Diastolic Pressure (mmHg)",
    224: "Unknown Pressure (maybe Central BP?)",
    225: "ECG Rhythm (1:1 means normal conduction)",
    778: "Systolic BP (Alternative)",
    779: "Diastolic BP (Alternative)",
    813: "Pulse Oximetry (%)"
}

# Map the concept
df['concept_name'] = df['id'].map(id_to_concept)

# Save the mapped file
df.to_csv(output_csv, index=False)

print("✅ Mapped CSV successfully saved at:", output_csv)
#%%

#%%
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Load your mapped CSV
mapped_file = '/Users/manishghimire/Documents/medforge_streamlit/mapped_patient_data.csv'  
df = pd.read_csv(mapped_file)

# 2. Keep only needed columns
df = df[['patient_id', 'file', 'id', 'v1', 'u1', 'v2']]

# 3. Define mappings from 'id' to feature names
id_to_feature = {
    51: ('systolic_bp', 'diastolic_bp'),  # v1 = systolic, v2 = diastolic
    52: ('mean_bp', None),
    55: ('diastolic_bp', None),
    211: ('heart_rate', None),
    646: ('spo2', None),
    676: ('temp_celsius', None),
    742: ('weight_kg', None),
    678: ('temp_fahrenheit', None),
    920: ('height_inches', None),
    618: ('resp_rate', None),
    492: ('cvp', None),
}

# 4. Create an empty list to hold the final timeline records
records = []

# 5. Generate synthetic timestamps
start_time = datetime(2025, 4, 1, 8, 0, 0)  # 8:00 AM

for patient_id, group in df.groupby('patient_id'):
    time = start_time
    patient_data = {}
    for idx, row in group.iterrows():
        feature = id_to_feature.get(row['id'], None)
        if feature:
            # New timestamp every 5 minutes
            time += timedelta(minutes=5)
            record = {
                'patient_id': patient_id,
                'timestamp': time.strftime("%Y-%m-%d %H:%M")
            }
            if feature[0]:
                record[feature[0]] = row['v1']
            if feature[1] and not pd.isna(row['v2']):
                record[feature[1]] = row['v2']
            records.append(record)

# 6. Create final dataframe
timeline_df = pd.DataFrame(records)

# 7. Aggregate so that same timestamp rows get merged
timeline_df = timeline_df.groupby(['patient_id', 'timestamp']).first().reset_index()

# 8. Save it
output_file = '/Users/manishghimire/Documents/medforge_streamlit/patient_timelines.csv'  # or wherever you want
timeline_df.to_csv(output_file, index=False)

print(f"✅ Timeline built successfully! File saved to {output_file}")
#%%
#%%
import pandas as pd
import numpy as np

# Load your merged DataFrame
df = pd.read_csv('/Users/manishghimire/Documents/medforge_streamlit/patient_timelines.csv')

# 1. Forward fill within each patient
df = df.sort_values(by=["patient_id", "timestamp"])  # Sort properly
df = df.groupby("patient_id").apply(lambda group: group.ffill()).reset_index(drop=True)

# 2. Add small random jitter
def add_jitter(series, jitter_range):
    noise = np.random.uniform(-jitter_range, jitter_range, size=series.shape)
    return series + noise

# Apply jitter only where it makes sense
df['heart_rate'] = add_jitter(df['heart_rate'], 2)
df['systolic_bp'] = add_jitter(df['systolic_bp'], 2)
df['diastolic_bp'] = add_jitter(df['diastolic_bp'], 2)
df['mean_bp'] = add_jitter(df['mean_bp'], 2)
df['temp_fahrenheit'] = add_jitter(df['temp_fahrenheit'], 0.2)
df['temp_celsius'] = add_jitter(df['temp_celsius'], 0.1)
df['resp_rate'] = add_jitter(df['resp_rate'], 1)
df['spo2'] = add_jitter(df['spo2'], 1)

# Clip physiological ranges to avoid crazy values
df['spo2'] = df['spo2'].clip(lower=85, upper=100)
df['heart_rate'] = df['heart_rate'].clip(lower=40, upper=180)
df['systolic_bp'] = df['systolic_bp'].clip(lower=70, upper=200)
df['diastolic_bp'] = df['diastolic_bp'].clip(lower=40, upper=120)
df['mean_bp'] = df['mean_bp'].clip(lower=50, upper=120)
df['resp_rate'] = df['resp_rate'].clip(lower=5, upper=40)
df['temp_fahrenheit'] = df['temp_fahrenheit'].clip(lower=95, upper=105)
df['temp_celsius'] = df['temp_celsius'].clip(lower=35, upper=40)

# Save clean filled + jittered file
output_path = '/Users/manishghimire/Documents/medforge_streamlit/cleaned_patient_data.csv'
df.to_csv(output_path, index=False)

print(f"✅ Saved cleaned file to {output_path}")
#%%
# Load your latest merged/filled file
file_path = '/Users/manishghimire/Documents/medforge_streamlit/cleaned_patient_data.csv'  # <<< put your latest filled file
df = pd.read_csv(file_path)

# Step 1: Calculate MAP if missing (optional if you want)
# MAP = (Systolic + 2*Diastolic) / 3
df['mean_bp'] = df['mean_bp'].fillna((df['systolic_bp'] + 2 * df['diastolic_bp']) / 3)

# Step 2: Create Hypotension Label
df['hypotension'] = ((df['mean_bp'] < 60) | (df['systolic_bp'] < 90)).astype(int)

# Step 3: (Optional) Save labeled dataset
output_file = '/Users/manishghimire/Documents/medforge_streamlit/labeled_patient_timelines.csv'
df.to_csv(output_file, index=False)

print("✅ Labeling complete. File saved:", output_file)
#%%
#%%

# Step 0: Load your final merged and filled dataset
file_path = '/Users/manishghimire/Documents/medforge_streamlit/labeled_patient_timelines.csv'  # <<< UPDATE THIS PATH
df = pd.read_csv(file_path)

# Step 1: Smart MAP calculation if mean_bp missing
# MAP = (Systolic BP + 2 × Diastolic BP) / 3
df['mean_bp'] = df['mean_bp'].fillna((df['systolic_bp'] + 2 * df['diastolic_bp']) / 3)

# Step 2: Create the Hypotension Label
df['hypotension'] = ((df['mean_bp'] < 60) | (df['systolic_bp'] < 90)).astype(int)

# Step 3: (Optional) Quick check
print(df[['patient_id', 'timestamp', 'systolic_bp', 'diastolic_bp', 'mean_bp', 'hypotension']].head(20))

# Step 4: Save the final file
output_file = '/Users/manishghimire/Documents/medforge_streamlit/labeled_patient_timelines.csv'  # <<< UPDATE IF NEEDED
df.to_csv(output_file, index=False)

print("✅ Labeling complete. Saved at:", output_file)
#%%
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your labeled data
df = pd.read_csv('/Users/manishghimire/Documents/medforge_streamlit/labeled_patient_timelines.csv')

# Drop rows with missing important vitals
df = df.dropna(subset=['systolic_bp', 'diastolic_bp', 'mean_bp'])

# Select features and target
X = df[['systolic_bp', 'diastolic_bp', 'mean_bp']]  # Features
y = df['hypotension']                               # Label (0 = no hypo, 1 = hypo)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
#%%
import numpy as np
import random

# SEED THE RANDOMNESS
np.random.seed(42)
random.seed(42)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your patient timeline data (after filling vitals and labels)
patient_data = pd.read_csv('/Users/manishghimire/Documents/medforge_streamlit/labeled_patient_timelines.csv')

# Drop rows where BP or mean BP is NaN
patient_data = patient_data.dropna(subset=['systolic_bp', 'diastolic_bp', 'mean_bp'])

# Features and Labels
X = patient_data[['systolic_bp', 'diastolic_bp', 'mean_bp']]
y = patient_data['hypotension']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
#%%
import joblib

# Save the model
joblib.dump(model, 'icu_hypotension_predictor.pkl')

print("✅ Model saved as 'icu_hypotension_predictor.pkl'")
#%%
joblib.dump(model, '/Users/manishghimire/Documents/medforge_streamlit/icu_hypotension_predictor.pkl')
#%%