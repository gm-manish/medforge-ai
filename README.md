# MedForge AI 🏥🤖

**Live App:** [https://medforge.streamlit.app](https://medforge.streamlit.app)  
**GitHub Repo:** [https://github.com/gm-manish/medforge-ai](https://github.com/gm-manish/medforge-ai)

---

## 🚀 Project Overview

**MedForge AI** is a real-time, AI-in-the-loop platform for:
- ✨ **Generating synthetic, HIPAA-safe patient timelines** (e.g., Hypertension, Diabetes, Depression)
- 🔢 **Predicting hypotension risk** from ICU vitals using a trained machine learning model

This app empowers researchers, developers, and clinicians to safely prototype and validate EHR-based decision tools without exposing real patient data.

---

## 🏛️ Features

### 1. 🧬 Synthetic Cohort Generator
- Choose disease profiles like Hypertension, Depression, Heart Failure
- Generate realistic 30-day patient timelines (vitals, notes, medication starts)
- Download results as clean CSV

### 2. 📊 Hypotension Prediction Engine
- Upload real or synthetic ICU vitals
- Supported features:
  - `systolic_bp`, `diastolic_bp`, `mean_bp`
- Predicts hypotension risk (`0 = no`, `1 = yes`)
- Risky rows highlighted in red for easy triage

---

## 🔧 How It Works

1. A trained RandomForestClassifier model (on labeled ICU timelines) is stored as `icu_hypotension_predictor.pkl`
2. `utils.py` handles preprocessing (missing values, feature alignment)
3. Streamlit handles the frontend interface with interactive tabs, tables, and buttons

---

## 📚 Tech Stack

- Python ✨
- Streamlit (frontend)
- Pandas / scikit-learn (modeling)
- joblib (model persistence)

---

## 🔍 Sample CSV Format (for prediction)

```csv
systolic_bp,diastolic_bp,mean_bp
140,90,106.7
120,80,93.3
100,60,73.3
```

---

## 🚀 What's Next

- [ ] 🔊 Add Sepsis and Cardiac Arrest early-warning models
- [ ] 🌎 Integrate FHIR-compatible data ingestion
- [ ] 📈 Streamlit dashboard for multi-patient cohort monitoring
- [ ] 🔧 Build a MedForge API to embed into hospital systems

---

## 🤝 Want to Collaborate?

I'm looking for co-founders, healthcare AI researchers, and early users! Let's turn MedForge into the future of simulation-first clinical AI.

Connect with me on [LinkedIn](https://www.linkedin.com/in/manishghimire) or shoot a message — let's build.

---

## 📄 License

MIT License. Use freely and ethically. Clinical use not validated.

