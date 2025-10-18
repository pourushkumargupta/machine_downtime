🧠 Machine Downtime Prediction — End-to-End MLOps Pipeline
│
├── 📘 Project Overview
│   │
│   ├── This project demonstrates a complete MLOps workflow for predicting machine downtime
│   │   in an industrial environment.
│   │
│   ├── The pipeline automates every step — data ingestion, preprocessing, model training,
│   │   evaluation, data drift detection, retraining, and deployment — using:
│   │       ├── DVC (Data Version Control) — reproducibility
│   │       ├── Evidently AI — continuous data drift monitoring
│   │       ├── Scikit-learn — machine learning models
│   │       └── Streamlit / Flask — deployment
│   │
│   └── The goal is to build a self-learning predictive system that monitors its performance,
│       detects when input data changes, and retrains automatically when required.
│
├── 📁 Folder Structure
│   │
│   ├── mlops/
│   │   │
│   │   ├── data/
│   │   │   ├── raw/                  # Original dataset (Machine_downtime.csv)
│   │   │   ├── processed/            # Cleaned dataset after preprocessing (train.csv)
│   │   │   └── live/                 # Live simulation data (current_low, current_mid, current_high)
│   │   │
│   │   ├── src/                      # Source scripts for each pipeline stage
│   │   │   ├── preprocess.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   └── detect_drift.py
│   │   │
│   │   ├── models/                   # Trained model files (model.pkl)
│   │   ├── reports/                  # Drift and metric reports (HTML/JSON)
│   │   ├── app.py                    # Streamlit/Flask app for deployment
│   │   ├── dvc.yaml                  # DVC pipeline definition
│   │   ├── params.yaml               # Central configuration file
│   │   ├── requirements.txt          # Python dependencies
│   │   └── drift_history.json        # Historical record of drift detections
│
├── ⚙️ Step-by-Step Pipeline Description
│
│   ├── 1️⃣ Data Ingestion & Preprocessing
│   │   │
│   │   ├── Raw dataset:
│   │   │       data/raw/Machine_downtime.csv
│   │   │
│   │   ├── Script used:
│   │   │       src/preprocess.py
│   │   │
│   │   ├── Description:
│   │   │       ├── Handles missing values
│   │   │       ├── Encodes categorical features
│   │   │       ├── Removes irrelevant columns
│   │   │       └── Saves cleaned dataset as data/processed/train.csv
│   │   │
│   │   └── Configuration (params.yaml):
│   │           preprocess:
│   │             input_file: data/raw/Machine_downtime.csv
│   │             output_file: data/processed/train.csv
│   │
│   ├── 2️⃣ Model Training
│   │   │
│   │   ├── Stage name: train
│   │   ├── Script: src/train.py
│   │   ├── Algorithm: RandomForestClassifier
│   │   │
│   │   ├── Parameters (params.yaml):
│   │   │       train:
│   │   │         model_type: RandomForestClassifier
│   │   │         test_size: 0.2
│   │   │         random_state: 42
│   │   │         smote: true
│   │   │         n_estimators: 100
│   │   │         max_depth: 10
│   │   │
│   │   ├── Target variable:
│   │   │       Downtime (1 = Failure, 0 = Non-Failure)
│   │   │
│   │   ├── Input features:
│   │   │       ├── Hydraulic Pressure
│   │   │       ├── Coolant Temperature
│   │   │       ├── Tool Vibration
│   │   │       ├── Torque
│   │   │       ├── Spindle Speed
│   │   │       └── Voltage
│   │   │
│   │   └── Output model file:
│   │           models/model.pkl
│   │
│   ├── 3️⃣ Model Evaluation
│   │   │
│   │   ├── Script used:
│   │   │       src/evaluate.py
│   │   │
│   │   ├── Process:
│   │   │       ├── Load data and trained model
│   │   │       ├── Split into 80/20 train-test
│   │   │       ├── Compute metrics (Accuracy, F1-score, Precision, Recall)
│   │   │       └── Save results to reports/metrics.json
│   │   │
│   │   └── Example output:
│   │           accuracy: 0.83
│   │           f1_score: 0.81
│   │           precision: 0.82
│   │           recall: 0.79
│   │
│   ├── 4️⃣ Continuous Data Drift Detection
│   │   │
│   │   ├── Script used:
│   │   │       src/detect_drift.py
│   │   │
│   │   ├── Tool: Evidently AI
│   │   │
│   │   ├── Reference dataset:
│   │   │       data/processed/train.csv
│   │   │
│   │   ├── Live datasets:
│   │   │       ├── data/live/current_low.csv
│   │   │       ├── data/live/current_mid.csv
│   │   │       └── data/live/current_high.csv
│   │   │
│   │   ├── Generated reports:
│   │   │       ├── reports/drift_report_low.html
│   │   │       └── reports/drift_low.json
│   │   │
│   │   ├── Drift detection process:
│   │   │       ├── Drop irrelevant columns (Date, Machine_ID, Downtime)
│   │   │       ├── Convert categorical → numeric
│   │   │       ├── Compute drift share using DataDriftPreset()
│   │   │       └── Categorize drift level:
│   │   │             🟢 Low (<10%) → Stable
│   │   │             🟡 Moderate (<30%) → Monitor
│   │   │             🔴 High (≥30%) → Retrain
│   │   │
│   │   ├── Automatic retraining command:
│   │   │       os.system("dvc repro train")
│   │   │
│   │   └── Logging:
│   │           ├── logs.txt
│   │           └── drift_history.json
│   │
│   ├── 5️⃣ Model Deployment
│   │   │
│   │   ├── File:
│   │   │       app.py
│   │   │
│   │   ├── Framework: Streamlit / Flask
│   │   │
│   │   └── Features:
│   │           ├── User-friendly parameter input form
│   │           ├── Real-time downtime prediction
│   │           └── SHAP visualizations for feature importance
│   │
│   └── 🔄 DVC Pipeline Definition
│           ├── preprocess → python src/preprocess.py
│           ├── detect_drift → python src/detect_drift.py
│           ├── train → python src/train.py
│           └── evaluate → python src/evaluate.py
│
├── ⚡ CI/CD Automation
│   │
│   ├── Pipeline flow:
│   │       ├── New data arrives → triggers drift detection
│   │       ├── High drift → triggers retraining
│   │       ├── Evaluation → updates metrics
│   │       └── Redeploys updated model via app.py
│   │
│   └── Supported automation:
│           ├── GitHub Actions
│           └── AWS CodePipeline
│
├── 🧾 Logs & Monitoring
│   │
│   ├── logs.txt               — Timestamped drift and retraining logs
│   ├── drift_history.json     — Historical drift summary
│   ├── reports/*.html         — Evidently drift visualization reports
│   └── reports/metrics.json   — Latest evaluation metrics
│
├── 🧰 Technologies Used
│   │
│   ├── Data Handling: Pandas, NumPy
│   ├── Modeling: Scikit-learn (RandomForestClassifier), SMOTE
│   ├── Monitoring: Evidently AI
│   ├── Version Control: DVC
│   ├── Storage: Local / MySQL
│   ├── Deployment: Streamlit / Flask
│   └── Automation: GitHub Actions / AWS CodePipeline
│
├── 🎯 Key Features
│   │
│   ├── ✅ Fully reproducible ML pipeline using DVC
│   ├── ✅ Automated drift detection with Evidently AI
│   ├── ✅ Self-healing retraining loop
│   ├── ✅ Centralized configuration via params.yaml
│   ├── ✅ Transparent logs and drift reports
│   └── ✅ Real-time prediction UI with SHAP explainability
│
├── 🧩 Summary
│   │
│   ├── This project delivers a production-ready MLOps pipeline that connects data,
│   │   model, and deployment in a continuous feedback loop.
│   │
│   ├── It monitors incoming data, detects drift, retrains automatically, and redeploys
│   │   the model.
│   │
│   └── By integrating DVC and Evidently AI, it ensures reproducibility, transparency,
│       and long-term adaptability for predictive maintenance.
│
└── 👨‍💻 Author
    │
    ├── Pourush Kumar Gupta
    └── Machine Downtime Prediction | MLOps & Data Science | 2025
