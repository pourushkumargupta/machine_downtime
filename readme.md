project:
  name: "Machine Downtime Prediction — End-to-End MLOps Pipeline"
  explanation: |
    This project demonstrates a complete MLOps workflow for predicting machine downtime in an industrial environment.
    The pipeline automates every step—data ingestion, preprocessing, model training, evaluation, data drift detection,
    retraining, and deployment—integrating multiple tools to ensure reproducibility, automation, and monitoring.

    The key objective is to create a self-learning predictive system that continuously monitors its performance,
    detects any drift or degradation in data quality, and automatically retrains itself to maintain accuracy.

tools_and_technologies:
  explanation: |
    The project combines several open-source tools for a seamless end-to-end ML lifecycle:

      • DVC (Data Version Control): Tracks data, models, and pipelines to ensure reproducibility.  
      • Evidently AI: Monitors data drift and model performance metrics automatically.  
      • Scikit-learn: Implements the core ML model (RandomForestClassifier).  
      • Streamlit / Flask: Used for interactive deployment and prediction interface.  
      • GitHub Actions / AWS CodePipeline: Enables CI/CD automation for model retraining and redeployment.  

folder_structure:
  explanation: |
    The folder structure ensures clarity, separation of responsibilities, and smooth automation across all stages.
    Each directory has a well-defined purpose, from raw data storage to deployment-ready model artifacts.

  structure:
    - path: "data/raw/"
      purpose: "Holds the original dataset (Machine_downtime.csv) before any processing."
    - path: "data/processed/"
      purpose: "Stores the cleaned dataset after preprocessing for training and evaluation."
    - path: "data/live/"
      purpose: "Contains live simulation data files (current_low, current_mid, current_high) used for drift detection."
    - path: "src/"
      purpose: "Source scripts for all pipeline stages: preprocessing, training, evaluation, and drift detection."
    - path: "models/"
      purpose: "Stores trained machine learning models such as model.pkl."
    - path: "reports/"
      purpose: "Saves generated drift reports and performance metrics (HTML and JSON formats)."
    - path: "app.py"
      purpose: "The Streamlit or Flask app file used for interactive deployment."
    - path: "dvc.yaml"
      purpose: "Defines the complete DVC pipeline including dependencies and outputs."
    - path: "params.yaml"
      purpose: "Central configuration file for parameters and paths used across all scripts."
    - path: "requirements.txt"
      purpose: "Lists Python dependencies required for this MLOps pipeline."
    - path: "drift_history.json"
      purpose: "Maintains a timestamped log of detected data drifts and retraining actions."

pipeline_description:
  step_1:
    name: "Data Ingestion & Preprocessing"
    explanation: |
      The first step of the pipeline reads the raw Machine_downtime.csv dataset from the `data/raw/` directory.
      It performs essential data cleaning operations such as handling missing values, encoding categorical features,
      and removing irrelevant columns. The cleaned dataset is stored in the `data/processed/train.csv` file.

      This ensures consistent, noise-free input for subsequent training stages and creates a standardized data version
      that can be tracked and reproduced using DVC.

    input_file: "data/raw/Machine_downtime.csv"
    script_used: "src/preprocess.py"
    output_file: "data/processed/train.csv"
    params_configuration: |
      preprocess:
        input_file: data/raw/Machine_downtime.csv
        output_file: data/processed/train.csv

  step_2:
    name: "Model Training"
    explanation: |
      This step is responsible for training the predictive model using the cleaned data.
      The script `src/train.py` utilizes the RandomForestClassifier algorithm due to its robustness,
      ability to handle nonlinear data, and interpretability.

      The parameters for training—such as the number of estimators, maximum depth, and test split—are controlled
      through `params.yaml`. The output of this stage is a serialized model (`model.pkl`) stored under the `models/` directory.

      Additionally, SMOTE (Synthetic Minority Oversampling Technique) can be applied to handle class imbalance
      between downtime and non-downtime instances.

    script: "src/train.py"
    model_algorithm: "RandomForestClassifier"
    parameters: |
      train:
        model_type: RandomForestClassifier
        test_size: 0.2
        random_state: 42
        smote: true
        n_estimators: 100
        max_depth: 10
    target_variable: "Downtime (1 = Failure, 0 = Non-Failure)"
    input_features:
      - "Hydraulic Pressure"
      - "Coolant Temperature"
      - "Tool Vibration"
      - "Torque"
      - "Spindle Speed"
      - "Voltage"
    output_model_file: "models/model.pkl"

  step_3:
    name: "Model Evaluation"
    explanation: |
      The evaluation phase measures the model’s performance on unseen data. The script `src/evaluate.py` loads the trained model
      and evaluates it using metrics like accuracy, precision, recall, and F1-score. These metrics are stored in JSON format
      within the `reports/` folder for tracking and comparison.

      This step ensures model reliability before deployment and establishes baseline metrics for drift detection.

    script_used: "src/evaluate.py"
    metrics_output_file: "reports/metrics.json"
    example_metrics: |
      accuracy: 0.83
      f1_score: 0.81
      precision: 0.82
      recall: 0.79

  step_4:
    name: "Continuous Data Drift Detection"
    explanation: |
      Drift detection ensures that the incoming live data distribution matches the training data.
      Using Evidently AI, this step compares the reference dataset (`train.csv`) with live data files (`current_low.csv`, `current_mid.csv`, `current_high.csv`).
      It automatically generates drift reports in HTML and JSON formats.

      When the percentage of drifted features exceeds a threshold (30% by default), the system triggers automatic retraining
      using DVC commands. All drift events and retraining timestamps are logged in `drift_history.json` and `logs.txt`.

    script_used: "src/detect_drift.py"
    tool: "Evidently AI"
    reference_dataset: "data/processed/train.csv"
    live_datasets:
      - "data/live/current_low.csv"
      - "data/live/current_mid.csv"
      - "data/live/current_high.csv"
    generated_reports:
      - "reports/drift_report_low.html"
      - "reports/drift_low.json"
    drift_logic: |
      - Drop irrelevant columns such as Date, Machine_ID, and Downtime.
      - Convert categorical variables to numeric values for comparison.
      - Use Evidently’s DataDriftPreset to compute drift share.
      - Categorize drift severity:
          🟢 Low (<10%)  → Stable
          🟡 Moderate (<30%) → Monitor
          🔴 High (≥30%)  → Retraining triggered.
    retraining_command: "os.system('dvc repro train')"
    logs:
      - "logs.txt"
      - "drift_history.json"

  step_5:
    name: "Model Deployment"
    explanation: |
      The deployment component exposes the trained model as an interactive web application
      built using Streamlit or Flask. Users can input real-time sensor values and view downtime predictions
      immediately.

      Additionally, SHAP visualizations explain which features contributed most to each prediction,
      enhancing transparency for operators and maintenance teams.

    file: "app.py"
    framework: "Streamlit / Flask"
    features:
      - "User-friendly parameter input form"
      - "Real-time downtime prediction"
      - "SHAP-based interpretability for model outputs"

dvc_pipeline:
  explanation: |
    The DVC pipeline defines a dependency graph for all steps—ensuring that only changed stages are re-executed.
    This structure makes the workflow modular and efficient. Each stage specifies its command, dependencies, and outputs.

  stages:
    preprocess:
      cmd: "python src/preprocess.py"
      deps:
        - "src/preprocess.py"
        - "data/raw/Machine_downtime.csv"
      outs:
        - "data/processed/train.csv"

    detect_drift:
      cmd: "python src/detect_drift.py"
      deps:
        - "src/detect_drift.py"
        - "data/processed/train.csv"
        - "data/live/current_high.csv"
      outs:
        - "reports/drift_report_high.html"
        - "drift_history.json"

    train:
      cmd: "python src/train.py"
      deps:
        - "data/processed/train.csv"
        - "src/train.py"
      outs:
        - "models/model.pkl"

    evaluate:
      cmd: "python src/evaluate.py"
      deps:
        - "data/processed/train.csv"
        - "models/model.pkl"
        - "src/evaluate.py"
      outs:
        - "reports/metrics.json"

ci_cd_automation:
  explanation: |
    The CI/CD setup ensures that every time new data or code is pushed, the entire MLOps workflow
    can automatically detect drift, retrain models, evaluate performance, and deploy updates.
    This makes the system self-healing and production-ready.

  flow:
    - "New data arrives → triggers drift detection"
    - "High drift → triggers retraining through DVC"
    - "Model evaluation updates the metrics"
    - "Streamlit app redeploys the new model automatically"
  tools:
    - "GitHub Actions"
    - "AWS CodePipeline"

logs_and_monitoring:
  explanation: |
    Logs and reports provide transparency for all retraining events and drift occurrences.
    The `drift_history.json` file stores cumulative drift summaries, while `logs.txt`
    records real-time operational details such as drift level and retraining triggers.
    Evidently-generated HTML reports visualize drift metrics in detail.

  files:
    - name: "logs.txt"
      purpose: "Chronological log of drift checks and retraining operations."
    - name: "drift_history.json"
      purpose: "Aggregated JSON record of drift percentages and retraining timestamps."
    - name: "reports/*.html"
      purpose: "Interactive visual drift reports created by Evidently AI."
    - name: "reports/metrics.json"
      purpose: "Latest model performance metrics post-retraining."

key_features:
  explanation: |
    The key highlights of the MLOps pipeline are:
      - Fully reproducible model lifecycle through DVC.
      - Automated drift detection and retraining using Evidently AI.
      - End-to-end traceability and logging for every operation.
      - Configurable parameters stored centrally in `params.yaml`.
      - Interactive and explainable deployment using Streamlit.

summary:
  explanation: |
    This project delivers a production-grade MLOps solution integrating data processing,
    drift monitoring, model training, evaluation, and deployment.
    It continuously adapts to new data, ensuring consistent model performance and transparency.
    By combining DVC, Evidently AI, and automation pipelines, it achieves end-to-end reliability
    and maintainability in predictive maintenance workflows.

author:
  name: "Pourush Kumar Gupta"
  role: "Machine Downtime Prediction | MLOps & Data Science | 2025"
