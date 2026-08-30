# AeroPredict AI

### Machine Learning-Based Aerospace Telemetry Anomaly Detection

AeroPredict AI is an end-to-end machine learning system designed to identify abnormal patterns in simulated aerospace flight telemetry.

The project combines **telemetry simulation, domain-informed feature engineering, unsupervised anomaly detection, data-quality monitoring, and independent model evaluation** into a single analytical workflow.

## Key Capabilities

* Simulated aerospace flight telemetry across multiple mission phases
* Expected-vs-observed telemetry deviation analysis
* Temporal and rolling-statistical feature engineering
* Data-quality monitoring for missing, invalid, and sudden sensor changes
* Real-time anomaly detection during mission simulation
* Independent evaluation using controlled fault injection
* Comparison of multiple unsupervised machine learning algorithms
* Feature-level analysis of detected anomalies

## Machine Learning Approach

### Feature Engineering

Telemetry data is transformed into features capturing both deviation from expected flight behaviour and temporal dynamics:

* Telemetry deviations
* Rate-of-change features
* Rolling mean and standard deviation
* Dynamic stress indicators
* Thermal stress indicators
* Trajectory deviation indicators

### Anomaly Detection

Three unsupervised learning algorithms are trained on nominal flight telemetry:

| Model                | Approach                                           |
| -------------------- | -------------------------------------------------- |
| Isolation Forest     | Tree-based isolation of anomalous observations     |
| One-Class SVM        | Learns the boundary of nominal telemetry behaviour |
| Local Outlier Factor | Detects observations with abnormal local density   |

The models are trained using nominal mission sequences and evaluated separately on an independent mission containing controlled anomaly scenarios.

## Evaluation

Model performance is assessed using:

* Precision
* Recall
* F1 Score
* ROC-AUC
* False Positive Rate
* Confusion Matrix

The system supports evaluation of simulated fault scenarios including:

* Propulsion degradation
* Thermal events
* Trajectory deviation
* Sensor failure
* Compound anomalies

## Technology Stack

**Programming:** Python

**Machine Learning:** scikit-learn

**Data Processing:** Pandas, NumPy

**Visualisation:** Plotly

**Application:** Streamlit

**Model Persistence:** Joblib

## Project Workflow

```text
Telemetry Simulation
        ↓
Expected vs Observed Behaviour
        ↓
Feature Engineering
        ↓
Data Quality Checks
        ↓
Feature Scaling
        ↓
Unsupervised ML Models
        ↓
Anomaly Detection
        ↓
Independent Evaluation
        ↓
Model Comparison & Analysis
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/notacsgeeeeek/AeroPredict-AI.git
cd AeroPredict-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
AeroPredict-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── screenshots/
    ├── dashboard.png
    └── model_evaluation.png
```

## Scope

AeroPredict AI uses **simulated telemetry and controlled fault injection** for research and demonstration purposes. It is intended as a machine learning and data-science prototype rather than a certified aerospace flight-control or safety system.


