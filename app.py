import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroPredict AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL TYPOGRAPHY
       ======================================================== */

    html,
    body,
    [class*="css"],
    [data-testid="stAppViewContainer"] {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 750;
        letter-spacing: -1px;
        margin-top: 0.2rem;
        margin-bottom: 0.35rem;
        line-height: 1.15;
    }

    .main-subtitle {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: 0.1px;
    }

    .main-description {
        text-align: center;
        font-size: 1.02rem;
        line-height: 1.6;
        max-width: 900px;
        margin: 0 auto 2rem auto;
    }

    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        font-size: 3rem !important;
        font-weight: 750 !important;
        text-align: center !important;
        letter-spacing: -0.8px !important;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px !important;
        margin-top: 1.6rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 1.35rem !important;
        font-weight: 650 !important;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        width: 340px !important;
    }

    section[data-testid="stSidebar"] * {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] p {
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
    }

    section[data-testid="stSidebar"] label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stSlider label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] input {
        font-size: 1rem !important;
    }

    section[data-testid="stSidebar"] button {
        font-size: 1rem !important;
        font-weight: 650 !important;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.95rem !important;
    }

    /* ========================================================
       DATAFRAMES
       ======================================================== */

    [data-testid="stDataFrame"] {
        font-size: 0.98rem !important;
    }

    /* ========================================================
       CAPTIONS
       ======================================================== */

    .stCaption {
        font-size: 0.95rem !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton button {
        font-size: 1rem !important;
        font-weight: 650 !important;
        padding: 0.55rem 1rem !important;
    }

    /* ========================================================
       SELECT BOX / RADIO
       ======================================================== */

    .stSelectbox div,
    .stRadio div {
        font-size: 1rem !important;
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        font-size: 1rem !important;
    }

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* ========================================================
       ML OVERVIEW CARDS
       ======================================================== */

    .overview-card {
        padding: 1.2rem 1.3rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 145px;
    }

    .overview-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .overview-card-item {
        font-size: 0.98rem;
        line-height: 1.7;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {
        text-align: center;
        font-size: 0.9rem;
        opacity: 0.75;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIRECTORY = "saved_models"

MODEL_FILE = os.path.join(
    MODEL_DIRECTORY,
    "aeropredict_models.joblib"
)

SCALER_FILE = os.path.join(
    MODEL_DIRECTORY,
    "aeropredict_scaler.joblib"
)

METADATA_FILE = os.path.join(
    MODEL_DIRECTORY,
    "aeropredict_metadata.joblib"
)


FEATURE_COLUMNS = [
    "Altitude_Deviation",
    "Velocity_Deviation",
    "Acceleration_Deviation",
    "Temperature_Deviation",
    "Pressure_Deviation",
    "Orientation_Deviation",
    "Vibration",

    "Altitude_Rate",
    "Velocity_Rate",
    "Acceleration_Rate",
    "Temperature_Rate",
    "Pressure_Rate",
    "Orientation_Rate",
    "Vibration_Rate",

    "Temperature_RollingMean",
    "Temperature_RollingStd",

    "Vibration_RollingMean",
    "Vibration_RollingStd",

    "Acceleration_RollingMean",
    "Acceleration_RollingStd",

    "Dynamic_Stress_Index",
    "Thermal_Stress_Index",
    "Trajectory_Deviation_Index"
]


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 AeroPredict AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Intelligent Aerospace Telemetry & ML-Based Anomaly Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'An end-to-end machine learning system for aerospace telemetry '
    'monitoring, anomaly detection, model comparison and real-time inference.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🛰️ Mission Control"
)

st.sidebar.markdown(
    "Telemetry Analytics & ML Monitoring System"
)

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Live Mission",
        "ML Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### Mission Configuration"
)

mission_duration = st.sidebar.slider(
    "Mission Duration (seconds)",
    min_value=60,
    max_value=240,
    value=120,
    step=10
)

random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1
)

st.sidebar.markdown(
    "### Mission Scenario"
)

anomaly_type = st.sidebar.selectbox(
    "Injected Fault Scenario",
    [
        "None",
        "Propulsion Degradation",
        "Thermal Event",
        "Trajectory Deviation",
        "Sensor Failure",
        "Compound Anomaly"
    ]
)

st.sidebar.markdown("---")

retrain_models = st.sidebar.button(
    "🔄 Retrain ML Models",
    use_container_width=True
)

st.sidebar.caption(
    "Models are trained on nominal telemetry and "
    "persisted locally for reusable inference."
)


# ============================================================
# EXPECTED FLIGHT PROFILE
# ============================================================

def expected_flight_profile(t):

    if t < 20:
        altitude = 4.5 * t ** 2

    elif t < 60:
        altitude = 1800 + 170 * (t - 20)

    elif t < 100:
        altitude = 8600 + 85 * (t - 60)

    else:
        altitude = 12000 + 35 * (t - 100)

    if t < 20:
        velocity = 10 * t

    elif t < 60:
        velocity = 200 + 5.5 * (t - 20)

    elif t < 100:
        velocity = 420 + 2.0 * (t - 60)

    else:
        velocity = 500 - 0.8 * (t - 100)

    if t < 20:
        acceleration = 8.5

    elif t < 60:
        acceleration = 6.0

    elif t < 100:
        acceleration = 3.5

    else:
        acceleration = 1.5

    temperature = (
        25
        + min(t * 0.35, 50)
        + np.sin(t / 8) * 1.5
    )

    pressure = max(
        0.8,
        5.5 - altitude / 3500
    )

    fuel = max(
        0,
        100 - (t / 180) * 90
    )

    orientation = np.sin(t / 20) * 1.5

    vibration = (
        1.5
        + np.sin(t / 7) * 0.15
    )

    return {
        "Altitude": altitude,
        "Velocity": velocity,
        "Acceleration": acceleration,
        "Temperature": temperature,
        "Pressure": pressure,
        "Fuel": fuel,
        "Orientation": orientation,
        "Vibration": vibration
    }


# ============================================================
# NORMAL TELEMETRY GENERATOR
# ============================================================

def generate_normal_mission(
    duration,
    seed
):

    rng = np.random.default_rng(seed)

    rows = []

    for t in range(duration):

        expected = expected_flight_profile(t)

        observed = expected.copy()

        observed["Altitude"] += rng.normal(0, 12)
        observed["Velocity"] += rng.normal(0, 2.5)
        observed["Acceleration"] += rng.normal(0, 0.2)
        observed["Temperature"] += rng.normal(0, 0.8)
        observed["Pressure"] += rng.normal(0, 0.05)
        observed["Fuel"] += rng.normal(0, 0.25)
        observed["Orientation"] += rng.normal(0, 0.25)
        observed["Vibration"] += rng.normal(0, 0.1)

        rows.append(
            {
                "Time": t,

                "Altitude": observed["Altitude"],
                "Expected_Altitude": expected["Altitude"],

                "Velocity": observed["Velocity"],
                "Expected_Velocity": expected["Velocity"],

                "Acceleration": observed["Acceleration"],
                "Expected_Acceleration": expected["Acceleration"],

                "Temperature": observed["Temperature"],
                "Expected_Temperature": expected["Temperature"],

                "Pressure": observed["Pressure"],
                "Expected_Pressure": expected["Pressure"],

                "Fuel": observed["Fuel"],

                "Orientation": observed["Orientation"],
                "Expected_Orientation": expected["Orientation"],

                "Vibration": observed["Vibration"],

                "Anomaly": 0,
                "Anomaly_Type": "Normal"
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MISSION GENERATOR WITH FAULT INJECTION
# ============================================================

def generate_mission(
    duration,
    anomaly_type,
    seed
):

    rng = np.random.default_rng(seed)

    rows = []

    anomaly_start = int(duration * 0.55)
    anomaly_end = int(duration * 0.75)

    for t in range(duration):

        expected = expected_flight_profile(t)

        observed = expected.copy()

        anomaly = 0
        anomaly_name = "Normal"

        observed["Altitude"] += rng.normal(0, 12)
        observed["Velocity"] += rng.normal(0, 2.5)
        observed["Acceleration"] += rng.normal(0, 0.2)
        observed["Temperature"] += rng.normal(0, 0.8)
        observed["Pressure"] += rng.normal(0, 0.05)
        observed["Fuel"] += rng.normal(0, 0.25)
        observed["Orientation"] += rng.normal(0, 0.25)
        observed["Vibration"] += rng.normal(0, 0.1)

        if (
            anomaly_type != "None"
            and anomaly_start <= t <= anomaly_end
        ):

            anomaly = 1
            anomaly_name = anomaly_type

            if anomaly_type == "Propulsion Degradation":

                observed["Acceleration"] *= 0.55
                observed["Velocity"] *= 0.90
                observed["Vibration"] += 1.5

            elif anomaly_type == "Thermal Event":

                observed["Temperature"] += (
                    25
                    + (t - anomaly_start) * 0.25
                )

                observed["Vibration"] += 0.5

            elif anomaly_type == "Trajectory Deviation":

                observed["Orientation"] += (
                    8
                    + (t - anomaly_start) * 0.12
                )

                observed["Altitude"] *= 0.91
                observed["Velocity"] *= 0.95

            elif anomaly_type == "Sensor Failure":

                observed["Pressure"] += rng.normal(
                    3,
                    0.6
                )

                observed["Temperature"] += rng.normal(
                    18,
                    4
                )

                observed["Orientation"] += rng.normal(
                    10,
                    3
                )

            elif anomaly_type == "Compound Anomaly":

                observed["Acceleration"] *= 0.55
                observed["Velocity"] *= 0.88
                observed["Temperature"] += 20
                observed["Orientation"] += 8
                observed["Vibration"] += 2

        observed["Fuel"] = max(
            0,
            min(100, observed["Fuel"])
        )

        observed["Pressure"] = max(
            0.1,
            observed["Pressure"]
        )

        observed["Vibration"] = max(
            0,
            observed["Vibration"]
        )

        rows.append(
            {
                "Time": t,

                "Altitude": observed["Altitude"],
                "Expected_Altitude": expected["Altitude"],

                "Velocity": observed["Velocity"],
                "Expected_Velocity": expected["Velocity"],

                "Acceleration": observed["Acceleration"],
                "Expected_Acceleration": expected["Acceleration"],

                "Temperature": observed["Temperature"],
                "Expected_Temperature": expected["Temperature"],

                "Pressure": observed["Pressure"],
                "Expected_Pressure": expected["Pressure"],

                "Fuel": observed["Fuel"],

                "Orientation": observed["Orientation"],
                "Expected_Orientation": expected["Orientation"],

                "Vibration": observed["Vibration"],

                "Anomaly": anomaly,
                "Anomaly_Type": anomaly_name
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DATA QUALITY MONITORING
# ============================================================

def assess_data_quality(
    current_data,
    previous_data=None
):

    issues = []

    row = current_data.iloc[0]

    required_columns = [
        "Altitude",
        "Velocity",
        "Acceleration",
        "Temperature",
        "Pressure",
        "Fuel",
        "Orientation",
        "Vibration"
    ]

    missing_count = 0

    for column in required_columns:

        if pd.isna(row[column]):

            missing_count += 1

            issues.append(
                f"Missing value detected: {column}"
            )

    range_rules = {
        "Altitude": (-100, 100000),
        "Velocity": (-50, 10000),
        "Acceleration": (-30, 50),
        "Temperature": (-80, 250),
        "Pressure": (0, 20),
        "Fuel": (0, 100),
        "Orientation": (-180, 180),
        "Vibration": (0, 20)
    }

    out_of_range_count = 0

    for column, limits in range_rules.items():

        minimum = limits[0]
        maximum = limits[1]

        if (
            not pd.isna(row[column])
            and (
                row[column] < minimum
                or row[column] > maximum
            )
        ):

            out_of_range_count += 1

            issues.append(
                f"Out-of-range reading: {column}"
            )

    sudden_change_count = 0

    if previous_data is not None:

        previous_row = previous_data.iloc[-1]

        change_thresholds = {
            "Altitude": 500,
            "Velocity": 80,
            "Acceleration": 8,
            "Temperature": 15,
            "Pressure": 2,
            "Orientation": 20,
            "Vibration": 4
        }

        for column, threshold in change_thresholds.items():

            difference = abs(
                row[column]
                - previous_row[column]
            )

            if difference > threshold:

                sudden_change_count += 1

                issues.append(
                    f"Sudden change detected: {column}"
                )

    quality_score = (
        100
        - missing_count * 30
        - out_of_range_count * 20
        - sudden_change_count * 10
    )

    quality_score = max(
        0,
        quality_score
    )

    if quality_score >= 90:
        quality_status = "Good"

    elif quality_score >= 70:
        quality_status = "Warning"

    else:
        quality_status = "Poor"

    return {
        "Quality_Score": quality_score,
        "Quality_Status": quality_status,
        "Missing_Values": missing_count,
        "Out_of_Range": out_of_range_count,
        "Sudden_Changes": sudden_change_count,
        "Issues": issues
    }


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):

    data = df.copy()

    data["Altitude_Deviation"] = (
        data["Altitude"]
        - data["Expected_Altitude"]
    )

    data["Velocity_Deviation"] = (
        data["Velocity"]
        - data["Expected_Velocity"]
    )

    data["Acceleration_Deviation"] = (
        data["Acceleration"]
        - data["Expected_Acceleration"]
    )

    data["Temperature_Deviation"] = (
        data["Temperature"]
        - data["Expected_Temperature"]
    )

    data["Pressure_Deviation"] = (
        data["Pressure"]
        - data["Expected_Pressure"]
    )

    data["Orientation_Deviation"] = (
        data["Orientation"]
        - data["Expected_Orientation"]
    )

    sensors = [
        "Altitude",
        "Velocity",
        "Acceleration",
        "Temperature",
        "Pressure",
        "Orientation",
        "Vibration"
    ]

    for sensor in sensors:

        data[f"{sensor}_Rate"] = (
            data[sensor]
            .diff()
            .fillna(0)
        )

    rolling_sensors = [
        "Temperature",
        "Vibration",
        "Acceleration"
    ]

    for sensor in rolling_sensors:

        data[
            f"{sensor}_RollingMean"
        ] = (
            data[sensor]
            .rolling(
                window=5,
                min_periods=1
            )
            .mean()
        )

        data[
            f"{sensor}_RollingStd"
        ] = (
            data[sensor]
            .rolling(
                window=5,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

    data["Dynamic_Stress_Index"] = (
        abs(data["Acceleration_Deviation"])
        +
        abs(data["Velocity_Deviation"]) / 10
        +
        abs(data["Vibration"] - 1.5)
    )

    data["Thermal_Stress_Index"] = (
        abs(data["Temperature_Deviation"])
        +
        data["Vibration"] * 2
    )

    data["Trajectory_Deviation_Index"] = (
        abs(data["Altitude_Deviation"]) / 100
        +
        abs(data["Orientation_Deviation"]) * 2
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data[FEATURE_COLUMNS] = (
        data[FEATURE_COLUMNS]
        .ffill()
        .fillna(0)
    )

    return data


# ============================================================
# TRAINING DATASET
# ============================================================

def generate_training_dataset(
    duration,
    seed
):

    missions = []

    for i in range(10):

        mission = generate_normal_mission(
            duration,
            seed + 100 + i
        )

        missions.append(mission)

    return pd.concat(
        missions,
        ignore_index=True
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def create_and_train_models(
    duration,
    seed
):

    training_data = generate_training_dataset(
        duration,
        seed
    )

    training_features = engineer_features(
        training_data
    )

    X_train = training_features[
        FEATURE_COLUMNS
    ]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination=0.08,
        random_state=42,
        n_jobs=-1
    )

    isolation_forest.fit(
        X_train_scaled
    )

    one_class_svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.08
    )

    one_class_svm.fit(
        X_train_scaled
    )

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.08,
        novelty=True
    )

    lof.fit(
        X_train_scaled
    )

    models = {
        "Isolation Forest": isolation_forest,
        "One-Class SVM": one_class_svm,
        "Local Outlier Factor": lof
    }

    metadata = {
        "training_samples": len(training_data),
        "feature_count": len(FEATURE_COLUMNS),
        "training_seed": seed,
        "mission_duration": duration,
        "training_missions": 10
    }

    return (
        models,
        scaler,
        metadata
    )


# ============================================================
# SAVE ML PIPELINE
# ============================================================

def save_ml_pipeline(
    models,
    scaler,
    metadata
):

    os.makedirs(
        MODEL_DIRECTORY,
        exist_ok=True
    )

    joblib.dump(
        models,
        MODEL_FILE
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    joblib.dump(
        metadata,
        METADATA_FILE
    )


# ============================================================
# LOAD ML PIPELINE
# ============================================================

def load_ml_pipeline():

    if (
        os.path.exists(MODEL_FILE)
        and os.path.exists(SCALER_FILE)
        and os.path.exists(METADATA_FILE)
    ):

        models = joblib.load(
            MODEL_FILE
        )

        scaler = joblib.load(
            SCALER_FILE
        )

        metadata = joblib.load(
            METADATA_FILE
        )

        # Backward compatibility
        metadata.setdefault(
            "training_missions",
            10
        )

        metadata.setdefault(
            "feature_count",
            len(FEATURE_COLUMNS)
        )

        return (
            models,
            scaler,
            metadata
        )

    return None


# ============================================================
# LOAD OR TRAIN PIPELINE
# ============================================================

@st.cache_resource
def get_ml_pipeline(
    duration,
    seed,
    force_retrain=False
):

    if not force_retrain:

        saved_pipeline = load_ml_pipeline()

        if saved_pipeline is not None:

            return saved_pipeline

    models, scaler, metadata = (
        create_and_train_models(
            duration,
            seed
        )
    )

    save_ml_pipeline(
        models,
        scaler,
        metadata
    )

    return (
        models,
        scaler,
        metadata
    )


if retrain_models:

    st.cache_resource.clear()

    models, scaler, metadata = get_ml_pipeline(
        mission_duration,
        random_seed,
        force_retrain=True
    )

    st.sidebar.success(
        "ML models retrained successfully."
    )

else:

    models, scaler, metadata = get_ml_pipeline(
        mission_duration,
        random_seed
    )


# ============================================================
# ANOMALY SEVERITY
# ============================================================

def determine_severity(
    anomaly_prediction,
    anomaly_score,
    quality_score
):

    if anomaly_prediction == 0:

        return "Normal"

    if (
        anomaly_score >= 0.20
        or quality_score < 50
    ):

        return "Critical"

    return "Warning"


# ============================================================
# ANOMALY EXPLANATION
# ============================================================

def explain_anomaly(
    X_current,
    scaler,
    top_n=5
):

    scaled_values = scaler.transform(
        X_current
    )[0]

    contributions = np.abs(
        scaled_values
    )

    explanation_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Standardized Deviation": scaled_values,
            "Absolute Contribution": contributions
        }
    )

    return (
        explanation_df
        .sort_values(
            "Absolute Contribution",
            ascending=False
        )
        .head(top_n)
    )


# ============================================================
# LIVE MISSION MODE
# ============================================================

if analysis_mode == "Live Mission":

    st.markdown(
        '<h2 style="text-align:center;">🛰️ Live ML Mission</h2>',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Select ML Model",
        list(models.keys())
    )

    model = models[selected_model]

    st.markdown("---")

    info_col1, info_col2, info_col3 = st.columns(3)

    info_col1.metric(
        "Training Samples",
        metadata.get(
            "training_samples",
            0
        )
    )

    info_col2.metric(
        "Engineered Features",
        metadata.get(
            "feature_count",
            len(FEATURE_COLUMNS)
        )
    )

    info_col3.metric(
        "Inference Model",
        selected_model
    )

    st.markdown("---")

    start_mission = st.button(
        "🚀 Start ML Mission",
        use_container_width=True
    )

    if start_mission:

        mission_data = generate_mission(
            mission_duration,
            anomaly_type,
            random_seed
        )

        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        quality_placeholder = st.empty()
        telemetry_placeholder = st.empty()
        anomaly_placeholder = st.empty()
        explanation_placeholder = st.empty()
        event_placeholder = st.empty()

        live_rows = []
        detected_events = []
        previous_data = None

        for i in range(mission_duration):

            current_raw = (
                mission_data
                .iloc[i]
                .to_frame()
                .T
            )

            quality_result = assess_data_quality(
                current_raw,
                previous_data
            )

            partial_data = (
                mission_data
                .iloc[:i + 1]
                .copy()
            )

            partial_features = engineer_features(
                partial_data
            )

            current_features = (
                partial_features
                .iloc[-1]
            )

            X_current = (
                current_features[
                    FEATURE_COLUMNS
                ]
                .to_frame()
                .T
            )

            X_scaled = scaler.transform(
                X_current
            )

            prediction = model.predict(
                X_scaled
            )[0]

            anomaly_prediction = (
                1
                if prediction == -1
                else 0
            )

            anomaly_score = float(
                -model.decision_function(
                    X_scaled
                )[0]
            )

            severity = determine_severity(
                anomaly_prediction,
                anomaly_score,
                quality_result["Quality_Score"]
            )

            explanation_df = explain_anomaly(
                X_current,
                scaler
            )

            row = {
                "Time": i,
                "Altitude": current_raw["Altitude"].iloc[0],
                "Velocity": current_raw["Velocity"].iloc[0],
                "Acceleration": current_raw["Acceleration"].iloc[0],
                "Temperature": current_raw["Temperature"].iloc[0],
                "Pressure": current_raw["Pressure"].iloc[0],
                "Fuel": current_raw["Fuel"].iloc[0],
                "Orientation": current_raw["Orientation"].iloc[0],
                "Vibration": current_raw["Vibration"].iloc[0],
                "AI_Anomaly": anomaly_prediction,
                "Anomaly_Score": anomaly_score,
                "Severity": severity,
                "Data_Quality_Score": quality_result["Quality_Score"],
                "Data_Quality_Status": quality_result["Quality_Status"]
            }

            live_rows.append(row)

            if anomaly_prediction == 1:

                detected_events.append(
                    {
                        "Time": i,
                        "Severity": severity,
                        "Anomaly Score": anomaly_score,
                        "Data Quality": quality_result[
                            "Quality_Status"
                        ],
                        "Primary Contributor": explanation_df.iloc[0][
                            "Feature"
                        ],
                        "Injected Scenario": anomaly_type
                    }
                )

            live_df = pd.DataFrame(
                live_rows
            )

            if severity == "Critical":

                status_placeholder.error(
                    "🔴 CRITICAL AI ANOMALY DETECTED"
                )

            elif severity == "Warning":

                status_placeholder.warning(
                    "🟠 WARNING: AI ANOMALY DETECTED"
                )

            else:

                status_placeholder.success(
                    "🟢 AI CLASSIFICATION: NORMAL"
                )

            with metrics_placeholder.container():

                col1, col2, col3, col4, col5, col6 = (
                    st.columns(6)
                )

                col1.metric(
                    "Mission Time",
                    f"T+{i}s"
                )

                col2.metric(
                    "Altitude",
                    f"{row['Altitude']:.0f} m"
                )

                col3.metric(
                    "Velocity",
                    f"{row['Velocity']:.1f}"
                )

                col4.metric(
                    "Temperature",
                    f"{row['Temperature']:.1f} °C"
                )

                col5.metric(
                    "Anomaly Score",
                    f"{anomaly_score:.3f}"
                )

                col6.metric(
                    "Severity",
                    severity
                )

            with quality_placeholder.container():

                st.markdown(
                    '<h3 style="text-align:center;">'
                    '🔍 Telemetry Data Quality'
                    '</h3>',
                    unsafe_allow_html=True
                )

                q1, q2, q3, q4 = st.columns(4)

                q1.metric(
                    "Quality Score",
                    f"{quality_result['Quality_Score']:.0f}/100"
                )

                q2.metric(
                    "Status",
                    quality_result["Quality_Status"]
                )

                q3.metric(
                    "Missing Values",
                    quality_result["Missing_Values"]
                )

                q4.metric(
                    "Sudden Changes",
                    quality_result["Sudden_Changes"]
                )

                if quality_result["Issues"]:

                    st.warning(
                        " | ".join(
                            quality_result["Issues"]
                        )
                    )

            # ------------------------------------------------
            # LIVE TELEMETRY CHART
            # ------------------------------------------------

            telemetry_fig = go.Figure()

            telemetry_fig.add_trace(
                go.Scatter(
                    x=live_df["Time"],
                    y=live_df["Altitude"],
                    mode="lines",
                    name="Altitude"
                )
            )

            telemetry_fig.add_trace(
                go.Scatter(
                    x=live_df["Time"],
                    y=live_df["Velocity"],
                    mode="lines",
                    name="Velocity",
                    yaxis="y2"
                )
            )

            telemetry_fig.update_layout(
                title="Live Flight Telemetry",
                xaxis_title="Mission Time (s)",
                yaxis_title="Altitude (m)",
                yaxis2=dict(
                    title="Velocity",
                    overlaying="y",
                    side="right"
                ),
                height=400
            )

            telemetry_placeholder.plotly_chart(
                telemetry_fig,
                use_container_width=True,
                key=f"telemetry_chart_{i}"
            )

            # ------------------------------------------------
            # ANOMALY SCORE CHART
            # ------------------------------------------------

            anomaly_fig = go.Figure()

            anomaly_fig.add_trace(
                go.Scatter(
                    x=live_df["Time"],
                    y=live_df["Anomaly_Score"],
                    mode="lines",
                    name="Anomaly Score"
                )
            )

            anomaly_fig.add_hline(
                y=0,
                line_dash="dash",
                annotation_text="Decision Boundary"
            )

            anomaly_fig.update_layout(
                title="Real-Time ML Anomaly Score",
                xaxis_title="Mission Time (s)",
                yaxis_title="Anomaly Score",
                height=350
            )

            anomaly_placeholder.plotly_chart(
                anomaly_fig,
                use_container_width=True,
                key=f"anomaly_score_chart_{i}"
            )

            # ------------------------------------------------
            # ANOMALY DRIVERS
            # ------------------------------------------------

            with explanation_placeholder.container():

                st.markdown(
                    '<h3 style="text-align:center;">'
                    '🧠 Anomaly Drivers'
                    '</h3>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Top telemetry-derived features contributing "
                    "to the current deviation from the nominal profile."
                )

                st.dataframe(
                    explanation_df.style.format(
                        {
                            "Standardized Deviation": "{:.3f}",
                            "Absolute Contribution": "{:.3f}"
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # EVENT LOG
            # ------------------------------------------------

            if detected_events:

                with event_placeholder.container():

                    st.markdown(
                        '<h3 style="text-align:center;">'
                        '🚨 AI Event Log'
                        '</h3>',
                        unsafe_allow_html=True
                    )

                    event_df = pd.DataFrame(
                        detected_events
                    )

                    st.dataframe(
                        event_df.tail(10),
                        use_container_width=True,
                        hide_index=True
                    )

            previous_data = current_raw.copy()

            time.sleep(0.12)

        st.success(
            "✅ ML mission simulation completed."
        )

        final_live_df = pd.DataFrame(
            live_rows
        )

        # ----------------------------------------------------
        # MISSION SUMMARY
        # ----------------------------------------------------

        st.markdown(
            '<h2 style="text-align:center;">'
            '📊 Mission ML Summary'
            '</h2>',
            unsafe_allow_html=True
        )

        summary1, summary2, summary3, summary4 = (
            st.columns(4)
        )

        summary1.metric(
            "Total AI Anomalies",
            int(
                final_live_df[
                    "AI_Anomaly"
                ].sum()
            )
        )

        summary2.metric(
            "Maximum Anomaly Score",
            f"{final_live_df['Anomaly_Score'].max():.3f}"
        )

        anomaly_percentage = (
            final_live_df[
                "AI_Anomaly"
            ].mean()
            * 100
        )

        summary3.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.1f}%"
        )

        average_quality = (
            final_live_df[
                "Data_Quality_Score"
            ].mean()
        )

        summary4.metric(
            "Average Data Quality",
            f"{average_quality:.1f}/100"
        )

        # ----------------------------------------------------
        # SEVERITY DISTRIBUTION
        # ----------------------------------------------------

        st.markdown(
            '<h2 style="text-align:center;">'
            '🚦 Anomaly Severity Distribution'
            '</h2>',
            unsafe_allow_html=True
        )

        severity_counts = (
            final_live_df[
                "Severity"
            ]
            .value_counts()
            .reset_index()
        )

        severity_counts.columns = [
            "Severity",
            "Count"
        ]

        severity_fig = go.Figure()

        severity_fig.add_trace(
            go.Bar(
                x=severity_counts["Severity"],
                y=severity_counts["Count"],
                name="Severity Count"
            )
        )

        severity_fig.update_layout(
            title="Mission Classification Distribution",
            xaxis_title="Classification",
            yaxis_title="Telemetry Samples",
            height=400
        )

        st.plotly_chart(
            severity_fig,
            use_container_width=True,
            key="severity_distribution"
        )

        # ----------------------------------------------------
        # FINAL EVENT LOG
        # ----------------------------------------------------

        st.markdown(
            '<h2 style="text-align:center;">'
            '🚨 Final Detected Event Log'
            '</h2>',
            unsafe_allow_html=True
        )

        if detected_events:

            final_event_df = pd.DataFrame(
                detected_events
            )

            st.dataframe(
                final_event_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No anomalous events detected by the selected model."
            )

        # ----------------------------------------------------
        # DOWNLOAD
        # ----------------------------------------------------

        csv = (
            final_live_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Mission ML Log",
            csv,
            "aeropredict_live_ml_log.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# ML MODEL EVALUATION MODE
# ============================================================

elif analysis_mode == "ML Model Evaluation":

    st.markdown(
        '<h2 style="text-align:center;">'
        '🤖 Machine Learning Model Evaluation'
        '</h2>',
        unsafe_allow_html=True
    )

    # ========================================================
    # COMPACT ML SYSTEM OVERVIEW
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '🧠 ML System Overview'
        '</h2>',
        unsafe_allow_html=True
    )

    overview_col1, overview_col2, overview_col3 = (
        st.columns(3)
    )

    with overview_col1:

        st.markdown(
            """
            <div class="overview-card">

            <div class="overview-card-title">
            Feature Engineering
            </div>

            <div class="overview-card-item">
            • Telemetry deviations<br>
            • Temporal dynamics<br>
            • Rolling statistics
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with overview_col2:

        st.markdown(
            """
            <div class="overview-card">

            <div class="overview-card-title">
            Anomaly Detection
            </div>

            <div class="overview-card-item">
            • Isolation Forest<br>
            • One-Class SVM<br>
            • Local Outlier Factor
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with overview_col3:

        st.markdown(
            """
            <div class="overview-card">

            <div class="overview-card-title">
            Model Evaluation
            </div>

            <div class="overview-card-item">
            • Precision<br>
            • Recall<br>
            • F1 / ROC-AUC
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ========================================================
    # EVALUATION SCENARIO
    # ========================================================

    evaluation_scenario = st.selectbox(
        "Evaluation Scenario",
        [
            "Propulsion Degradation",
            "Thermal Event",
            "Trajectory Deviation",
            "Sensor Failure",
            "Compound Anomaly"
        ]
    )

    # ========================================================
    # INDEPENDENT HOLDOUT MISSION
    # ========================================================

    evaluation_data = generate_mission(
        mission_duration,
        evaluation_scenario,
        random_seed + 500
    )

    evaluation_features = engineer_features(
        evaluation_data
    )

    X_test = evaluation_features[
        FEATURE_COLUMNS
    ]

    X_test_scaled = scaler.transform(
        X_test
    )

    y_test = evaluation_data[
        "Anomaly"
    ]

    results = []

    model_predictions = {}
    model_scores = {}

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    for name, model in models.items():

        predictions_raw = model.predict(
            X_test_scaled
        )

        predictions = (
            predictions_raw == -1
        ).astype(int)

        scores = (
            -model.decision_function(
                X_test_scaled
            )
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        try:

            auc = roc_auc_score(
                y_test,
                scores
            )

        except ValueError:

            auc = np.nan

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1]
            ).ravel()
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        results.append(
            {
                "Model": name,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": auc,
                "False Positive Rate": false_positive_rate
            }
        )

        model_predictions[name] = predictions
        model_scores[name] = scores

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # INDEPENDENT EVALUATION
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '🧪 Independent Mission Evaluation'
        '</h2>',
        unsafe_allow_html=True
    )

    st.caption(
        "Models are trained exclusively on nominal mission sequences "
        "and evaluated on an independent holdout mission containing "
        "controlled fault injection."
    )

    evaluation_col1, evaluation_col2, evaluation_col3 = (
        st.columns(3)
    )

    evaluation_col1.metric(
        "Training Missions",
        metadata.get(
            "training_missions",
            10
        )
    )

    evaluation_col2.metric(
        "Holdout Samples",
        len(evaluation_data)
    )

    evaluation_col3.metric(
        "Evaluation Scenario",
        evaluation_scenario
    )

    # ========================================================
    # DATASET BREAKDOWN
    # ========================================================

    data_col1, data_col2, data_col3 = (
        st.columns(3)
    )

    data_col1.metric(
        "Evaluation Samples",
        len(evaluation_data)
    )

    data_col2.metric(
        "Normal Samples",
        int(
            (y_test == 0).sum()
        )
    )

    data_col3.metric(
        "Injected Anomaly Samples",
        int(
            (y_test == 1).sum()
        )
    )

    # ========================================================
    # MODEL BENCHMARK
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '📊 Model Benchmark'
        '</h2>',
        unsafe_allow_html=True
    )

    st.dataframe(
        results_df.style.format(
            {
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "False Positive Rate": "{:.3f}"
            }
        ),
        use_container_width=True
    )

    best_model_row = (
        results_df
        .sort_values(
            "F1 Score",
            ascending=False
        )
        .iloc[0]
    )

    best_model_name = (
        best_model_row["Model"]
    )

    st.success(
        f"Best model by F1 Score: "
        f"{best_model_name} "
        f"({best_model_row['F1 Score']:.3f})"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_fig = go.Figure()

    comparison_fig.add_trace(
        go.Bar(
            x=results_df["Model"],
            y=results_df["Precision"],
            name="Precision"
        )
    )

    comparison_fig.add_trace(
        go.Bar(
            x=results_df["Model"],
            y=results_df["Recall"],
            name="Recall"
        )
    )

    comparison_fig.add_trace(
        go.Bar(
            x=results_df["Model"],
            y=results_df["F1 Score"],
            name="F1 Score"
        )
    )

    comparison_fig.update_layout(
        title="Anomaly Detection Model Comparison",
        barmode="group",
        yaxis_title="Score",
        yaxis=dict(
            range=[0, 1]
        ),
        height=450
    )

    st.plotly_chart(
        comparison_fig,
        use_container_width=True,
        key="model_comparison_chart"
    )

    # ========================================================
    # ROC-AUC
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '📈 ROC-AUC Comparison'
        '</h2>',
        unsafe_allow_html=True
    )

    auc_fig = go.Figure()

    auc_fig.add_trace(
        go.Bar(
            x=results_df["Model"],
            y=results_df["ROC-AUC"],
            name="ROC-AUC"
        )
    )

    auc_fig.update_layout(
        yaxis_title="ROC-AUC",
        yaxis=dict(
            range=[0, 1]
        ),
        height=400
    )

    st.plotly_chart(
        auc_fig,
        use_container_width=True,
        key="roc_auc_chart"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        f'<h2 style="text-align:center;">'
        f'🔎 Confusion Matrix — {best_model_name}'
        f'</h2>',
        unsafe_allow_html=True
    )

    best_predictions = (
        model_predictions[
            best_model_name
        ]
    )

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=[0, 1]
    )

    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Predicted Normal",
                "Predicted Anomaly"
            ],
            y=[
                "Actual Normal",
                "Actual Anomaly"
            ],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    cm_fig.update_layout(
        title="Prediction Confusion Matrix",
        height=450
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True,
        key="evaluation_confusion_matrix"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '📋 Classification Report'
        '</h2>',
        unsafe_allow_html=True
    )

    report = classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=[
            "Normal",
            "Anomaly"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )

    # ========================================================
    # ANOMALY SCORES
    # ========================================================

    st.markdown(
        '<h2 style="text-align:center;">'
        '📉 Model Anomaly Scores'
        '</h2>',
        unsafe_allow_html=True
    )

    score_fig = go.Figure()

    for name, scores in model_scores.items():

        score_fig.add_trace(
            go.Scatter(
                x=evaluation_data["Time"],
                y=scores,
                mode="lines",
                name=name
            )
        )

    score_fig.update_layout(
        title="Anomaly Scores Across the Independent Holdout Mission",
        xaxis_title="Mission Time (s)",
        yaxis_title="Anomaly Score",
        height=500
    )

    st.plotly_chart(
        score_fig,
        use_container_width=True,
        key="model_score_distribution_chart"
    )

    # ========================================================
    # PROJECT SUMMARY
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<h2 style="text-align:center;">'
        '📌 Project Summary'
        '</h2>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3 = (
        st.columns(3)
    )

    summary_col1.metric(
        "ML Algorithms",
        "3"
    )

    summary_col2.metric(
        "Engineered Features",
        len(FEATURE_COLUMNS)
    )

    summary_col3.metric(
        "Evaluation Metrics",
        "4"
    )

    st.markdown(
        '<div class="main-description" '
        'style="margin-top:1rem;">'
        'AeroPredict AI combines aerospace telemetry simulation, '
        'feature engineering, unsupervised anomaly detection and '
        'independent mission evaluation into a unified ML workflow.'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer-text">'
    'AeroPredict AI &nbsp;|&nbsp; '
    'Aerospace telemetry analytics • '
    'anomaly detection • '
    'machine learning • '
    'mission evaluation'
    '</div>',
    unsafe_allow_html=True
)