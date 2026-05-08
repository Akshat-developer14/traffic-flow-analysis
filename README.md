# 🚦 Urban Traffic Flow Intelligence & Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/Machine%20Learning-Pipeline-green.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning project designed to analyze, cluster, and predict urban traffic patterns. This repository contains the complete workflow from raw data cleaning and exploratory data analysis (EDA) to the deployment of a multi-model Streamlit dashboard.

---

## 📖 Project Overview

Urban traffic congestion is a significant challenge in modern city planning. This project utilizes the **Metro Interstate Traffic Volume dataset** to build an intelligence system capable of:
- Predicting future traffic volume based on weather and temporal features.
- Classifying congestion levels (Low, Medium, High).
- Identifying behavioral clusters in traffic flow.
- Detecting real-time anomalies (e.g., accidents or extreme weather events).

## 🚀 Key Features

-   **Multi-Model Pipeline**: Integrates Supervised, Unsupervised, and Anomaly Detection models into a single interface.
-   **Cyclical Feature Engineering**: Implements Sine/Cosine transformations for time-based data to preserve temporal continuity.
-   **Intuitive Pattern Naming**: Automatically translates K-Means cluster IDs into human-readable traffic states (e.g., "Morning Rush Hour").
-   **Premium Dashboard UI**: A custom-styled Streamlit frontend featuring responsive CSS grids and real-time inference cards.
-   **End-to-End Documentation**: Complete Jupyter Notebooks detailing the data science lifecycle.

## 📈 Model Performance & Results

The system's accuracy and reliability are backed by rigorous training and evaluation across four distinct ML paradigms:

### 1. Traffic Volume Prediction (Supervised)
- **Model**: XGBoost Regressor
- **Performance**: 
  - **R² Score**: 0.9555 (Explains 95.5% of variance)
  - **MAE**: 249.71 (Mean Absolute Error)
  - **RMSE**: 417.04 (Root Mean Squared Error)

### 2. Congestion Classification (Supervised)
- **Model**: Random Forest Classifier
- **Accuracy**: **~92%** on test data.
- **Outcome**: Successfully categorizes traffic into Low, Medium, and High congestion states.

### 3. Traffic Pattern Clustering (Unsupervised)
- **Model**: K-Means Clustering
- **Optimization**: Optimal **K=6** clusters identified using the Elbow Method and Davies-Bouldin Index.
- **Insights**: Clusters differentiate between Weekday Peaks, Holiday flows, and Late Night patterns.

### 4. Anomaly Detection (Unsupervised)
- **Model**: Isolation Forest
- **Logic**: Statistical outliers are flagged with a **1% contamination threshold**, identifying unusual traffic spikes or drops not explained by weather or time.

---

## 🧠 Key Technical Challenges & Solutions

-   **Cyclical Time Representation**: 
    - *Challenge*: Traditional integer hours (0-23) fail to represent the proximity between 23:00 and 00:00.
    - *Solution*: Implemented **Sine/Cosine Encoding** to map time onto a circular coordinate system, allowing the models to learn temporal continuity.
-   **Model Integration**: 
    - *Challenge*: Combining 4 models with different feature requirements into a single dashboard.
    - *Solution*: Developed a centralized preprocessing pipeline that dynamically routes features to the correct model based on real-time user input.
-   **Statistical Anomalies**:
    - *Challenge*: Differentiating between "high traffic" and an "anomaly."
    - *Solution*: Used **Isolation Forest** to isolate observations based on feature rarity rather than just high volume, improving detection accuracy for accidents vs. rush hour.

---

## 📸 Screenshots

| 🚗 Main Prediction Dashboard | 🧪 Technical Feature Vector |
|---|---|
| ![Main Dashboard](<Screenshot 2026-05-08 143929.png>) | ![Technical View](<Screenshot 2026-05-08 144949.png>) |

| 🚦 Congestion & Pattern Results | ⚠️ Anomaly Detection State |
|---|---|
| ![Results Grid](<Screenshot 2026-05-08 145146.png>) | ![Anomaly Alert](<Screenshot 2026-05-08 145254.png>) |

| 📖 Project Documentation | 🛠️ Model Architecture Info |
|---|---|
| ![Docs Tab](<Screenshot 2026-05-08 145246.png>) | ![Model Info](<Screenshot 2026-05-08 145325.png>) |
---

## 🛠️ Tech Stack

-   **Language**: Python 3.12+
-   **Web Framework**: Streamlit
-   **Machine Learning**: XGBoost, Scikit-Learn (Random Forest, K-Means, Isolation Forest)
-   **Data Manipulation**: Pandas, NumPy
-   **Visualization**: Matplotlib, Seaborn
-   **Environment Management**: `uv`

## 🧠 Model Architecture

The system utilizes four distinct models to provide a comprehensive traffic analysis:

1.  **XGBoost Regressor**: High-performance volume prediction based on weather and time.
2.  **Random Forest Classifier**: Categorizes traffic into **Low**, **Medium**, or **High** congestion levels.
3.  **K-Means Clustering**: Identifies 6 unique traffic behavior patterns (e.g., Weekday Peak, Late Night Flow).
4.  **Isolation Forest**: Detects statistical anomalies in traffic patterns with a 1% contamination threshold.

## 📊 Dataset

The dataset used is the **Metro Interstate Traffic Volume** dataset, which includes:
-   Hourly traffic volume on I-94.
-   Weather features (temp, rain, snow, clouds).
-   Temporal features (holidays, date/time).

## 🏃 How to Run

Ensure you have [uv](https://github.com/astral-sh/uv) installed.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Akshat-developer14/traffic-flow-analysis.git
    cd traffic-flow-analysis
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Launch the Dashboard**:
    ```bash
    uv run streamlit run app/main.py
    ```

## 📂 Project Structure

```text
├── app/               # Streamlit application
├── data/              # Raw and processed datasets
├── models/            # Trained model pipelines (.joblib)
├── notebooks/         # Training & EDA notebooks
│   └── training/      # Specific model training workflows
├── pyproject.toml     # Project dependencies
└── README.md
```

## 👤 Author

**Akshat**
-   GitHub: [@Akshat-developer14](https://github.com/Akshat-developer14)

---
*Developed as a comprehensive showcase of end-to-end Machine Learning deployment.*
