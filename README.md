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
-   **Cyclical Feature Engineering**: Implements Sine/Cosine transformations for time-based data (Hour, Month) to preserve temporal continuity.
-   **Interactive Dashboard**: A professional Streamlit frontend for real-time inference and simulation.
-   **End-to-End Documentation**: Complete Jupyter Notebooks detailing the data science lifecycle.

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
