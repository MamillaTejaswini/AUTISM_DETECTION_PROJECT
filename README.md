# 🧠 Multimodal Autism Risk Detection System

## 📌 Project Title

**A Multimodal Machine Learning Framework for Autism Risk Prediction Using Questionnaire and Eye-Gaze Analysis**

---

## 📖 Overview

This project presents a **multimodal Autism Spectrum Disorder (ASD) risk screening system** that combines:

* 📊 Behavioral Questionnaire Analysis
* 👁️ Eye-Gaze Pattern Modeling

The system integrates both **subjective behavioral indicators** and **objective visual attention biomarkers** using a **decision-level fusion approach** to provide a reliable ASD risk prediction.

---

## 🎯 Objectives

* Enable **early ASD risk screening**
* Combine **behavioral and physiological signals**
* Provide **interpretable and scalable predictions**
* Build a **real-world deployable system**

---

## ⚙️ Methodology

### 🔹 1. Questionnaire-Based Model

* Dataset: UCI ASD Screening Dataset
* Model: Logistic Regression
* Performance:

  * Accuracy: **96%**
  * F1-score: **0.964**
  * ROC-AUC: **0.964**

---

### 🔹 2. Eye-Gaze Based Model

* Dataset: Saliency4ASD (Eye-tracking dataset)
* Features:

  * Fixation statistics
  * Scanpath length
  * Saccade distance
  * Entropy & velocity features
* Model: Support Vector Machine (SVM)
* Performance:

  * Accuracy: **85%**
  * F1-score: **0.85**
  * ROC-AUC: **0.927**

---

### 🔹 3. Multimodal Fusion (Key Contribution)

A **decision-level fusion approach** is used:

```math
Final Score = 0.5316 × Qscore + 0.4684 × Gscore
```

* Combines predictions from both models
* Uses F1-score-based weighting
* Does NOT require aligned datasets

---

## 🚀 Features

* 👁️ Real-time eye tracking using OpenCV + MediaPipe
* 📊 Behavioral questionnaire input
* 🤖 Dual ML models (Logistic Regression + SVM)
* 🔀 Decision-level fusion
* 📈 Output includes:

  * ASD Risk Percentage
  * Risk Level (Low / Medium / High)
  * Model contribution

---

## 🛠️ Tech Stack

| Category        | Tools                |
| --------------- | -------------------- |
| UI              | Streamlit            |
| Computer Vision | OpenCV, MediaPipe    |
| ML              | Scikit-learn         |
| Data Processing | NumPy, Pandas, SciPy |
| Model Storage   | Joblib               |

---

## 📂 Project Structure

```
autism_detection_project/
│
├── app.py                     # Main Streamlit application
├── feature_extraction.py      # Eye-gaze feature extraction
├── requirements.txt          # Dependencies
├── runtime.txt               # Python version (3.10)
├── models/                   # Saved ML models
├── data/                     # Dataset files
└── README.md                 # Documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone <your-repo-url>
cd autism_detection_project
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run Application

```
streamlit run app.py
```

---

## 🌐 Deployment

The app can be deployed using **Streamlit Cloud**.

⚠️ **Note:**

* Webcam-based eye tracking **will not work in cloud deployment**
* It works only in **local environment**

---

## 📊 Results Summary

| Modality       | Accuracy | F1-score | ROC-AUC |
| -------------- | -------- | -------- | ------- |
| Questionnaire  | 96%      | 0.964    | 0.964   |
| Eye-Gaze (SVM) | 85%      | 0.85     | 0.927   |

👉 Multimodal fusion improves robustness by combining both signals.

---

## ⚠️ Limitations

* Requires webcam for eye tracking (local only)
* Eye-gaze model less accurate than questionnaire model
* Different datasets (no subject alignment)

---

## 🔮 Future Work

* Deep learning-based gaze modeling (CNN/RNN)
* Browser-based eye tracking (WebRTC)
* Multimodal dataset alignment
* Mobile/web compatibility

---

## 👩‍💻 Authors

* Mamilla Tejaswini
* T. Anusha
* P. Leela Sowmya Sri
* M. Divya Sri Ramalakshmi

**Guided by:**
DVH Venu Kumar (Assistant Professor)

---

## 📜 License

This project is developed for **academic and research purposes**.

---

## 🙌 Acknowledgment

Shri Vishnu Engineering College for Women, Bhimavaram

---
