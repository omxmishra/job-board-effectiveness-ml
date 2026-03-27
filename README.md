# 💼 Job Board Effectiveness for College Students

An end-to-end Machine Learning project that predicts whether a student will receive a job offer based on their academic profile and job search behavior.

---

## 🚀 Project Overview

This project analyzes how different factors like GPA, internships, networking, and job application behavior influence the probability of receiving a job offer.

The goal is to build a **realistic and production-ready ML pipeline**, not just a high-accuracy model.

---

## 🎯 Problem Statement

Predict:Offer_Received (0 = No, 1 = Yes)


Using:
- Academic background
- Experience
- Job search strategy

---

## ⚠️ Key Considerations

- ❌ Avoided data leakage (excluded interview rounds & post-offer data)
- ✅ Focused on real-world usable features
- ❌ Did NOT chase unrealistic accuracy
- ✅ Tuned model using threshold optimization

---

## 🧠 Model Used

- Logistic Regression (with class balancing)

Why:
- Interpretable
- Stable
- Works well for structured data
- Suitable for real-world deployment

---

## ⚙️ Pipeline

1. Data Loading
2. Data Cleaning
3. Feature Encoding
4. Train/Test Split
5. Feature Scaling
6. Model Training
7. Threshold Optimization
8. Model Saving
9. Inference Pipeline
10. Streamlit App Deployment

---

## 📊 Results

| Metric | Value |
|------|------|
| Accuracy | ~0.62 |
| Recall (Offer=1) | ~0.59 |
| Precision (Offer=1) | ~0.47 |

👉 Focus: Balanced recall over accuracy (real-world scenario)

---

## 🏗️ Project Structure

Using:
- Academic background
- Experience
- Job search strategy

---

## ⚠️ Key Considerations

- ❌ Avoided data leakage (excluded interview rounds & post-offer data)
- ✅ Focused on real-world usable features
- ❌ Did NOT chase unrealistic accuracy
- ✅ Tuned model using threshold optimization

---

## 🧠 Model Used

- Logistic Regression (with class balancing)

Why:
- Interpretable
- Stable
- Works well for structured data
- Suitable for real-world deployment

---

## ⚙️ Pipeline

1. Data Loading
2. Data Cleaning
3. Feature Encoding
4. Train/Test Split
5. Feature Scaling
6. Model Training
7. Threshold Optimization
8. Model Saving
9. Inference Pipeline
10. Streamlit App Deployment

---

## 📊 Results

| Metric | Value |
|------|------|
| Accuracy | ~0.62 |
| Recall (Offer=1) | ~0.59 |
| Precision (Offer=1) | ~0.47 |

👉 Focus: Balanced recall over accuracy (real-world scenario)

---

## 🏗️ Project Structure
job-board-effectiveness-ml/
│
├── app/
│ └── app.py # Streamlit UI
│
├── data/
│ └── raw/
│ └── dataset.csv
│
├── models/
│ ├── logistic_model.pkl
│ ├── scaler.pkl
│ └── columns.pkl
│
├── notebooks/
│ └── experimentation.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── train_model.py
│ └── evaluate_model.py
│
├── requirements.txt
└── README.md


---

## ▶️ How to Run

### 1. Clone repo
```bash
git clone <your-repo-link>
cd job-board-effectiveness-ml

conda create -n job_env python=3.10
conda activate job_env

pip install -r requirements.txt

python src/train_model.py

streamlit run app/app.py
