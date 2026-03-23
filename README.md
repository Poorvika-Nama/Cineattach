# 🎬 CineAttach — Movie Emotional Attachment Predictor

> Predict how emotionally attached you will feel to a film, powered by machine learning and 10,000 real survey responses.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What is this?

CineAttach is an end-to-end machine learning web app. You enter a movie name, your age group, and gender — and it predicts your emotional attachment score on a scale of **1.0 to 5.0**.

It was built on a real survey dataset of **10,000 responses** across **269 films**, covering the full data science workflow from raw data to a deployed Flask dashboard.

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cineattach.git
cd cineattach

# 2. Create environment
conda create -n movie_env python=3.11
conda activate movie_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## Project Structure

```
CineAttach/
├── app.py                    ← Flask backend
├── regression_pipeline.pkl   ← Trained ML model
├── requirements.txt          ← Dependencies
├── Movie_vs_Emotional_...csv ← Survey dataset (10,000 rows)
└── templates/
    └── index.html            ← Frontend (form + charts)
```

---

## Dataset

| Property | Value |
|---|---|
| Responses | 10,000 |
| Movies covered | 269 |
| Survey period | 2024 – 2026 |
| Target | Attachment Score (1.0 – 5.0) |

---

## ML Pipeline

```
Raw Data → EDA → Feature Engineering → Preprocessing → Model Comparison → Deployment
```

- **8 models tested:** Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN, SVR
- **Final model:** Ridge Regression
- **CV R²:** 0.644 &nbsp;|&nbsp; **MAE:** 0.425

Ridge was chosen over Lasso because Lasso zeroed out `age` and `gender` coefficients — Ridge keeps every input contributing to the result.

---

## What the App Does

1. User enters **movie name**, **age**, and **gender** — nothing else
2. Storytelling ratings and all other features are **auto-filled** from the dataset
3. The model returns a predicted score with **3 comparison charts**:
   - Your score vs all age groups
   - Your score vs all genders
   - Where your score sits in the full 10,000-response distribution

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn, Pandas, NumPy |
| Frontend | HTML, CSS, Chart.js |
| Model storage | Joblib |

---

## Author

**Poorv** — Built as an end-to-end ML portfolio project · 2026

---

*MIT License*
