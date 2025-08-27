# Fake News Detection Website (FastAPI + Vanilla HTML)

This is a **minimal, production-ready scaffold** for a fake news detection website that supports **multiple ML models**:
- Logistic Regression
- Linear SVM (LinearSVC)
- Random Forest Classifier
- Multinomial Naive Bayes
- (Optional) Soft Voting Ensemble across the above (where probability is available)

## Project Structure
```
fake-news-detector-website/
├─ backend/
│  ├─ app.py               # FastAPI server exposing /predict
│  ├─ train.py             # Training script to create model artifacts
│  ├─ __init__.py
├─ data/
│  └─ sample.csv           # Tiny sample dataset (text,label) for quick testing
├─ models/                 # Saved pipelines (.joblib) will be written here
├─ frontend/
│  └─ index.html           # Simple UI calling the FastAPI backend
└─ requirements.txt
```

## Quickstart
1) **Create a virtual environment & install deps**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

2) **Train models** (uses the tiny `data/sample.csv` by default—replace with your dataset later):
```bash
python backend/train.py
# This will generate joblib files under ./models
```

3) **Run the API server**
```bash
uvicorn backend.app:app --reload --port 8000
```

4) **Open the frontend**
- Just open `frontend/index.html` in your browser **OR**
- Serve it (optional) with any static server:
```bash
python -m http.server 8080 -d frontend
# then open http://localhost:8080
```
> The frontend expects the API at `http://localhost:8000`. Change it in `index.html` if needed.

## Replace the dataset
- Put your CSV (columns: `text,label`) under `data/` and change the `--csv` argument:
```bash
python backend/train.py --csv data/your_dataset.csv
```

## Notes
- Pipelines use `TfidfVectorizer` + chosen classifier per model.
- For models that don’t support probabilities (e.g., `LinearSVC`), the API returns a **decision score** instead of probability.
- The ensemble uses soft voting when probabilities are available; SVM votes via sigmoid-calibrated probabilities (`CalibratedClassifierCV`).

