import argparse
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=200, n_jobs=None),
    "svm": lambda: CalibratedClassifierCV(LinearSVC(), cv=3, method="sigmoid"),
    "rf": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    "nb": lambda: MultinomialNB(),
}

def build_pipeline(estimator):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", estimator),
    ])

def main(args):
    df = pd.read_csv(args.csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have columns: text,label")
    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs(args.outdir, exist_ok=True)

    trained = {}
    for key, est_fn in MODELS.items():
        print(f"Training {key}...")
        pipe = build_pipeline(est_fn())
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print(f"=== {key} report ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        joblib.dump(pipe, os.path.join(args.outdir, f"{key}.joblib"))
        trained[key] = pipe

    # Build soft-voting ensemble (only for models with predict_proba)
    proba_models = []
    for key in ["logreg", "svm", "rf", "nb"]:
        if key in trained:
            pipe = trained[key]
            # If pipeline supports predict_proba
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                proba_models.append((key, pipe))

    if len(proba_models) >= 2:
        print("Training soft-voting ensemble...")
        # In VotingClassifier, we add base estimators again. Here we rebuild to avoid pipeline reuse issues.
        from sklearn.base import clone
        estimators = []
        for name, pipe in proba_models:
            estimators.append((name, clone(pipe)))
        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        print("=== ensemble report ===")
        print(classification_report(y_test, y_pred, zero_division=0))
        joblib.dump(ensemble, os.path.join(args.outdir, "ensemble.joblib"))
    else:
        print("Not enough probability-capable models to create ensemble. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample.csv", help="Path to CSV with columns text,label")
    parser.add_argument("--outdir", default="models", help="Output directory for model artifacts")
    args = parser.parse_args()
    main(args)
