import os
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="Fake News Detection API", version="1.0.0")

# Allow local dev from file:// or different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.getenv("MODELS_DIR", "models")
AVAILABLE = {}  # name -> model (loaded lazily)

def _load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}. Run training first.")
    return joblib.load(path)

class PredictRequest(BaseModel):
    text: str
    model: str = "logreg"  # logreg | svm | rf | nb | ensemble

@app.get("/")
def root():
    return {"ok": True, "message": "Fake News Detection API. POST /predict to classify."}

@app.get("/models")
def list_models():
    models = []
    for name in ["logreg", "svm", "rf", "nb", "ensemble"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models.append(name)
    return {"available": models}

@app.post("/predict")
def predict(req: PredictRequest):
    name = req.model.lower()
    if name not in ["logreg", "svm", "rf", "nb", "ensemble"]:
        return {"ok": False, "error": f"Unknown model: {name}"}

    if name not in AVAILABLE:
        AVAILABLE[name] = _load_model(name)

    model = AVAILABLE[name]
    text = [req.text]

    response: Dict[str, Any] = {"model": name}
    pred = model.predict(text)[0]
    response["label"] = str(pred)

    # Probability or decision score
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(text)[0]
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps") and "clf" in model.named_steps:
                classes = model.named_steps["clf"].classes_
            if classes is not None:
                response["class_probs"] = {str(c): float(p) for c, p in zip(classes, proba)}
        elif hasattr(model, "decision_function"):
            score = model.decision_function(text)[0]
            if isinstance(score, (list, tuple)):
                score = float(score[0])
            response["decision_score"] = float(score)
    except Exception as e:
        response["probability_error"] = str(e)

    return {"ok": True, "result": response}
