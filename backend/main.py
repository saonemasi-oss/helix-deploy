"""
HELIX Phase 6 — FastAPI Backend
Loads the four trained models + ontology and serves them as REST endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import joblib, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdflib import Graph, Namespace

# ────────────────────────────────────────────────────────────
#  Paths
# ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def mpath(fn):
    return os.path.join(MODELS_DIR, fn)

# ────────────────────────────────────────────────────────────
#  LSTM architecture — MUST match the Phase 4 definition exactly
#  (a .pt state_dict stores weights only, not the class)
# ────────────────────────────────────────────────────────────
class RecoveryLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)

# ────────────────────────────────────────────────────────────
#  Load everything ONCE at startup
# ────────────────────────────────────────────────────────────
print("Loading models...")

# Phase 3 — tree models + feature schemas
bodyfat_model    = joblib.load(mpath("bodyfat_rf_tuned.joblib"))
bodyfat_features = json.load(open(mpath("bodyfat_features.json")))
calories_model    = joblib.load(mpath("calories_best.joblib"))
calories_features = json.load(open(mpath("calories_features.json")))
archetype_model    = joblib.load(mpath("archetype_classifier.joblib"))
archetype_encoder  = joblib.load(mpath("archetype_label_encoder.joblib"))
archetype_features = json.load(open(mpath("archetype_features.json")))

# Phase 4 — LSTM + scaler + meta
lstm_meta   = json.load(open(mpath("recovery_lstm_meta.json")))
lstm_scaler = joblib.load(mpath("recovery_lstm_scaler.joblib"))
N_FEATURES  = len(lstm_meta["features"])
WINDOW      = lstm_meta["window"]
lstm_model  = RecoveryLSTM(n_features=N_FEATURES)
lstm_model.load_state_dict(torch.load(mpath("recovery_lstm.pt"),
                                      map_location="cpu"))
lstm_model.eval()

# Phase 4 — test-user histories (for the dashboard selector)
test_histories = pd.read_csv(mpath("lstm_test_histories.csv"))

# Phase 5 — ontology
graph = Graph()
graph.parse(mpath("helix_ontology.ttl"), format="turtle")
HELIX = Namespace("http://helix.qmul.ac.uk/ontology#")

print(" All models + ontology loaded")

# ────────────────────────────────────────────────────────────
#  FastAPI app
# ────────────────────────────────────────────────────────────
app = FastAPI(title="HELIX API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "HELIX API running",
            "models": ["bodyfat", "calories", "archetype", "recovery_lstm"],
            "ontology_triples": len(graph)}

# ---- Helper: build a one-row DataFrame in the exact training column order ----
def build_row(payload: dict, feature_list: List[str]) -> pd.DataFrame:
    row = {f: float(payload.get(f, 0)) for f in feature_list}
    return pd.DataFrame([row], columns=feature_list)

# ════════════════════════════════════════════════════════════
#  Endpoint 1 — Body fat prediction
# ════════════════════════════════════════════════════════════
class FeaturePayload(BaseModel):
    features: dict   # {feature_name: value}

@app.post("/predict/bodyfat")
def predict_bodyfat(p: FeaturePayload):
    X = build_row(p.features, bodyfat_features)
    pred = float(bodyfat_model.predict(X)[0])
    return {"predicted_body_fat_pct": round(pred, 1)}

# ════════════════════════════════════════════════════════════
#  Endpoint 2 — Calorie prediction
# ════════════════════════════════════════════════════════════
@app.post("/predict/calories")
def predict_calories(p: FeaturePayload):
    X = build_row(p.features, calories_features)
    pred = float(calories_model.predict(X)[0])
    return {"predicted_calories": round(pred, 0)}

# ════════════════════════════════════════════════════════════
#  Endpoint 3 — Archetype classification
# ════════════════════════════════════════════════════════════
@app.post("/predict/archetype")
def predict_archetype(p: FeaturePayload):
    X = build_row(p.features, archetype_features)
    idx = archetype_model.predict(X)[0]
    label = archetype_encoder.inverse_transform([idx])[0]
    return {"predicted_archetype": str(label)}

# ════════════════════════════════════════════════════════════
#  Endpoint 4 — Recovery readiness (LSTM) for a chosen test user
# ════════════════════════════════════════════════════════════
@app.get("/lstm/users")
def list_lstm_users():
    return {"users": sorted(test_histories["Id"].unique().tolist())}

@app.get("/predict/recovery/{user_id}")
def predict_recovery(user_id: int):
    u = test_histories[test_histories["Id"] == user_id].sort_values("ActivityDate")
    if len(u) <= WINDOW:
        raise HTTPException(404, f"User {user_id} has too few days")
    feats = u[lstm_meta["features"]].values
    actual = u[lstm_meta["target"]].values
    preds, trues, dates = [], [], []
    for i in range(len(u) - WINDOW):
        seq = lstm_scaler.transform(feats[i:i+WINDOW])
        x = torch.tensor(seq[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            preds.append(round(float(lstm_model(x)), 1))
        trues.append(round(float(actual[i+WINDOW]), 1))
        dates.append(str(u["ActivityDate"].values[i+WINDOW])[:10])
    return {"user_id": user_id, "dates": dates,
            "predicted": preds, "actual": trues}

# ════════════════════════════════════════════════════════════
#  Endpoint 5 — Knowledge graph: philosophy → principles
# ════════════════════════════════════════════════════════════
@app.get("/ontology/philosophy/{name}")
def philosophy_principles(name: str):
    q = """
    PREFIX helix: <http://helix.qmul.ac.uk/ontology#>
    SELECT ?desc WHERE {
        helix:%s helix:hasPrinciple ?p .
        ?p helix:description ?desc .
    }""" % name
    rows = [str(r[0]) for r in graph.query(q)]
    return {"philosophy": name, "principles": rows}