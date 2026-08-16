# HELIX — Muscle-Growth Prediction and Personalised Training Platform

MSc Data Science dissertation project, Queen Mary University of London
**Author:** Sawan Masih Nayak (250885248) · **Supervisor:** Dr Sukhpal Singh Gill

HELIX unifies four kinds of work into one deployed system: statistical analysis and machine-learning prediction of body composition, deep-learning forecasting of recovery readiness, an OWL knowledge graph encoding 130 years of bodybuilding theory, and a live two-service cloud deployment on Google Cloud Run.

## Live system

No installation needed — the platform runs in a browser:

- **Frontend (user interface):** https://helix-frontend-150551383910.europe-west2.run.app
- **Backend (API):** https://helix-backend-150551383910.europe-west2.run.app
- **Interactive API docs:** https://helix-backend-150551383910.europe-west2.run.app/docs

Both services scale to zero when idle, so the first request after a quiet period may take several seconds while the container starts. This is normal.

## Headline results

| Task | Model | Baseline | HELIX | Metric |
|---|---|---|---|---|
| Body fat | Random Forest | 0.60 | 0.76 | R² |
| Calories | XGBoost | 0.96 | 0.98 | R² |
| Archetype | XGBoost | — | 0.9949 | Accuracy |
| Recovery (next-day readiness) | LSTM | 0.00 | 0.77 | R² |

Two things worth reading carefully rather than at face value:

- The recovery LSTM is evaluated **split-by-user** (28 training users, 7 held out entirely; 894 and 237 windowed sequences). Its score reflects generalisation to people the model has never seen, which is the realistic deployment condition.
- The calorie model's R² = 0.98 looks like the strongest number here and is the least informative one — the linear baseline already reached 0.96. Every model was benchmarked against a simpler baseline and kept only where it demonstrably won ("earned complexity"), and saying so when the margin is thin is part of that discipline.

## Repository structure

```
helix-deploy/
├── backend/          FastAPI service: serves all four models + the ontology (SPARQL)
│   ├── main.py       API routes, server-side feature derivation, model loading
│   ├── models/       Trained artefacts + feature schemas, loaded at startup
│   ├── requirements.txt
│   └── .python-version
├── frontend/         Streamlit dashboard, a pure client of the backend API
├── notebooks/        The six phase notebooks (see below)
└── README.md         This file
```

### The six phase notebooks

All six run top to bottom with outputs saved. Each follows a markdown (what/why) → code → markdown (interpretation) structure and states its data requirements at the top.

| Phase | Covers | Key content |
|---|---|---|
| 1 | Data collection | Six public datasets, cleaning, ~26,000 records |
| 2 | Statistical analysis | Hypothesis testing with effect sizes, k-means archetypes, association-rule validation |
| 3 | Machine learning | Body-fat / calorie / archetype models, leakage control, SHAP, dual-feature-set ablation |
| 4 | Deep learning | LSTM recovery forecasting, split-by-user protocol, best-epoch checkpointing |
| 5 | Knowledge representation | OWL ontology (217 RDF triples), multi-hop SPARQL query suite |
| 6 | Cloud deployment | Deployment method, build troubleshooting, live end-to-end verification |

Phases 1–5 run in Google Colab or any Jupyter environment. Phase 6's verification cells run anywhere with an internet connection — they test the live services and need no credentials or Drive mount.

## Running locally

Requires Python 3.13.

```bash
git clone https://github.com/saonemasi-oss/helix-deploy.git
cd helix-deploy

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Point the frontend at `http://localhost:8000` for local use.

### Using the API directly

Prediction endpoints accept **raw measurements**; engineered features (BMI, HRR, training-intensity index, lean body mass, the frequency-by-experience interaction, and the calorie ratios) are derived server-side using the Phase 2 formulas. Anything that cannot be derived and has not been supplied returns **HTTP 422 naming the missing fields** rather than a silently defaulted prediction.

```python
import requests
BACKEND = 'https://helix-backend-150551383910.europe-west2.run.app'

features = {
    'Age': 28, 'Weight_kg': 82.0, 'Height_m': 1.78,
    'Max_BPM': 185, 'Avg_BPM': 128, 'Resting_BPM': 62,
    'Session_Duration_hours': 1.3, 'Calories_Burned': 900.0,
    'Fat_Percentage': 24.0, 'Water_Intake_liters': 2.5,
    'Workout_Frequency_days_per_week': 4, 'Experience_Level': 2,
}
r = requests.post(f'{BACKEND}/predict/calories', json={'features': features})
print(r.json())      # {'predicted_calories': 920.0}
```

## Deployment notes

Both services are containerised from source with **Cloud Native Buildpacks** (no hand-written Dockerfile) and deployed to **Cloud Run** in `europe-west2`. The runtime is pinned to Python 3.13 via `.python-version`, dependencies are constrained to CPU wheels, and the backend runs with 2 GiB memory and 2 vCPU.

```powershell
gcloud run deploy helix-backend --source . --region europe-west2 `
  --allow-unauthenticated --memory 2Gi --cpu 2
```

Getting here took five failed builds — an incompatible Python/torch wheel combination, GPU dependencies pulled onto a CPU-only service, model files left outside the deployed directory, a startup memory ceiling, and an encoding problem in the frontend. Phase 6 documents each with its diagnosis and fix.

Two further defects surfaced only once the live system was compared against the same models running locally, and both are documented in Phase 6 §2.2:

- **Silent zero-filling.** The input helper defaulted missing features to zero, so an incomplete request still returned HTTP 200 while the model predicted from a point far outside its training distribution. The calorie route returned 22 kcal where the same model returned 920 locally. Fixed by deriving features server-side and returning 422 for anything missing.
- **XGBoost version mismatch.** The model was pickled under XGBoost 3.3.0 while the container pinned 2.1.0. Loading a 3.x pickle under 2.1.0 raises no error and preserves feature count and names, but does not reproduce the trained model's predictions. Scikit-learn was pinned identically in both environments, which is why the RandomForest body-fat model was unaffected and the fault looked model-specific rather than environmental. Fixed by aligning the pin; the deployed route now returns 920.0 kcal, matching the local prediction exactly.

Both were invisible to component tests and to the container's own health checks. Only executing the live system and comparing its output against the original artefacts revealed them.

## Honest limitations

- The wearable cohort is 35 users; sleep records cover roughly 30% of days, with per-user median imputation for the remainder.
- The recovery readiness target is engineered rather than clinically validated, and is partly derived from training load, which is also a model input. The LSTM result confirms the network learns the temporal load–recovery relationship; it does not demonstrate a clinically meaningful predictor.
- The nutritional layer is structured in the ontology (eighteen amino-acid profiles) but is not connected to the live recommender.
- All findings derive from public datasets rather than a recruited cohort, so external validity is bounded accordingly.

HELIX is a research prototype and decision-support tool, not a medical device.
