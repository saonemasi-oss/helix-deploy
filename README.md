# HELIX — Muscle-Growth Prediction and Personalised Training Platform

MSc Data Science dissertation project, Queen Mary University of London
**Author:** Sawan Masih Nayak (250885248) · **Supervisor:** Dr Sukhpal Singh Gill

HELIX unifies four kinds of work into one deployed system: statistical analysis and machine-learning prediction of body composition, deep-learning forecasting of recovery readiness, an OWL knowledge graph encoding 130 years of bodybuilding theory, and a live two-service cloud deployment on Google Cloud Run.

## Live system

No installation needed — the platform runs in a browser:

- **Frontend (user interface):** https://helix-frontend-150551383910.europe-west2.run.app
- **Backend (API):** https://helix-backend-150551383910.europe-west2.run.app

Both services scale to zero when idle, so the first request after a quiet period may take a few seconds while the container starts. This is normal.

## Headline results

| Task | Model | Baseline | HELIX | Metric |
|---|---|---|---|---|
| Body fat | Random Forest | 0.60 | 0.76 | R² |
| Calories | XGBoost | 0.96 | 0.98 | R² |
| Archetype | XGBoost | — | 0.995 | Accuracy |
| Recovery (next-day readiness) | LSTM | 0.00 | 0.77 | R² |

The recovery LSTM is evaluated **split-by-user** (28 train / 7 held-out users), so its score reflects generalisation to people the model has never seen. Every model was benchmarked against a simpler baseline and kept only where it demonstrably won ("earned complexity").

## Repository structure

```
helix-deploy/
├── backend/          FastAPI service: serves all four models + the ontology (SPARQL)
├── frontend/         Streamlit dashboard, talks to the backend over HTTP/REST
├── models/           Trained model artefacts loaded by the backend at startup
├── notebooks/        The six phase notebooks (see below)
├── README.md         This file
└── .python-version   Runtime pin (Python 3.13) used by Cloud Native Buildpacks
```

### The six phase notebooks

| Phase | Notebook | Covers |
|---|---|---|
| 1 | Data collection | Six public datasets, cleaning, ~26,000 records |
| 2 | Statistical analysis | Hypothesis testing with effect sizes, k-means archetypes, association-rule validation |
| 3 | Machine learning | Body-fat / calorie / archetype models, leakage control, SHAP |
| 4 | Deep learning | LSTM recovery forecasting, split-by-user evaluation |
| 5 | Knowledge representation | OWL ontology (217 RDF triples), SPARQL query suite |
| 6 | Cloud deployment | Cloud Run deployment, validation, build troubleshooting |

Each notebook follows a markdown (what/why) → code → markdown (interpretation) pattern and states its data requirements at the top. They run in Google Colab or any Jupyter environment.

## Running locally

Requires Python 3.13.

```bash
git clone https://github.com/saonemasi-oss/helix-deploy.git
cd helix-deploy

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows (use source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The frontend expects the backend URL in its configuration; point it at `http://localhost:8000` for local use.

## Deployment notes

Both services are containerised from source with **Cloud Native Buildpacks** (no hand-written Dockerfile) and deployed to **Cloud Run** in `europe-west2`. The runtime is pinned to Python 3.13 via `.python-version`, dependencies are constrained to CPU wheels, and the services run with 2 GiB memory / 2 vCPU. These choices resolved real build failures (incompatible Python/torch wheels, GPU dependencies on a CPU service, model files outside the deploy directory) that are documented in the Phase 6 notebook.

## Honest limitations

- The wearable cohort is 35 users; sleep records cover ~30% of days (per-user median imputation for the rest).
- The recovery readiness target is engineered, not clinically validated, and is partly derived from training load.
- The nutritional layer is structured in the ontology (18 amino-acid profiles) but not yet connected to the live recommender.

HELIX is a research prototype and decision-support tool, not a medical device.
