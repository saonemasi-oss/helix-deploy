# HELIX MSc Project — Phase 6 Completion Report

**Student:** Sawan Masih Nayak
**Programme:** MSc Data Science, Queen Mary University of London
**Phase:** 6 — Cloud Deployment (FastAPI + Streamlit on Google Cloud Run)

---

## 1. Objective

Phase 6 integrates the predictive models (Phases 3–4) and the knowledge graph (Phase 5) into a single deployed platform, served live on Google Cloud Platform. It is the core deliverable for the Cloud Computing module and demonstrates the Distributed Systems module through a microservice architecture of independently deployable services.

## 2. Architecture

The platform follows a two-tier microservice design:

- **Backend** — a FastAPI service exposing the four trained models and the ontology as REST endpoints. Loads all artifacts at startup: the body fat and calorie regressors, the archetype classifier, the recovery LSTM (with its scaler and metadata), and the OWL ontology (queried via SPARQL).
- **Frontend** — a Streamlit dashboard providing a four-tab interface (Body Composition, Archetype, Recovery Forecast, Training Philosophy). It communicates with the backend over HTTP.

The two services are deployed independently on Cloud Run, each in its own container, scaling separately. This separation of concerns is itself a demonstration of distributed-systems principles: loosely-coupled services communicating over a network via well-defined interfaces.

## 3. Deployment Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerisation | Cloud Native Buildpacks (source-to-container, no Dockerfile) |
| Hosting | Google Cloud Run (region: europe-west2, London) |
| Build | Google Cloud Build |
| Container registry | Artifact Registry |
| Runtime | Python 3.13, CPU-only PyTorch |

## 4. Live Endpoints

- Frontend dashboard: `https://helix-frontend-150551383910.europe-west2.run.app`
- Backend API: `https://helix-backend-150551383910.europe-west2.run.app`
- API docs (interactive): `/docs` on the backend URL

## 5. Backend Endpoints Served

| Endpoint | Function |
|---|---|
| `GET /` | Service status + ontology triple count |
| `POST /predict/bodyfat` | Body fat % prediction (Phase 3) |
| `POST /predict/calories` | Calorie expenditure prediction (Phase 3) |
| `POST /predict/archetype` | Athlete archetype classification (Phase 3) |
| `GET /predict/recovery/{user_id}` | Next-day readiness forecast (Phase 4 LSTM) |
| `GET /lstm/users` | Lists held-out test users for the recovery demo |
| `GET /ontology/philosophy/{name}` | SPARQL query of a philosophy's principles (Phase 5) |

## 6. Key Engineering Decisions & Challenges Resolved

- **CPU-only PyTorch:** the standard torch build pulls GPU/CUDA libraries unnecessary on a GPU-less Cloud Run service; switching to the CPU-only variant reduced container size and build time.
- **Python version pinning:** the buildpack defaulted to Python 3.14, for which torch had no compatible wheel. Pinned to 3.13 (the newest version with torch support available in the builder) via a `.python-version` file.
- **Model bundling:** model artifacts had to be copied into the service source directory and the load path adjusted, so the buildpack would include them in the container image.
- **Scaler persistence:** the Phase 4 LSTM requires the exact StandardScaler fit during training; this was saved alongside the model (with feature order and window metadata) to guarantee correct inference in production.
- **Startup configuration:** increased memory (2 GiB), CPU (2 cores), and health-check timeout (300s) so the container could load all models before Cloud Run's startup probe expired.
- **Version consistency:** scikit-learn pinned to 1.6.1 (the training version) to avoid cross-version unpickling inconsistencies.

## 7. Honest Limitations

- **Cold starts:** Cloud Run scales to zero when idle; the first request after inactivity incurs a ~20–30s delay while the container restarts and reloads the PyTorch model. Acceptable for a demonstration platform; a minimum-instance configuration would eliminate it at higher cost.
- **Recovery demo scope:** the LSTM tab forecasts for the held-out test users (real, unseen data) rather than accepting arbitrary live input, since a meaningful prediction requires seven consecutive days of physiological history.
- **Models served at training versions:** pinned dependency versions ensure reproducibility but will require maintenance as libraries evolve.

## 8. Module Coverage

- **Cloud Computing: COMPLETE** — live deployment on Google Cloud Run with Cloud Build, Artifact Registry, containerisation, and managed scaling.
- **Distributed Systems: COMPLETE** — two independently-deployed, network-communicating microservices demonstrating service decomposition, loose coupling, and independent scaling.

## 9. Artefacts

- Live frontend and backend Cloud Run services (URLs above)
- `helix_deploy/backend/` — FastAPI app, requirements, build config, bundled models
- `helix_deploy/frontend/` — Streamlit app, requirements, build config
- Demo screen recording + dashboard screenshots

## 10. Handoff to Phase 7 — Dissertation

All six implementation phases are complete. The deployed platform, the five prior completion reports, and the figures generated throughout provide the material for the dissertation. Phase 7 consolidates these into the written submission.

---

*HELIX MSc Project — Phase 6 Completion Report — End of Document*