# qem-grover-robustness

Grover's Algorithm robustness study under realistic noise with a PyTorch quantum error mitigation (QEM) pipeline and an interactive Next.js dashboard.

## What is implemented

- Parameterized Grover circuit generation for `n_qubits in [2, 8]`
- Aer noise modeling with:
  - single- and two-qubit depolarizing errors
  - readout error
  - thermal relaxation (`T1/T2` + gate times)
- Batch experiment utilities for:
  - success probability vs iteration count
  - success probability vs noise strength
- QEM supervised learning pipeline:
  - noisy/ideal paired dataset generation
  - PyTorch MLP and Autoencoder options
  - MAE and fidelity evaluation
- FastAPI backend endpoints for simulation, sweeps, training, and status
- Next.js dashboard with:
  - controls for qubits, marked state, shots, and noise knobs
  - Recharts analytics
  - React Three Fiber 3D probability field
  - real-time API inference with mitigated outputs
- Technical report draft in `report/technical_report.md`

## Repository layout

- `backend/src/quantum`: circuits, noise models, simulator, metrics, dataset generation
- `backend/src/ml`: QEM models and training/inference service
- `backend/src/api`: FastAPI app and request/response schemas
- `backend/scripts`: batch experiment and model training scripts
- `frontend/src/app`: Next.js dashboard UI
- `frontend/src/components/ProbabilityField.tsx`: 3D quantum-state view
- `report/technical_report.md`: academic report draft

## Backend setup

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Backend docs: `http://localhost:8000/docs`

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Dashboard: `http://localhost:3000`

Set `NEXT_PUBLIC_API_URL` if backend is not on `http://localhost:8000`.

## Key API routes

- `GET /`: health check
- `GET /api/grover/ideal`: circuit metadata
- `POST /api/grover/run`: ideal/noisy/mitigated single run
- `POST /api/grover/sweep/iterations`: success vs iteration
- `POST /api/grover/sweep/noise`: success vs noise
- `POST /api/qem/train`: train QEM model for specific qubit count
- `GET /api/qem/status`: list loaded QEM models

## Batch experiments

From repository root:

```bash
python backend/scripts/run_batch_experiments.py --shots 10000 --include-mitigation
```

Outputs CSV files under `data/experiments/`:

- `iteration_sweep.csv`
- `noise_sweep.csv`
- `failure_thresholds.csv`

## QEM model training CLI

```bash
python backend/scripts/train_qem.py --n-qubits 4 --sample-count 512 --epochs 80
```

Model checkpoint is written to `data/experiments/qem_model.pt` by default.
