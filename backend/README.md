# Backend

FastAPI + Qiskit + PyTorch backend for Grover robustness experiments.

## Run API

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Swagger: `http://localhost:8000/docs`

## Scripts

```bash
python scripts/run_batch_experiments.py --shots 10000 --include-mitigation
python scripts/train_qem.py --n-qubits 4 --sample-count 512 --epochs 80
```
