"""
main.py — FastAPI Application for QEM-Grover Robustness
=======================================================

Entrypoint for the backend API.  Run with:

    uvicorn src.api.main:app --reload --port 8000

CORS is configured to accept requests from the Next.js dev server
(http://localhost:3000).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.quantum.grover_ideal import build_grover_circuit

# ────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────

app = FastAPI(
    title="QEM-Grover Robustness API",
    description="Backend for systematic robustness analysis of Grover's algorithm under noise.",
    version="0.1.0",
)

# CORS — allow the Next.js frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "project": "qem-grover-robustness"}


@app.get("/api/grover/ideal", tags=["grover"])
async def grover_ideal(n_qubits: int = 3, marked_state: int = 5):
    """Build an ideal (noiseless) Grover circuit and return metadata.

    Query params
    ------------
    n_qubits : int   — number of qubits (default 3)
    marked_state : int — target state index (default 5 = |101⟩)
    """
    qc = build_grover_circuit(n_qubits=n_qubits, marked_states=[marked_state])
    return {
        "n_qubits": n_qubits,
        "marked_state": marked_state,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "circuit_ascii": qc.draw(output="text").single_string(),
    }
