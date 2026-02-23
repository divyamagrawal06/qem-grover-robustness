"""
grover_ideal.py — Parameterized Noiseless Grover's Search Circuit
=================================================================

Builds a standard Grover circuit for *n* qubits that searches for one or
more marked computational-basis states.  The implementation follows the
textbook construction:

    1. Uniform superposition  (H^⊗n)
    2. Oracle  (phase-flip marked states)
    3. Diffusion operator  (2|s⟩⟨s| − I)
    4. Repeat steps 2-3 for the optimal number of iterations
    5. Measure all qubits

Everything stays within Qiskit's circuit API so the returned
`QuantumCircuit` can be handed directly to Aer simulators or transpiled
for hardware backends.
"""

from __future__ import annotations

import math
from typing import List, Optional

from qiskit import QuantumCircuit


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def build_grover_circuit(
    n_qubits: int,
    marked_states: List[int],
    num_iterations: Optional[int] = None,
) -> QuantumCircuit:
    """Return a complete, measurement-ready Grover circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (search space size N = 2**n_qubits).
    marked_states : list[int]
        Indices of the target states to search for (0-indexed).
        e.g. ``[5]`` marks the state |101⟩ in a 3-qubit register.
    num_iterations : int, optional
        Number of Grover iterations (oracle + diffuser applications).
        When *None* the optimal count ⌊π/4 · √(N/M)⌋ is used.

    Returns
    -------
    QuantumCircuit
        The assembled circuit with measurements on all qubits.
    """
    N = 2 ** n_qubits
    M = len(marked_states)

    # Validate inputs
    if M == 0:
        raise ValueError("At least one marked state is required.")
    if any(s < 0 or s >= N for s in marked_states):
        raise ValueError(
            f"All marked states must be in [0, {N - 1}] for {n_qubits} qubits."
        )

    # Optimal iteration count: ⌊π/4 · √(N / M)⌋
    if num_iterations is None:
        num_iterations = max(1, math.floor(math.pi / 4 * math.sqrt(N / M)))

    qc = QuantumCircuit(n_qubits, n_qubits)

    # 1. Uniform superposition
    qc.h(range(n_qubits))
    qc.barrier()

    # 2-3. Repeat: Oracle → Diffuser
    for _ in range(num_iterations):
        _apply_oracle(qc, n_qubits, marked_states)
        qc.barrier()
        _apply_diffuser(qc, n_qubits)
        qc.barrier()

    # 4. Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


# ────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────

def _apply_oracle(
    qc: QuantumCircuit, n_qubits: int, marked_states: List[int]
) -> None:
    """Phase-flip each marked state via X-gates + multi-controlled Z."""
    for state in marked_states:
        # Flip qubits that are |0⟩ in the binary representation of `state`
        # so the target becomes |11…1⟩
        _flip_to_all_ones(qc, n_qubits, state)

        # Multi-controlled Z  =  H on last qubit → MCX → H on last qubit
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        # Undo the X-flips
        _flip_to_all_ones(qc, n_qubits, state)


def _apply_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    """Standard Grover diffusion operator  2|s⟩⟨s| − I."""
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    # Multi-controlled Z on |11…1⟩
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def _flip_to_all_ones(
    qc: QuantumCircuit, n_qubits: int, state: int
) -> None:
    """Apply X gates to qubits whose corresponding bit in *state* is 0."""
    for i in range(n_qubits):
        if not (state >> i) & 1:
            qc.x(i)


# ────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    qc = build_grover_circuit(n_qubits=3, marked_states=[5])
    print(qc.draw(output="text"))
    print(f"\nDepth: {qc.depth()},  Gates: {qc.size()}")
