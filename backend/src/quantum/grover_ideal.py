"""Parameterized noiseless Grover circuit construction."""

from __future__ import annotations

import math
from typing import Iterable, Optional

from qiskit import QuantumCircuit


def build_grover_circuit(
    n_qubits: int,
    marked_states: Iterable[int],
    num_iterations: Optional[int] = None,
) -> QuantumCircuit:
    """Build a measurement-ready Grover circuit."""
    n_states = 2 ** n_qubits
    marked_states = sorted(set(marked_states))
    n_marked = len(marked_states)

    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    if n_marked == 0:
        raise ValueError("At least one marked state is required.")
    if any(state < 0 or state >= n_states for state in marked_states):
        raise ValueError(
            f"All marked states must be in [0, {n_states - 1}] for {n_qubits} qubits."
        )

    if num_iterations is None:
        num_iterations = optimal_grover_iterations(n_qubits=n_qubits, n_marked=n_marked)
    elif num_iterations < 0:
        raise ValueError("num_iterations must be >= 0.")

    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(range(n_qubits))
    circuit.barrier()

    for _ in range(num_iterations):
        _apply_oracle(circuit, n_qubits, marked_states)
        circuit.barrier()
        _apply_diffuser(circuit, n_qubits)
        circuit.barrier()

    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def optimal_grover_iterations(n_qubits: int, n_marked: int = 1) -> int:
    """Return floor(pi/4 * sqrt(N/M)), clipped to at least one iteration."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1.")
    if n_marked < 1:
        raise ValueError("n_marked must be >= 1.")

    n_states = 2 ** n_qubits
    return max(1, math.floor(math.pi / 4 * math.sqrt(n_states / n_marked)))


def _apply_oracle(circuit: QuantumCircuit, n_qubits: int, marked_states: Iterable[int]) -> None:
    """Phase-flip each marked state via X gates and MCZ decomposition."""
    for state in marked_states:
        _flip_to_all_ones(circuit, n_qubits, state)

        if n_qubits == 1:
            circuit.z(0)
        else:
            circuit.h(n_qubits - 1)
            circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            circuit.h(n_qubits - 1)

        _flip_to_all_ones(circuit, n_qubits, state)


def _apply_diffuser(circuit: QuantumCircuit, n_qubits: int) -> None:
    """Apply the standard Grover diffusion operator."""
    circuit.h(range(n_qubits))
    circuit.x(range(n_qubits))

    if n_qubits == 1:
        circuit.z(0)
    else:
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)

    circuit.x(range(n_qubits))
    circuit.h(range(n_qubits))


def _flip_to_all_ones(circuit: QuantumCircuit, n_qubits: int, state: int) -> None:
    """Apply X gates on qubits whose bit in state is 0."""
    for index in range(n_qubits):
        if not (state >> index) & 1:
            circuit.x(index)
