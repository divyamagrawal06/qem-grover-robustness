const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface NoiseConfig {
  single_qubit_depolarizing: number;
  two_qubit_depolarizing: number;
  readout_error: number;
  enable_thermal_relaxation: boolean;
  t1_seconds: number;
  t2_seconds: number;
  one_qubit_gate_time_seconds: number;
  two_qubit_gate_time_seconds: number;
}

export interface GroverRunResponse {
  n_qubits: number;
  marked_state: number;
  shots: number;
  iterations: number;
  basis_labels: string[];
  ideal_distribution: number[];
  noisy_distribution: number[];
  mitigated_distribution: number[] | null;
  ideal_success_probability: number;
  noisy_success_probability: number;
  mitigated_success_probability: number | null;
  noisy_mae: number;
  mitigated_mae: number | null;
  noisy_fidelity: number;
  mitigated_fidelity: number | null;
  depth: number;
  gate_count: number;
}

export interface IterationSweepPoint {
  iteration: number;
  ideal_success_probability: number;
  noisy_success_probability: number;
  mitigated_success_probability: number | null;
}

export interface NoiseSweepPoint {
  noise_strength: number;
  ideal_success_probability: number;
  noisy_success_probability: number;
  mitigated_success_probability: number | null;
}

export interface QEMTrainResponse {
  n_qubits: number;
  model_type: string;
  sample_count: number;
  trained_at_utc: string;
  train_loss: number;
  val_loss: number;
  baseline_mae: number;
  mitigated_mae: number;
  mae_reduction_pct: number;
  baseline_fidelity: number;
  mitigated_fidelity: number;
}

export interface QEMStatusItem {
  n_qubits: number;
  model_type: string;
  sample_count: number;
  trained_at_utc: string;
  mae_reduction_pct: number;
  baseline_mae: number;
  mitigated_mae: number;
}

export interface QEMStatusResponse {
  models: QEMStatusItem[];
}

async function postJSON<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API ${path} failed: ${response.status} ${message}`);
  }

  return (await response.json()) as T;
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`API ${path} failed: ${response.status} ${message}`);
  }
  return (await response.json()) as T;
}

export async function runGrover(payload: {
  n_qubits: number;
  marked_state: number;
  shots: number;
  num_iterations?: number | null;
  apply_mitigation: boolean;
  auto_train_if_missing: boolean;
  noise: NoiseConfig;
}): Promise<GroverRunResponse> {
  return postJSON<GroverRunResponse>("/api/grover/run", payload);
}

export async function sweepIterations(payload: {
  n_qubits: number;
  marked_state: number;
  shots: number;
  max_iterations?: number | null;
  include_mitigation: boolean;
  noise: NoiseConfig;
}): Promise<IterationSweepPoint[]> {
  return postJSON<IterationSweepPoint[]>("/api/grover/sweep/iterations", payload);
}

export async function sweepNoise(payload: {
  n_qubits: number;
  marked_state: number;
  shots: number;
  num_iterations?: number | null;
  noise_min: number;
  noise_max: number;
  steps: number;
  include_mitigation: boolean;
  noise: NoiseConfig;
}): Promise<NoiseSweepPoint[]> {
  return postJSON<NoiseSweepPoint[]>("/api/grover/sweep/noise", payload);
}

export async function trainQEM(payload: {
  n_qubits: number;
  sample_count: number;
  shots: number;
  include_thermal_relaxation: boolean;
  model_type: "mlp" | "autoencoder";
  hidden_dim: number;
  latent_dim: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed: number;
}): Promise<QEMTrainResponse> {
  return postJSON<QEMTrainResponse>("/api/qem/train", payload);
}

export async function getQEMStatus(): Promise<QEMStatusResponse> {
  return getJSON<QEMStatusResponse>("/api/qem/status");
}

export async function getHealth(): Promise<{ status: string; project: string }> {
  return getJSON<{ status: string; project: string }>("/");
}
