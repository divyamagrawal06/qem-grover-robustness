"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import ProbabilityField from "@/components/ProbabilityField";
import {
  NoiseConfig,
  getHealth,
  getQEMStatus,
  runGrover,
  sweepIterations,
  sweepNoise,
  trainQEM,
  type GroverRunResponse,
  type IterationSweepPoint,
  type NoiseSweepPoint,
  type QEMStatusItem,
} from "@/lib/api";

const DEFAULT_NOISE: NoiseConfig = {
  single_qubit_depolarizing: 0.01,
  two_qubit_depolarizing: 0.02,
  readout_error: 0.01,
  enable_thermal_relaxation: true,
  t1_seconds: 120e-6,
  t2_seconds: 80e-6,
  one_qubit_gate_time_seconds: 50e-9,
  two_qubit_gate_time_seconds: 300e-9,
};

function pct(value: number | null): string {
  if (value == null) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

function dec(value: number | null): string {
  if (value == null) return "N/A";
  return value.toFixed(4);
}

function chartValue(value: number | string | undefined): string {
  if (typeof value === "number") {
    return value.toFixed(4);
  }
  if (typeof value === "string") {
    return value;
  }
  return "";
}

export default function Home() {
  const [status, setStatus] = useState<string>("connecting");
  const [error, setError] = useState<string | null>(null);

  const [nQubits, setNQubits] = useState<number>(4);
  const [markedState, setMarkedState] = useState<number>(5);
  const [shots, setShots] = useState<number>(10000);

  const [singleNoise, setSingleNoise] = useState<number>(DEFAULT_NOISE.single_qubit_depolarizing);
  const [twoNoise, setTwoNoise] = useState<number>(DEFAULT_NOISE.two_qubit_depolarizing);
  const [readoutError, setReadoutError] = useState<number>(DEFAULT_NOISE.readout_error);
  const [thermalRelaxation, setThermalRelaxation] = useState<boolean>(true);

  const [runData, setRunData] = useState<GroverRunResponse | null>(null);
  const [iterationData, setIterationData] = useState<IterationSweepPoint[]>([]);
  const [noiseData, setNoiseData] = useState<NoiseSweepPoint[]>([]);
  const [qemStatus, setQemStatus] = useState<QEMStatusItem[]>([]);

  const [running, setRunning] = useState<boolean>(false);
  const [training, setTraining] = useState<boolean>(false);
  const [selectedField, setSelectedField] = useState<"ideal" | "noisy" | "mitigated">("mitigated");

  const maxMarkedState = useMemo(() => 2 ** nQubits - 1, [nQubits]);

  useEffect(() => {
    if (markedState > maxMarkedState) {
      setMarkedState(maxMarkedState);
    }
  }, [markedState, maxMarkedState]);

  const noiseConfig: NoiseConfig = useMemo(
    () => ({
      single_qubit_depolarizing: singleNoise,
      two_qubit_depolarizing: twoNoise,
      readout_error: readoutError,
      enable_thermal_relaxation: thermalRelaxation,
      t1_seconds: DEFAULT_NOISE.t1_seconds,
      t2_seconds: DEFAULT_NOISE.t2_seconds,
      one_qubit_gate_time_seconds: DEFAULT_NOISE.one_qubit_gate_time_seconds,
      two_qubit_gate_time_seconds: DEFAULT_NOISE.two_qubit_gate_time_seconds,
    }),
    [singleNoise, twoNoise, readoutError, thermalRelaxation]
  );

  const loadStatus = useCallback(async () => {
    const [health, models] = await Promise.all([getHealth(), getQEMStatus()]);
    setStatus(`${health.status} - ${health.project}`);
    setQemStatus(models.models);
  }, []);

  const runDashboard = useCallback(async () => {
    setRunning(true);
    setError(null);

    try {
      const [runResponse, iterationResponse, noiseResponse] = await Promise.all([
        runGrover({
          n_qubits: nQubits,
          marked_state: markedState,
          shots,
          num_iterations: null,
          apply_mitigation: true,
          auto_train_if_missing: true,
          noise: noiseConfig,
        }),
        sweepIterations({
          n_qubits: nQubits,
          marked_state: markedState,
          shots: Math.min(8000, shots),
          max_iterations: null,
          include_mitigation: true,
          noise: noiseConfig,
        }),
        sweepNoise({
          n_qubits: nQubits,
          marked_state: markedState,
          shots: Math.min(8000, shots),
          num_iterations: null,
          noise_min: 0,
          noise_max: 0.15,
          steps: 12,
          include_mitigation: true,
          noise: noiseConfig,
        }),
      ]);

      setRunData(runResponse);
      setIterationData(iterationResponse);
      setNoiseData(noiseResponse);

      await loadStatus();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unknown dashboard error";
      setError(message);
    } finally {
      setRunning(false);
    }
  }, [loadStatus, markedState, nQubits, noiseConfig, shots]);

  useEffect(() => {
    void loadStatus().catch((caught) => {
      const message = caught instanceof Error ? caught.message : "Failed to load backend status";
      setError(message);
      setStatus("backend unavailable");
    });
  }, [loadStatus]);

  useEffect(() => {
    void runDashboard();
  }, [runDashboard]);

  const activeModel = useMemo(
    () => qemStatus.find((item) => item.n_qubits === nQubits) ?? null,
    [nQubits, qemStatus]
  );

  const distributionData = useMemo(() => {
    if (!runData) {
      return [] as Array<{
        state: string;
        ideal: number;
        noisy: number;
        mitigated: number | null;
      }>;
    }

    const mapped = runData.basis_labels.map((label, idx) => ({
      state: label,
      ideal: runData.ideal_distribution[idx],
      noisy: runData.noisy_distribution[idx],
      mitigated: runData.mitigated_distribution ? runData.mitigated_distribution[idx] : null,
    }));

    if (mapped.length <= 48) {
      return mapped;
    }

    return mapped
      .sort(
        (a, b) =>
          Math.max(b.ideal, b.noisy, b.mitigated ?? 0) - Math.max(a.ideal, a.noisy, a.mitigated ?? 0)
      )
      .slice(0, 48)
      .sort((a, b) => a.state.localeCompare(b.state));
  }, [runData]);

  const fieldValues = useMemo(() => {
    if (!runData) return [];
    if (selectedField === "ideal") return runData.ideal_distribution;
    if (selectedField === "noisy") return runData.noisy_distribution;
    return runData.mitigated_distribution ?? runData.noisy_distribution;
  }, [runData, selectedField]);

  const fieldColor = selectedField === "ideal" ? "#2a9d8f" : selectedField === "noisy" ? "#e76f51" : "#264653";

  const trainModel = useCallback(async () => {
    setTraining(true);
    setError(null);
    try {
      await trainQEM({
        n_qubits: nQubits,
        sample_count: 320,
        shots,
        include_thermal_relaxation: thermalRelaxation,
        model_type: "mlp",
        hidden_dim: 128,
        latent_dim: 48,
        epochs: 80,
        batch_size: 32,
        learning_rate: 1e-3,
        seed: 7,
      });
      await loadStatus();
      await runDashboard();
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Training failed";
      setError(message);
    } finally {
      setTraining(false);
    }
  }, [loadStatus, nQubits, runDashboard, shots, thermalRelaxation]);

  return (
    <div className="dashboard-shell">
      <header className="hero-panel">
        <div>
          <p className="kicker">Quantum Reliability Lab</p>
          <h1>Grover Robustness Dashboard</h1>
          <p>
            Systematic stress testing of Grover search under depolarizing, readout, and decoherence noise,
            with neural mitigation in the loop.
          </p>
        </div>
        <div className="status-block">
          <span className="status-label">Backend</span>
          <strong>{status}</strong>
          {error ? <p className="error-text">{error}</p> : null}
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel controls">
          <h2>Simulation Controls</h2>

          <label>
            Qubits: <strong>{nQubits}</strong>
            <input type="range" min={2} max={8} value={nQubits} onChange={(e) => setNQubits(Number(e.target.value))} />
          </label>

          <label>
            Marked State: <strong>{markedState}</strong>
            <input
              type="range"
              min={0}
              max={maxMarkedState}
              value={markedState}
              onChange={(e) => setMarkedState(Number(e.target.value))}
            />
          </label>

          <label>
            Shots
            <select value={shots} onChange={(e) => setShots(Number(e.target.value))}>
              <option value={2000}>2,000</option>
              <option value={5000}>5,000</option>
              <option value={10000}>10,000</option>
              <option value={20000}>20,000</option>
            </select>
          </label>

          <label>
            Single-Qubit Depolarizing: <strong>{singleNoise.toFixed(3)}</strong>
            <input
              type="range"
              min={0}
              max={0.2}
              step={0.005}
              value={singleNoise}
              onChange={(e) => setSingleNoise(Number(e.target.value))}
            />
          </label>

          <label>
            Two-Qubit Depolarizing: <strong>{twoNoise.toFixed(3)}</strong>
            <input
              type="range"
              min={0}
              max={0.3}
              step={0.005}
              value={twoNoise}
              onChange={(e) => setTwoNoise(Number(e.target.value))}
            />
          </label>

          <label>
            Readout Error: <strong>{readoutError.toFixed(3)}</strong>
            <input
              type="range"
              min={0}
              max={0.2}
              step={0.005}
              value={readoutError}
              onChange={(e) => setReadoutError(Number(e.target.value))}
            />
          </label>

          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={thermalRelaxation}
              onChange={(e) => setThermalRelaxation(e.target.checked)}
            />
            Thermal relaxation noise
          </label>

          <div className="button-row">
            <button onClick={() => void runDashboard()} disabled={running}>
              {running ? "Running..." : "Run Simulation"}
            </button>
            <button onClick={() => void trainModel()} disabled={training} className="secondary">
              {training ? "Training..." : "Train QEM"}
            </button>
          </div>

          {activeModel ? (
            <div className="model-pill">
              <p>Model n={activeModel.n_qubits}</p>
              <p>MAE reduction: {activeModel.mae_reduction_pct.toFixed(2)}%</p>
            </div>
          ) : (
            <div className="model-pill">
              <p>No cached model for n={nQubits}</p>
              <p>Mitigation will auto-train on first run.</p>
            </div>
          )}
        </section>

        <section className="panel metrics">
          <h2>Single-Run Metrics</h2>
          <div className="metric-grid">
            <article>
              <span>Ideal Success</span>
              <strong>{pct(runData?.ideal_success_probability ?? null)}</strong>
            </article>
            <article>
              <span>Noisy Success</span>
              <strong>{pct(runData?.noisy_success_probability ?? null)}</strong>
            </article>
            <article>
              <span>Mitigated Success</span>
              <strong>{pct(runData?.mitigated_success_probability ?? null)}</strong>
            </article>
            <article>
              <span>Noisy MAE</span>
              <strong>{dec(runData?.noisy_mae ?? null)}</strong>
            </article>
            <article>
              <span>Mitigated MAE</span>
              <strong>{dec(runData?.mitigated_mae ?? null)}</strong>
            </article>
            <article>
              <span>Noisy Fidelity</span>
              <strong>{dec(runData?.noisy_fidelity ?? null)}</strong>
            </article>
            <article>
              <span>Mitigated Fidelity</span>
              <strong>{dec(runData?.mitigated_fidelity ?? null)}</strong>
            </article>
            <article>
              <span>Circuit Depth</span>
              <strong>{runData?.depth ?? "N/A"}</strong>
            </article>
          </div>
        </section>

        <section className="panel chart-card">
          <h2>Success Probability vs Iteration Count</h2>
          <ResponsiveContainer width="100%" height={290}>
            <LineChart data={iterationData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#8c8f84" />
              <XAxis dataKey="iteration" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line dataKey="ideal_success_probability" name="Ideal" stroke="#2a9d8f" strokeWidth={2.5} dot={false} />
              <Line dataKey="noisy_success_probability" name="Noisy" stroke="#e76f51" strokeWidth={2.5} dot={false} />
              <Line
                dataKey="mitigated_success_probability"
                name="Mitigated"
                stroke="#264653"
                strokeWidth={2.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>

        <section className="panel chart-card">
          <h2>Performance vs Noise Strength</h2>
          <ResponsiveContainer width="100%" height={290}>
            <LineChart data={noiseData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#8c8f84" />
              <XAxis dataKey="noise_strength" tickFormatter={(value) => Number(value).toFixed(2)} />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value) => chartValue(value as number | string | undefined)} />
              <Legend />
              <Line dataKey="ideal_success_probability" name="Ideal" stroke="#2a9d8f" strokeWidth={2.5} dot={false} />
              <Line dataKey="noisy_success_probability" name="Noisy" stroke="#e76f51" strokeWidth={2.5} dot={false} />
              <Line
                dataKey="mitigated_success_probability"
                name="Mitigated"
                stroke="#264653"
                strokeWidth={2.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </section>

        <section className="panel chart-card wide">
          <h2>Ideal vs Noisy vs Mitigated Distribution</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={distributionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#8c8f84" />
              <XAxis dataKey="state" hide={distributionData.length > 24} />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value) => chartValue(value as number | string | undefined)} />
              <Legend />
              <Bar dataKey="ideal" fill="#2a9d8f" />
              <Bar dataKey="noisy" fill="#e76f51" />
              <Bar dataKey="mitigated" fill="#264653" />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="panel chart-card wide">
          <div className="three-header">
            <h2>3D Probability Diffusion</h2>
            <div className="three-toggle">
              <button
                className={selectedField === "ideal" ? "active" : ""}
                onClick={() => setSelectedField("ideal")}
              >
                Ideal
              </button>
              <button
                className={selectedField === "noisy" ? "active" : ""}
                onClick={() => setSelectedField("noisy")}
              >
                Noisy
              </button>
              <button
                className={selectedField === "mitigated" ? "active" : ""}
                onClick={() => setSelectedField("mitigated")}
              >
                Mitigated
              </button>
            </div>
          </div>

          <ProbabilityField
            labels={runData?.basis_labels ?? []}
            values={fieldValues}
            color={fieldColor}
          />
        </section>
      </main>
    </div>
  );
}
