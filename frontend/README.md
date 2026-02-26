# Frontend Dashboard

Next.js dashboard for visualizing Grover robustness experiments and QEM mitigation.

## Run

```bash
npm install
npm run dev
```

Default URL: `http://localhost:3000`

Set backend endpoint with:

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Features

- Parameter controls for qubits, marked state, shot count, and noise levels
- Recharts analytics:
  - success probability vs iteration count
  - performance vs noise strength
  - ideal/noisy/mitigated distribution comparison
- React Three Fiber 3D probability diffusion view
- On-demand QEM training and cached model status display
