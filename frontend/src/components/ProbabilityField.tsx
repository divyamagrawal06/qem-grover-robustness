"use client";

import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

interface ProbabilityFieldProps {
  labels: string[];
  values: number[];
  color: string;
}

interface BarPoint {
  angle: number;
  radius: number;
  height: number;
}

function ProbabilityBars({ values, color }: { values: number[]; color: string }) {
  const points = useMemo<BarPoint[]>(() => {
    if (values.length === 0) {
      return [];
    }

    const maxBars = 56;
    const step = Math.max(1, Math.ceil(values.length / maxBars));
    const sampled = values.filter((_, index) => index % step === 0);

    return sampled.map((value, index) => {
      const angle = (index / sampled.length) * Math.PI * 2;
      const radius = 2.4;
      const height = Math.max(0.05, value * 10);
      return { angle, radius, height };
    });
  }, [values]);

  return (
    <group>
      {points.map((point, index) => {
        const x = Math.cos(point.angle) * point.radius;
        const z = Math.sin(point.angle) * point.radius;
        return (
          <mesh key={index} position={[x, point.height / 2, z]}>
            <boxGeometry args={[0.12, point.height, 0.12]} />
            <meshStandardMaterial color={color} roughness={0.25} metalness={0.25} />
          </mesh>
        );
      })}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[2.2, 2.6, 80]} />
        <meshStandardMaterial color="#2f4858" opacity={0.35} transparent />
      </mesh>
    </group>
  );
}

export default function ProbabilityField({ labels, values, color }: ProbabilityFieldProps) {
  const hasData = labels.length > 0 && values.length > 0;

  return (
    <div className="probability-field">
      <Canvas camera={{ position: [0, 3.1, 5.2], fov: 52 }}>
        <color attach="background" args={["#f5f0e6"]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[2, 4, 3]} intensity={1.4} />
        <directionalLight position={[-3, 3, -2]} intensity={0.65} color="#2f4858" />
        {hasData ? <ProbabilityBars values={values} color={color} /> : null}
        <OrbitControls enableZoom={true} enablePan={false} minDistance={3.4} maxDistance={8} />
      </Canvas>
    </div>
  );
}
