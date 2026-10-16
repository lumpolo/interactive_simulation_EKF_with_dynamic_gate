"""
EKF localization with dynamic Mahalanobis gating for Lidar measurements.

Project: Localizzazione in Ambiente Dinamico (Rilevamento di Persone)

The simulation has:
- a unicycle robot with noisy odometry,
- a known map of static landmarks,
- range-bearing Lidar measurements,
- moving people that generate Lidar-like detections but are not part of the map.

The EKF update tries to associate each detection with a map landmark. Detections
whose innovation is outside a dynamic Mahalanobis gate are rejected, so moving
objects do not pollute the localization update.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class ExperimentConfig:
    seed: int = 7
    dt: float = 0.1
    steps: int = 550
    world_size: float = 18.0
    n_landmarks: int = 42
    n_people: int = 6
    lidar_max_range: float = 11.0
    lidar_fov: float = 2.0 * np.pi
    sigma_range: float = 0.12
    sigma_bearing: float = np.deg2rad(1.8)
    process_sigma_xy: float = 0.035
    process_sigma_theta: float = np.deg2rad(1.2)
    base_gate: float = 5.991  # 95% chi-square quantile for 2 DoF.
    min_gate: float = 3.2
    max_gate: float = 9.5
    gate_speed_gain: float = 0.45
    gate_uncertainty_gain: float = 0.75
    gate_range_tightening: float = 0.025


class PoseEKF:
    """Small EKF for 2D pose localization using range-bearing landmarks."""

    def __init__(self, initial_state: np.ndarray, initial_cov: np.ndarray, cfg: ExperimentConfig):
        self.x = initial_state.astype(float).reshape(3)
        self.P = initial_cov.astype(float)
        self.cfg = cfg

    def predict(self, control: tuple[float, float]) -> None:
        v, omega = control
        x, y, th = self.x
        dt = self.cfg.dt

        self.x = np.array([
            x + v * np.cos(th) * dt,
            y + v * np.sin(th) * dt,
            wrap_angle(th + omega * dt),
        ])

        F = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0, v * np.cos(th) * dt],
            [0.0, 0.0, 1.0],
        ])
        Q = np.diag([
            self.cfg.process_sigma_xy**2,
            self.cfg.process_sigma_xy**2,
            self.cfg.process_sigma_theta**2,
        ])
        self.P = F @ self.P @ F.T + Q

    def measurement_model(self, landmark: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = landmark[0] - self.x[0]
        dy = landmark[1] - self.x[1]
        q = max(dx * dx + dy * dy, 1e-9)
        r = np.sqrt(q)
        bearing = wrap_angle(np.arctan2(dy, dx) - self.x[2])

        H = np.array([
            [-dx / r, -dy / r, 0.0],
            [dy / q, -dx / q, -1.0],
        ])
        return np.array([r, bearing]), H

    def update(self, z: np.ndarray, landmark: np.ndarray, R: np.ndarray) -> float:
        z_hat, H = self.measurement_model(landmark)
        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return float(innovation.T @ np.linalg.inv(S) @ innovation)

    def innovation_distance(self, z: np.ndarray, landmark: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
        z_hat, H = self.measurement_model(landmark)
        innovation = z - z_hat
        innovation[1] = wrap_angle(innovation[1])
        S = H @ self.P @ H.T + R
        d2 = float(innovation.T @ np.linalg.inv(S) @ innovation)
        return d2, S


def control_at(t: float) -> tuple[float, float]:
    v = 1.05 + 0.22 * np.sin(0.33 * t)
    omega = 0.26 * np.sin(0.18 * t) + 0.12 * np.cos(0.045 * t)
    return v, omega


def step_unicycle(state: np.ndarray, control: tuple[float, float], dt: float) -> np.ndarray:
    v, omega = control
    x, y, th = state
    return np.array([
        x + v * np.cos(th) * dt,
        y + v * np.sin(th) * dt,
        wrap_angle(th + omega * dt),
    ])


def make_world(cfg: ExperimentConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    landmarks = rng.uniform(-cfg.world_size, cfg.world_size, size=(cfg.n_landmarks, 2))
    people = rng.uniform(-0.7 * cfg.world_size, 0.7 * cfg.world_size, size=(cfg.n_people, 2))
    people_vel = rng.normal(0.0, 0.45, size=(cfg.n_people, 2))
    return landmarks, people, people_vel


def move_people(people: np.ndarray, people_vel: np.ndarray, cfg: ExperimentConfig) -> None:
    people += people_vel * cfg.dt
    limit = 0.85 * cfg.world_size
    for i in range(len(people)):
        for axis in range(2):
            if people[i, axis] < -limit or people[i, axis] > limit:
                people_vel[i, axis] *= -1.0
                people[i, axis] = np.clip(people[i, axis], -limit, limit)


def visible_measurements(
    robot_state: np.ndarray,
    points: np.ndarray,
    rng: np.random.Generator,
    cfg: ExperimentConfig,
    source: str,
) -> list[dict]:
    measurements = []
    R = np.diag([cfg.sigma_range**2, cfg.sigma_bearing**2])
    for idx, point in enumerate(points):
        dx = point[0] - robot_state[0]
        dy = point[1] - robot_state[1]
        r = np.hypot(dx, dy)
        bearing = wrap_angle(np.arctan2(dy, dx) - robot_state[2])
        if r <= cfg.lidar_max_range and abs(bearing) <= cfg.lidar_fov / 2.0:
            z = np.array([
                r + rng.normal(0.0, cfg.sigma_range),
                wrap_angle(bearing + rng.normal(0.0, cfg.sigma_bearing)),
            ])
            measurements.append({"z": z, "point": point.copy(), "source": source, "index": idx, "R": R})
    return measurements


def dynamic_gate_threshold(S: np.ndarray, measurement_range: float, speed: float, cfg: ExperimentConfig) -> float:
    uncertainty = np.sqrt(max(np.linalg.det(S), 0.0))
    threshold = cfg.base_gate
    threshold += cfg.gate_speed_gain * abs(speed)
    threshold += cfg.gate_uncertainty_gain * uncertainty
    threshold -= cfg.gate_range_tightening * measurement_range
    return float(np.clip(threshold, cfg.min_gate, cfg.max_gate))


def associate_and_update(
    ekf: PoseEKF,
    measurements: list[dict],
    landmarks: np.ndarray,
    control: tuple[float, float],
    mode: str,
    cfg: ExperimentConfig,
) -> dict:
    accepted = 0
    rejected = 0
    accepted_dynamic = 0
    rejected_static = 0

    for measurement in measurements:
        z = measurement["z"]
        R = measurement["R"]

        candidates = [ekf.innovation_distance(z, landmark, R) + (i,) for i, landmark in enumerate(landmarks)]
        best_d2, best_S, best_idx = min(candidates, key=lambda item: item[0])

        if mode == "none":
            accept = True
            gate = np.inf
        elif mode == "static":
            gate = cfg.base_gate
            accept = best_d2 <= gate
        elif mode == "dynamic":
            gate = dynamic_gate_threshold(best_S, z[0], control[0], cfg)
            accept = best_d2 <= gate
        else:
            raise ValueError(f"Unknown gate mode: {mode}")

        if accept:
            ekf.update(z, landmarks[best_idx], R)
            accepted += 1
            accepted_dynamic += int(measurement["source"] == "person")
        else:
            rejected += 1
            rejected_static += int(measurement["source"] == "landmark")

    return {
        "accepted": accepted,
        "rejected": rejected,
        "accepted_dynamic": accepted_dynamic,
        "rejected_static": rejected_static,
    }


def run_filter(mode: str, cfg: ExperimentConfig, landmarks: np.ndarray, all_measurements: list[list[dict]], true_states: np.ndarray, odometry: np.ndarray) -> dict:
    ekf = PoseEKF(
        initial_state=true_states[0] + np.array([0.45, -0.35, np.deg2rad(7.0)]),
        initial_cov=np.diag([0.8, 0.8, np.deg2rad(12.0)]) ** 2,
        cfg=cfg,
    )

    estimated = []
    stats = {"accepted": 0, "rejected": 0, "accepted_dynamic": 0, "rejected_static": 0}

    for k in range(cfg.steps):
        control = tuple(odometry[k])
        ekf.predict(control)
        step_stats = associate_and_update(ekf, all_measurements[k], landmarks, control, mode, cfg)
        for key in stats:
            stats[key] += step_stats[key]
        estimated.append(ekf.x.copy())

    estimated = np.array(estimated)
    err_xy = estimated[:, :2] - true_states[1:, :2]
    rmse_xy = float(np.sqrt(np.mean(np.sum(err_xy**2, axis=1))))
    final_error = float(np.linalg.norm(err_xy[-1]))
    stats.update({"estimated": estimated, "rmse_xy": rmse_xy, "final_error": final_error})
    return stats


def simulate(cfg: ExperimentConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    landmarks, people, people_vel = make_world(cfg, rng)

    true_states = [np.array([-9.5, -7.5, np.deg2rad(18.0)])]
    all_measurements = []
    odometry = []
    people_log = []

    for k in range(cfg.steps):
        t = k * cfg.dt
        nominal_control = control_at(t)
        true_next = step_unicycle(true_states[-1], nominal_control, cfg.dt)
        true_states.append(true_next)

        noisy_odom = np.array([
            nominal_control[0] + rng.normal(0.0, 0.045),
            nominal_control[1] + rng.normal(0.0, np.deg2rad(0.9)),
        ])
        odometry.append(noisy_odom)

        move_people(people, people_vel, cfg)
        people_log.append(people.copy())

        measurements = visible_measurements(true_next, landmarks, rng, cfg, "landmark")
        measurements += visible_measurements(true_next, people, rng, cfg, "person")
        rng.shuffle(measurements)
        all_measurements.append(measurements)

    true_states = np.array(true_states)
    odometry = np.array(odometry)
    results = {
        "landmarks": landmarks,
        "people_log": np.array(people_log),
        "true_states": true_states,
        "measurements": all_measurements,
    }
    for mode in ("none", "static", "dynamic"):
        results[mode] = run_filter(mode, cfg, landmarks, all_measurements, true_states, odometry)
    return results


def plot_results(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lidar_dynamic_gate_results.png"

    true_states = results["true_states"]
    landmarks = results["landmarks"]
    final_people = results["people_log"][-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    ax = axes[0]
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=18, c="#5b6770", marker="x", label="landmark statici")
    ax.scatter(final_people[:, 0], final_people[:, 1], s=42, c="#d55e00", label="persone mobili")
    ax.plot(true_states[:, 0], true_states[:, 1], "k-", linewidth=2.2, label="traiettoria vera")

    styles = {
        "none": ("#cc79a7", "EKF senza gate"),
        "static": ("#0072b2", "EKF gate statico"),
        "dynamic": ("#009e73", "EKF gate dinamico"),
    }
    for mode, (color, label) in styles.items():
        est = results[mode]["estimated"]
        ax.plot(est[:, 0], est[:, 1], color=color, linewidth=1.8, label=label)

    ax.set_title("Localizzazione con Lidar in ambiente dinamico")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    modes = ["none", "static", "dynamic"]
    labels = ["no gate", "statico", "dinamico"]
    rmse = [results[m]["rmse_xy"] for m in modes]
    dyn = [results[m]["accepted_dynamic"] for m in modes]
    rej_static = [results[m]["rejected_static"] for m in modes]

    x = np.arange(len(modes))
    ax.bar(x - 0.25, rmse, width=0.25, color="#009e73", label="RMSE posizione [m]")
    ax.bar(x, dyn, width=0.25, color="#d55e00", label="persone accettate")
    ax.bar(x + 0.25, rej_static, width=0.25, color="#0072b2", label="landmark respinti")
    ax.set_xticks(x, labels)
    ax.set_title("Metriche esperimento")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(fontsize=8)

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def print_summary(results: dict) -> None:
    print("\nRisultati esperimento Lidar + EKF con gate dinamico")
    print("-" * 62)
    for mode, label in [("none", "Senza gate"), ("static", "Gate statico"), ("dynamic", "Gate dinamico")]:
        r = results[mode]
        print(
            f"{label:14s} | RMSE xy: {r['rmse_xy']:.3f} m | "
            f"errore finale: {r['final_error']:.3f} m | "
            f"persone accettate: {r['accepted_dynamic']:4d} | "
            f"landmark respinti: {r['rejected_static']:4d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="EKF localization with dynamic Lidar gating.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=550)
    parser.add_argument("--show", action="store_true", help="Show the result figure interactively.")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    cfg = ExperimentConfig(seed=args.seed, steps=args.steps)
    results = simulate(cfg)
    print_summary(results)
    fig_path = plot_results(results, args.out_dir)
    print(f"\nFigura salvata in: {fig_path.resolve()}")

    if args.show:
        image = plt.imread(fig_path)
        plt.figure(figsize=(13, 5.8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

