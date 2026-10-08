import argparse
import csv
import glob
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_csv(log_dir):
    patterns = [
        "pf_run_*.csv",
        "basic_pf_run_*.csv",
        "*.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(log_dir, pattern)))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")

    return files[-1]


def read_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def col(rows, name):
    values = []
    for row in rows:
        try:
            values.append(float(row.get(name, "nan")))
        except ValueError:
            values.append(float("nan"))
    return values


def save_line_plot(t, ys, labels, out_path, title, ylabel):
    plt.figure()
    for y, label in zip(ys, labels):
        plt.plot(t, y, label=label)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="")
    parser.add_argument("--log-dir", default=str(Path.home() / "tb3_projects_ws/log/pf_runs"))
    parser.add_argument("--out-dir", default=str(Path.home() / "tb3_projects_ws/log/pf_plots"))
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    csv_path = args.csv
    if args.latest or not csv_path:
        csv_path = latest_csv(args.log_dir)

    rows = read_csv(csv_path)

    t = col(rows, "time_sec")
    t0 = t[0]
    t = [v - t0 for v in t]

    position_error = col(rows, "position_error")
    yaw_error = col(rows, "yaw_error")
    neff = col(rows, "neff")
    std_x = col(rows, "std_x")
    std_y = col(rows, "std_y")
    std_yaw = col(rows, "std_yaw")
    progress = col(rows, "convergence_progress")
    processing_time_ms = col(rows, "processing_time_ms")

    x_est = col(rows, "x_est")
    y_est = col(rows, "y_est")
    x_true = col(rows, "x_true")
    y_true = col(rows, "y_true")

    run_name = Path(csv_path).stem
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_line_plot(
        t,
        [position_error],
        ["position error"],
        out_dir / "position_error.png",
        "Position Error vs Time",
        "position error [m]",
    )

    save_line_plot(
        t,
        [yaw_error],
        ["yaw error"],
        out_dir / "yaw_error.png",
        "Yaw Error vs Time",
        "yaw error [rad]",
    )

    save_line_plot(
        t,
        [neff],
        ["N_eff"],
        out_dir / "effective_sample_size.png",
        "Effective Sample Size vs Time",
        "N_eff",
    )

    save_line_plot(
        t,
        [std_x, std_y, std_yaw],
        ["std_x", "std_y", "std_yaw"],
        out_dir / "particle_spread.png",
        "Particle Spread vs Time",
        "spread",
    )

    if not all(math.isnan(v) for v in progress):
        save_line_plot(
            t,
            [progress],
            ["convergence progress"],
            out_dir / "convergence_progress.png",
            "Convergence Progress vs Time",
            "progress [%]",
        )

    if not all(math.isnan(v) for v in processing_time_ms):
        save_line_plot(
            t,
            [processing_time_ms],
            ["processing time"],
            out_dir / "processing_time.png",
            "Processing Time per Step",
            "time [ms]",
        )

    plt.figure()
    plt.plot(x_est, y_est, label="estimated path")
    plt.plot(x_true, y_true, label="true path")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Estimated vs True Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_est_vs_true.png", dpi=160)
    plt.close()

    print(f"Read CSV: {csv_path}")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
