"""Generate plots and summaries for CARLA RL training logs.

Usage:
    python -m carla_rl.scripts.plot_training_run sac_stage3
    python -m carla_rl.scripts.plot_training_run logs/sac_stage2/train_log.csv -o logs/plots/sac_stage2
    python -m carla_rl.scripts.plot_training_run sac_stage2 sac_stage3 --output logs/plots/stage2_vs_stage3
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "carla_rl" / "logs"

ALIASES = {
    "timestep": ("timestep", "timesteps", "step", "steps_total", "total_steps", "env_steps"),
    "episode": ("episode", "episodes", "ep"),
    "reward": ("reward", "ep_return", "episode_reward", "return", "r"),
    "env_reward": ("env_reward", "ep_env_return", "environment_reward"),
    "episode_length": ("episode_length", "ep_steps", "length", "l", "steps"),
    "critic_loss": ("critic_loss", "q_loss", "value_loss"),
    "actor_loss": ("actor_loss", "policy_loss"),
    "alpha": ("alpha", "entropy_alpha"),
    "entropy": ("entropy", "policy_entropy"),
    "restarts": ("restarts", "restart_count"),
    "wall_min": ("wall_min", "wall_minutes"),
    "speed": ("speed", "mean_speed", "avg_speed"),
    "crash": ("crash", "collision", "collided"),
    "success": ("success", "succeeded"),
    "steering": ("steering", "steer", "mean_steer"),
}

OUTCOME_PATTERNS = {
    "collision": "Collision occurred",
    "lane_deviation": "Deviated from lane",
    "wrong_way": "Wrong-way driving detected",
    "max_timesteps": "Exceeded maximum timesteps",
    "done": "done:",
}


@dataclass
class RunData:
    name: str
    input_csv: Path
    run_dir: Path | None
    rows: list[dict[str, float | int | str | None]]
    stdout_log: Path | None


def resolve_input(arg: str) -> Path:
    raw = Path(arg)
    candidates = [
        raw,
        Path.cwd() / raw,
        REPO_ROOT / "carla_rl" / raw,
        LOG_ROOT / arg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find run or CSV: {arg}")


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def first_present(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    by_lower = {col.lower(): col for col in columns}
    for alias in aliases:
        found = by_lower.get(alias.lower())
        if found is not None:
            return found
    return None


def coerce(value: str | None) -> float | int | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def numeric(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_run(arg: str) -> RunData:
    path = resolve_input(arg)
    run_dir: Path | None
    if path.is_dir():
        run_dir = path
        input_csv = path / "train_log.csv"
        if not input_csv.exists():
            raise FileNotFoundError(f"missing train_log.csv in {path}")
        name = path.name
    else:
        input_csv = path
        run_dir = path.parent if path.name.lower() == "train_log.csv" else None
        name = run_dir.name if run_dir is not None else path.stem

    columns, raw_rows = read_csv_rows(input_csv)
    column_map = {
        canonical: first_present(columns, aliases)
        for canonical, aliases in ALIASES.items()
    }

    normalized: list[dict[str, float | int | str | None]] = []
    for index, row in enumerate(raw_rows, start=1):
        item: dict[str, float | int | str | None] = {}
        for canonical, original in column_map.items():
            if original is not None:
                item[canonical] = coerce(row.get(original))
        if "timestep" not in item or item["timestep"] is None:
            item["timestep"] = index
        if "episode" not in item or item["episode"] is None:
            item["episode"] = index
        normalized.append(item)

    stdout_log = None
    if run_dir is not None and (run_dir / "stdout.log").exists():
        stdout_log = run_dir / "stdout.log"

    return RunData(name=name, input_csv=input_csv, run_dir=run_dir, rows=normalized,
                   stdout_log=stdout_log)


def rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values[:]
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(mean(values[start:idx + 1]))
    return out


def series(run: RunData, key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in run.rows:
        x = numeric(row.get("timestep"))
        y = numeric(row.get(key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def parse_outcomes(stdout_log: Path | None) -> dict[str, int]:
    counts = {key: 0 for key in OUTCOME_PATTERNS}
    if stdout_log is None or not stdout_log.exists():
        return {}
    try:
        data = stdout_log.read_bytes()
    except OSError:
        return {}
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    elif data.count(b"\x00") > max(8, len(data) // 20):
        encoding = "utf-16-le"
    else:
        encoding = "utf-8-sig"
    text = data.decode(encoding, errors="replace").replace("\x00", "")
    lines = text.splitlines()
    for line in lines:
        for key, pattern in OUTCOME_PATTERNS.items():
            if pattern in line:
                counts[key] += 1
    return {key: value for key, value in counts.items() if value}


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_line_plot(output: Path, title: str, x_label: str, y_label: str,
                   x: list[float], y: list[float], rolling_window: int,
                   raw_label: str, rolling_label: str) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(x, y, linewidth=1.0, alpha=0.35, label=raw_label)
    if len(y) >= 2:
        ax.plot(x, rolling(y, rolling_window), linewidth=2.0, label=rolling_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def save_multi_axis_plot(output: Path, run: RunData) -> bool:
    keys = [
        ("critic_loss", "Critic loss"),
        ("actor_loss", "Actor loss"),
        ("alpha", "Alpha"),
        ("entropy", "Entropy"),
    ]
    available = [(key, label, *series(run, key)) for key, label in keys]
    available = [entry for entry in available if entry[2] and entry[3]]
    if not available:
        return False

    plt = ensure_matplotlib()
    fig, axes = plt.subplots(len(available), 1, figsize=(10, 2.7 * len(available)),
                             dpi=140, sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, (key, label, x, y) in zip(axes, available):
        ax.plot(x, y, linewidth=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(f"{run.name} optimization signals")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def save_bar_plot(output: Path, title: str, counts: dict[str, int]) -> bool:
    if not counts:
        return False
    plt = ensure_matplotlib()
    labels = list(counts)
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=140)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def save_compare_plot(output: Path, title: str, runs: list[RunData], key: str,
                      y_label: str, rolling_window: int) -> bool:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    plotted = False
    for run in runs:
        x, y = series(run, key)
        if not x or not y:
            continue
        y_plot = rolling(y, rolling_window) if len(y) >= 2 else y
        ax.plot(x, y_plot, linewidth=2.0, label=run.name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return True


def write_normalized_csv(run: RunData, output: Path) -> None:
    keys = list(ALIASES)
    present = [key for key in keys if any(row.get(key) is not None for row in run.rows)]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=present)
        writer.writeheader()
        for row in run.rows:
            writer.writerow({key: row.get(key) for key in present})


def summarize(run: RunData, outcomes: dict[str, int]) -> dict[str, object]:
    _, reward_values = series(run, "reward")
    _, length_values = series(run, "episode_length")
    _, env_values = series(run, "env_reward")
    last = run.rows[-1] if run.rows else {}

    summary: dict[str, object] = {
        "name": run.name,
        "input_csv": str(run.input_csv),
        "rows": len(run.rows),
        "last_timestep": last.get("timestep"),
        "last_episode": last.get("episode"),
        "last_reward": last.get("reward"),
        "last_env_reward": last.get("env_reward"),
        "last_episode_length": last.get("episode_length"),
        "outcomes": outcomes,
    }
    if reward_values:
        best_index = max(range(len(reward_values)), key=reward_values.__getitem__)
        x, _ = series(run, "reward")
        summary.update({
            "mean_reward": round(mean(reward_values), 4),
            "reward_std": round(pstdev(reward_values), 4) if len(reward_values) > 1 else 0.0,
            "best_reward": round(reward_values[best_index], 4),
            "best_reward_timestep": x[best_index],
            "first_10_reward_mean": round(mean(reward_values[:10]), 4),
            "last_10_reward_mean": round(mean(reward_values[-10:]), 4),
        })
    if env_values:
        summary["last_10_env_reward_mean"] = round(mean(env_values[-10:]), 4)
    if length_values:
        summary.update({
            "mean_episode_length": round(mean(length_values), 4),
            "last_10_episode_length_mean": round(mean(length_values[-10:]), 4),
            "max_episode_length": max(length_values),
        })
    return summary


def write_summary_md(output: Path, summaries: list[dict[str, object]],
                     plots: list[Path]) -> None:
    lines = ["# CARLA RL Plot Report", ""]
    lines.append("## Runs")
    for item in summaries:
        lines.append(f"- `{item['name']}`: rows={item['rows']}, "
                     f"last_step={item.get('last_timestep')}, "
                     f"last_reward={item.get('last_reward')}, "
                     f"best_reward={item.get('best_reward')}, "
                     f"last_10_reward_mean={item.get('last_10_reward_mean')}, "
                     f"last_10_episode_length_mean={item.get('last_10_episode_length_mean')}")
        if item.get("outcomes"):
            outcomes = ", ".join(f"{key}={value}" for key, value in item["outcomes"].items())
            lines.append(f"  outcomes: {outcomes}")
    lines.extend(["", "## Plots"])
    if plots:
        for plot in plots:
            lines.append(f"- `{plot.name}`")
    else:
        lines.append("- No plots generated.")
    lines.extend(["", "## Notes"])
    lines.append("- Normalized CSV files map CARLA `ep_return` to `reward` for rl-doctor compatibility.")
    lines.append("- Speed/crash/success/steering diagnostics require those columns in the source CSV.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir(runs: list[RunData]) -> Path:
    if len(runs) == 1:
        return LOG_ROOT / "plots" / runs[0].name
    joined = "_vs_".join(run.name for run in runs[:3])
    if len(runs) > 3:
        joined += f"_plus_{len(runs) - 3}"
    return LOG_ROOT / "plots" / f"compare_{joined}"


def run_rl_doctor(normalized_csv: Path, output_dir: Path) -> bool:
    if shutil.which("rl-doctor") is None:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["rl-doctor", "analyze", str(normalized_csv), "--output", str(output_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_dir / "rl_doctor_command.log").write_text(result.stdout, encoding="utf-8")
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directory name, run directory path, or train_log.csv")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--rl-doctor", action="store_true",
                        help="Also run rl-doctor on normalized CSV inputs when available")
    args = parser.parse_args()

    runs = [load_run(item) for item in args.runs]
    output_dir = Path(args.output).resolve() if args.output else default_output_dir(runs)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[Path] = []
    summaries: list[dict[str, object]] = []
    normalized_dir = output_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run.name)
        normalized_csv = normalized_dir / f"{safe_name}_train_log.csv"
        write_normalized_csv(run, normalized_csv)

        x, reward = series(run, "reward")
        if x and reward:
            plot = output_dir / f"{safe_name}_reward_curve.png"
            save_line_plot(plot, f"{run.name} reward", "timestep", "reward",
                           x, reward, args.rolling_window, "episode reward",
                           f"rolling reward ({args.rolling_window})")
            plots.append(plot)

        x, env_reward = series(run, "env_reward")
        if x and env_reward:
            plot = output_dir / f"{safe_name}_env_reward_curve.png"
            save_line_plot(plot, f"{run.name} env reward", "timestep", "env reward",
                           x, env_reward, args.rolling_window, "env reward",
                           f"rolling env reward ({args.rolling_window})")
            plots.append(plot)

        x, episode_length = series(run, "episode_length")
        if x and episode_length:
            plot = output_dir / f"{safe_name}_episode_steps.png"
            save_line_plot(plot, f"{run.name} episode length", "timestep", "steps",
                           x, episode_length, args.rolling_window, "episode steps",
                           f"rolling steps ({args.rolling_window})")
            plots.append(plot)

        plot = output_dir / f"{safe_name}_optimization.png"
        if save_multi_axis_plot(plot, run):
            plots.append(plot)

        outcomes = parse_outcomes(run.stdout_log)
        plot = output_dir / f"{safe_name}_outcomes.png"
        if save_bar_plot(plot, f"{run.name} stdout outcomes", outcomes):
            plots.append(plot)

        if args.rl_doctor:
            run_rl_doctor(normalized_csv, output_dir / f"rl_doctor_{safe_name}")

        summaries.append(summarize(run, outcomes))

    if len(runs) > 1:
        compare = output_dir / "compare_reward.png"
        if save_compare_plot(compare, "Rolling reward comparison", runs, "reward",
                             "rolling reward", args.rolling_window):
            plots.append(compare)
        compare = output_dir / "compare_episode_steps.png"
        if save_compare_plot(compare, "Rolling episode length comparison", runs,
                             "episode_length", "rolling steps", args.rolling_window):
            plots.append(compare)

    (output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries, "plots": [str(path) for path in plots]},
                   indent=2),
        encoding="utf-8",
    )
    write_summary_md(output_dir / "summary.md", summaries, plots)

    print(f"Wrote report: {output_dir / 'summary.md'}")
    for plot in plots:
        print(f"Wrote plot: {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
