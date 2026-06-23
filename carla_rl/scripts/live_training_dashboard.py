"""Live browser dashboard for CARLA RL training logs.

Usage:
    python -m carla_rl.scripts.live_training_dashboard sac_avoid_v1
    python -m carla_rl.scripts.live_training_dashboard --latest --port 8765
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "carla_rl" / "logs"
CHECKPOINT_ROOT = REPO_ROOT / "carla_rl" / "checkpoints"

ALIASES = {
    "step": ("step", "timestep", "timesteps", "steps_total", "total_steps", "env_steps"),
    "episode": ("episode", "episodes", "ep"),
    "reward": ("ep_return", "reward", "episode_reward", "return", "r"),
    "env_reward": ("ep_env_return", "env_reward", "environment_reward"),
    "episode_steps": ("ep_steps", "episode_length", "length", "steps", "l"),
    "critic_loss": ("critic_loss", "q_loss", "value_loss"),
    "actor_loss": ("actor_loss", "policy_loss"),
    "alpha": ("alpha", "entropy_alpha"),
    "entropy": ("entropy", "policy_entropy"),
    "restarts": ("restarts", "restart_count"),
    "wall_min": ("wall_min", "wall_minutes"),
}


@dataclass(frozen=True)
class DashboardConfig:
    run: str | None
    latest: bool
    refresh_seconds: float
    window: int | None


def find_open_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"no open port found from {preferred} to {preferred + 49}")


def latest_run_dir() -> Path:
    candidates = [
        path.parent
        for path in LOG_ROOT.rglob("train_log.csv")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f"no train_log.csv found under {LOG_ROOT}")
    return max(candidates, key=lambda path: (path / "train_log.csv").stat().st_mtime)


def resolve_run(run: str | None, latest: bool) -> Path:
    if latest or not run:
        return latest_run_dir()
    raw = Path(run)
    candidates = [
        raw,
        Path.cwd() / raw,
        LOG_ROOT / run,
        REPO_ROOT / "carla_rl" / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.parent.resolve()
        if (candidate / "train_log.csv").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find run log for {run!r}")


def coerce(value: str | None) -> float | int | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    # Snapshot the file first so a concurrent append cannot leave csv.DictReader
    # looking at a half-written line.
    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return [], []
    header = lines[0]
    body = lines[1:]
    if body and len(body[-1].split(",")) != len(header.split(",")):
        body = body[:-1]
    reader = csv.DictReader([header, *body])
    columns = list(reader.fieldnames or [])
    return columns, [{key: coerce(value) for key, value in row.items()} for row in reader]


def first_present(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for alias in aliases:
        found = by_lower.get(alias.lower())
        if found is not None:
            return found
    return None


def normalize_rows(columns: list[str], rows: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    column_map = {
        canonical: original
        for canonical, aliases in ALIASES.items()
        if (original := first_present(columns, aliases)) is not None
    }
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = {canonical: row.get(original) for canonical, original in column_map.items()}
        if item.get("step") is None:
            item["step"] = index
        if item.get("episode") is None:
            item["episode"] = index
        for column in columns:
            item.setdefault(column, row.get(column))
        normalized.append(item)
    return column_map, normalized


def rolling_mean(values: list[float], window: int = 10) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    subset = clean[-window:]
    return sum(subset) / len(subset)


def checkpoint_info(run_name: str) -> dict[str, Any] | None:
    candidates = [
        CHECKPOINT_ROOT / f"{run_name}.pth",
        CHECKPOINT_ROOT / f"{run_name}_latest.pth",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        existing = list(CHECKPOINT_ROOT.glob(f"{run_name}*.pth"))
    if not existing:
        return None
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    stat = newest.stat()
    return {
        "name": newest.name,
        "path": str(newest),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def load_payload(config: DashboardConfig) -> dict[str, Any]:
    run_dir = resolve_run(config.run, config.latest)
    csv_path = run_dir / "train_log.csv"
    columns, raw_rows = read_rows(csv_path)
    column_map, rows = normalize_rows(columns, raw_rows)
    if config.window and config.window > 0:
        rows = rows[-config.window:]

    full_rows = normalize_rows(columns, raw_rows)[1]
    rewards = [value for row in full_rows if (value := numeric(row.get("reward"))) is not None]
    env_rewards = [value for row in full_rows if (value := numeric(row.get("env_reward"))) is not None]
    lengths = [value for row in full_rows if (value := numeric(row.get("episode_steps"))) is not None]
    last = full_rows[-1] if full_rows else {}
    stat = csv_path.stat()

    return {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "csv": str(csv_path),
        "rows": rows,
        "columns": columns,
        "column_map": column_map,
        "summary": {
            "row_count": len(full_rows),
            "visible_rows": len(rows),
            "last_step": last.get("step"),
            "last_episode": last.get("episode"),
            "last_reward": last.get("reward"),
            "last_env_reward": last.get("env_reward"),
            "last_episode_steps": last.get("episode_steps"),
            "best_reward": max(rewards) if rewards else None,
            "last_10_reward_mean": rolling_mean(rewards),
            "last_10_env_reward_mean": rolling_mean(env_rewards),
            "last_10_episode_steps_mean": rolling_mean(lengths),
            "csv_mtime": stat.st_mtime,
            "csv_age_seconds": max(0.0, time.time() - stat.st_mtime),
            "checkpoint": checkpoint_info(run_dir.name),
        },
    }


HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CARLA RL Live Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #677386;
      --line: #d8dee9;
      --blue: #2563eb;
      --green: #14935f;
      --red: #dc2626;
      --amber: #c47f00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 248, 251, 0.95);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px 12px;
      backdrop-filter: blur(8px);
    }
    h1 { margin: 0 0 8px; font-size: 20px; font-weight: 650; }
    .bar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 14px;
      color: var(--muted);
      font-size: 13px;
    }
    button, input {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button.active { border-color: var(--blue); color: var(--blue); }
    main { padding: 16px 18px 24px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric, .chart {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .metric { padding: 10px 12px; min-height: 70px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    .value { font-size: 20px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .charts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }
    .chart { padding: 10px 12px 12px; min-height: 310px; }
    .chart h2 { margin: 0 0 8px; font-size: 14px; font-weight: 650; }
    canvas { display: block; width: 100%; height: 250px; }
    .status-ok { color: var(--green); }
    .status-stale { color: var(--amber); }
    .status-error { color: var(--red); }
    @media (max-width: 520px) {
      .charts { grid-template-columns: 1fr; }
      .chart { min-height: 260px; }
      canvas { height: 210px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>CARLA RL Live Dashboard</h1>
    <div class="bar">
      <span id="run">run: loading</span>
      <span id="csv">csv: loading</span>
      <span id="status">status: loading</span>
      <button id="pause">Pause</button>
      <label>window <input id="window" type="number" min="0" step="10" value="0" style="width: 82px"></label>
    </div>
  </header>
  <main>
    <section class="metrics" id="metrics"></section>
    <section class="charts">
      <div class="chart"><h2>Reward</h2><canvas id="reward"></canvas></div>
      <div class="chart"><h2>Environment Reward</h2><canvas id="envReward"></canvas></div>
      <div class="chart"><h2>Episode Steps</h2><canvas id="steps"></canvas></div>
      <div class="chart"><h2>Optimization</h2><canvas id="optimization"></canvas></div>
    </section>
  </main>
  <script>
    const refreshMs = Number("__REFRESH_SECONDS__") * 1000;
    let paused = false;
    let lastPayload = null;

    document.getElementById("pause").addEventListener("click", () => {
      paused = !paused;
      const button = document.getElementById("pause");
      button.textContent = paused ? "Resume" : "Pause";
      button.classList.toggle("active", paused);
    });
    document.getElementById("window").addEventListener("change", () => loadData());

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") {
        if (!Number.isFinite(value)) return "-";
        if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, {maximumFractionDigits: digits});
        return value.toLocaleString(undefined, {maximumFractionDigits: digits});
      }
      return String(value);
    }

    function metric(label, value) {
      return `<div class="metric"><div class="label">${label}</div><div class="value">${fmt(value)}</div></div>`;
    }

    function drawChart(canvasId, rows, yKeys, options = {}) {
      const canvas = document.getElementById(canvasId);
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.scale(ratio, ratio);
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      const pad = {left: 58, right: 18, top: 12, bottom: 34};
      const plotW = Math.max(1, w - pad.left - pad.right);
      const plotH = Math.max(1, h - pad.top - pad.bottom);
      const xs = rows.map(row => Number(row.step)).filter(Number.isFinite);
      const series = yKeys.map((entry, index) => {
        const key = typeof entry === "string" ? entry : entry.key;
        const label = typeof entry === "string" ? entry : entry.label;
        const color = typeof entry === "string" ? palette(index) : entry.color;
        const points = rows.map(row => ({x: Number(row.step), y: Number(row[key])}))
          .filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
        return {key, label, color, points};
      }).filter(item => item.points.length);
      if (!xs.length || !series.length) {
        ctx.fillStyle = "#677386";
        ctx.fillText("No data yet", pad.left, pad.top + 20);
        return;
      }
      const ys = series.flatMap(item => item.points.map(point => point.y));
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minX === maxX) maxX += 1;
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const yPad = Math.max(1e-9, (maxY - minY) * 0.08);
      minY -= yPad; maxY += yPad;
      const xToPx = value => pad.left + ((value - minX) / (maxX - minX)) * plotW;
      const yToPx = value => pad.top + plotH - ((value - minY) / (maxY - minY)) * plotH;

      ctx.strokeStyle = "#d8dee9";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(pad.left, pad.top, plotW, plotH);
      ctx.stroke();
      ctx.fillStyle = "#677386";
      ctx.font = "11px Segoe UI, Arial";
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH * i / 4);
        const value = maxY - (maxY - minY) * i / 4;
        ctx.strokeStyle = "#edf0f5";
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
        ctx.fillText(shortNumber(value), 6, y + 4);
      }
      ctx.fillText(shortNumber(minX), pad.left, h - 10);
      ctx.fillText(shortNumber(maxX), pad.left + plotW - 48, h - 10);

      series.forEach((item, idx) => {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = options.thick ? 2 : 1.7;
        ctx.beginPath();
        item.points.forEach((point, i) => {
          const x = xToPx(point.x), y = yToPx(point.y);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = item.color;
        ctx.fillRect(pad.left + idx * 130, 0, 10, 3);
        ctx.fillText(item.label, pad.left + 14 + idx * 130, 5);
      });
    }

    function palette(index) {
      return ["#2563eb", "#14935f", "#dc2626", "#7c3aed", "#c47f00"][index % 5];
    }

    function shortNumber(value) {
      if (!Number.isFinite(value)) return "-";
      const abs = Math.abs(value);
      if (abs >= 1_000_000) return (value / 1_000_000).toFixed(1) + "M";
      if (abs >= 1_000) return (value / 1_000).toFixed(1) + "k";
      if (abs >= 10) return value.toFixed(0);
      return value.toFixed(2);
    }

    async function loadData() {
      if (paused) return;
      const windowValue = Number(document.getElementById("window").value || 0);
      const url = windowValue > 0 ? `/api/data?window=${windowValue}` : "/api/data";
      try {
        const response = await fetch(url, {cache: "no-store"});
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        lastPayload = payload;
        render(payload);
      } catch (error) {
        const status = document.getElementById("status");
        status.className = "status-error";
        status.textContent = "status: " + error.message;
      }
    }

    function render(payload) {
      const summary = payload.summary || {};
      document.getElementById("run").textContent = "run: " + payload.run;
      document.getElementById("csv").textContent = "csv: " + payload.csv;
      const age = Number(summary.csv_age_seconds || 0);
      const status = document.getElementById("status");
      status.className = age < 90 ? "status-ok" : "status-stale";
      status.textContent = `status: updated ${age.toFixed(0)}s ago`;
      const checkpoint = summary.checkpoint ? summary.checkpoint.name : "-";
      document.getElementById("metrics").innerHTML = [
        metric("Last step", summary.last_step),
        metric("Episode", summary.last_episode),
        metric("Last return", summary.last_reward),
        metric("Last env_return", summary.last_env_reward),
        metric("Episode steps", summary.last_episode_steps),
        metric("Best return", summary.best_reward),
        metric("Last 10 mean", summary.last_10_reward_mean),
        metric("Checkpoint", checkpoint),
      ].join("");
      const rows = payload.rows || [];
      drawChart("reward", rows, [{key: "reward", label: "return", color: "#2563eb"}], {thick: true});
      drawChart("envReward", rows, [{key: "env_reward", label: "env_return", color: "#14935f"}], {thick: true});
      drawChart("steps", rows, [{key: "episode_steps", label: "episode steps", color: "#7c3aed"}], {thick: true});
      drawChart("optimization", rows, [
        {key: "critic_loss", label: "critic", color: "#dc2626"},
        {key: "actor_loss", label: "actor", color: "#2563eb"},
        {key: "alpha", label: "alpha", color: "#14935f"},
        {key: "entropy", label: "entropy", color: "#c47f00"},
      ]);
    }

    window.addEventListener("resize", () => { if (lastPayload) render(lastPayload); });
    loadData();
    setInterval(loadData, refreshMs);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                html = HTML.replace("__REFRESH_SECONDS__", str(self.config.refresh_seconds))
                self.send_text(html)
                return
            if parsed.path == "/api/data":
                query = parse_qs(parsed.query)
                window = self.config.window
                if "window" in query:
                    try:
                        window = int(query["window"][0])
                    except (TypeError, ValueError):
                        window = None
                config = DashboardConfig(
                    run=self.config.run,
                    latest=self.config.latest,
                    refresh_seconds=self.config.refresh_seconds,
                    window=window,
                )
                self.send_json(load_payload(config))
                return
            self.send_text("not found", "text/plain; charset=utf-8", 404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a live CARLA RL training dashboard.")
    parser.add_argument("run", nargs="?", help="Run name under carla_rl/logs, run directory, or train_log.csv")
    parser.add_argument("--latest", action="store_true", help="Follow the newest train_log.csv under carla_rl/logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh", type=float, default=3.0, help="Browser refresh interval in seconds")
    parser.add_argument("--window", type=int, default=0, help="Only display the last N rows; 0 shows all rows")
    args = parser.parse_args()

    port = find_open_port(args.host, args.port)
    config = DashboardConfig(
        run=args.run,
        latest=args.latest or args.run is None,
        refresh_seconds=max(0.5, args.refresh),
        window=args.window if args.window > 0 else None,
    )
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"config": config})
    server = ThreadingHTTPServer((args.host, port), handler)
    url = f"http://{args.host}:{port}/"
    run_dir = resolve_run(config.run, config.latest)
    print(f"Serving CARLA RL live dashboard for {run_dir.name}")
    print(f"Source: {run_dir / 'train_log.csv'}")
    print(f"URL: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
