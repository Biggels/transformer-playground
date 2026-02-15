from __future__ import annotations

import json
import math
import time
from pathlib import Path


def load_loss_points(metrics_path: str | Path) -> tuple[list[int], list[float], list[int], list[float]]:
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []

    with Path(metrics_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.get("step", 0))
            if "val_loss" in row:
                val_steps.append(step)
                val_losses.append(float(row["val_loss"]))
            if "train_loss" in row and "val_loss" not in row:
                train_steps.append(step)
                train_losses.append(float(row["train_loss"]))

    if not train_steps:
        with Path(metrics_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "train_loss" in row:
                    train_steps.append(int(row.get("step", 0)))
                    train_losses.append(float(row["train_loss"]))

    return train_steps, train_losses, val_steps, val_losses


def _polyline(points: list[tuple[float, float]], color: str, width: int = 2) -> str:
    if not points:
        return ""
    seq = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{seq}" />'


def save_loss_plot(
    run_path: str | Path,
    out_path: str | Path | None = None,
    title: str | None = None,
    log_y: bool = False,
) -> Path:
    run = Path(run_path)
    metrics_path = run / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    train_steps, train_losses, val_steps, val_losses = load_loss_points(metrics_path)
    if not val_steps and not train_steps:
        raise ValueError("No train/val loss points found in metrics file")

    if out_path is None:
        plots_dir = run / "reports"
        plots_dir.mkdir(parents=True, exist_ok=True)
        out_path = plots_dir / f"loss-curve-{int(time.time())}.svg"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 720
    left, right, top, bottom = 90, 30, 60, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_steps = train_steps + val_steps
    all_losses = train_losses + val_losses
    min_step = min(all_steps)
    max_step = max(all_steps)

    if log_y:
        y_values = [math.log10(max(v, 1e-12)) for v in all_losses]
        y_label = "log10(loss)"
    else:
        y_values = all_losses
        y_label = "loss"

    min_y = min(y_values)
    max_y = max(y_values)
    if max_step == min_step:
        max_step += 1
    if math.isclose(max_y, min_y):
        max_y += 1.0

    def x_map(step: int) -> float:
        return left + (step - min_step) * plot_w / (max_step - min_step)

    def y_map(raw_loss: float) -> float:
        yv = math.log10(max(raw_loss, 1e-12)) if log_y else raw_loss
        return top + (max_y - yv) * plot_h / (max_y - min_y)

    train_pts = [(x_map(s), y_map(l)) for s, l in zip(train_steps, train_losses)]
    val_pts = [(x_map(s), y_map(l)) for s, l in zip(val_steps, val_losses)]

    best_step = None
    best_val = None
    if val_steps:
        i = min(range(len(val_losses)), key=lambda idx: val_losses[idx])
        best_step = val_steps[i]
        best_val = val_losses[i]

    grid_lines = []
    for i in range(6):
        y = top + i * plot_h / 5
        grid_lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb" />')

    x_ticks = []
    for i in range(6):
        step = int(min_step + i * (max_step - min_step) / 5)
        x = x_map(step)
        x_ticks.append(f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" y2="{height-bottom+6}" stroke="#374151" />')
        x_ticks.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="13" fill="#111827">{step}</text>'
        )

    y_ticks = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = max_y - i * (max_y - min_y) / 5
        label = f"{value:.2f}"
        y_ticks.append(f'<line x1="{left-6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#374151" />')
        y_ticks.append(
            f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="13" fill="#111827">{label}</text>'
        )

    best_line = ""
    best_dot = ""
    best_text = ""
    if best_step is not None and best_val is not None:
        bx = x_map(best_step)
        by = y_map(best_val)
        best_line = f'<line x1="{bx:.2f}" y1="{top}" x2="{bx:.2f}" y2="{height-bottom}" stroke="#7c3aed" stroke-dasharray="6 4" />'
        best_dot = f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="4" fill="#7c3aed" />'
        best_text = f'<text x="{bx+8:.2f}" y="{top+18}" font-size="12" fill="#7c3aed">best val @ step {best_step}</text>'

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
<text x="{left}" y="32" font-size="24" font-family="sans-serif" fill="#111827">{title or f"Training Curves: {run.name}"}</text>
{''.join(grid_lines)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827" />
{''.join(x_ticks)}
{''.join(y_ticks)}
{_polyline(train_pts, '#2563eb', 2)}
{_polyline(val_pts, '#dc2626', 2)}
{best_line}
{best_dot}
{best_text}
<text x="{(left + (width-right))/2:.2f}" y="{height-24}" text-anchor="middle" font-size="14" fill="#111827">step</text>
<text x="24" y="{(top + (height-bottom))/2:.2f}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {(top + (height-bottom))/2:.2f})" fill="#111827">{y_label}</text>
<rect x="{width-245}" y="{top+8}" width="14" height="4" fill="#2563eb" />
<text x="{width-225}" y="{top+14}" font-size="13" fill="#111827">train_loss</text>
<rect x="{width-145}" y="{top+8}" width="14" height="4" fill="#dc2626" />
<text x="{width-125}" y="{top+14}" font-size="13" fill="#111827">val_loss</text>
</svg>
'''

    out.write_text(svg, encoding="utf-8")
    return out
