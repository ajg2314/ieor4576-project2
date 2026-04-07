"""Professional chart generation for the EDA agent.

Produces consulting-quality charts (Bloomberg/McKinsey style):
- Clean white background, minimal chrome
- Professional color palette
- Proper axis formatting (billions, percentages)
- Subtle horizontal gridlines only
- End-of-line labels for multi-series line charts
- Source attribution footer
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── Professional palette (colorblind-friendly, consulting-style) ──────────────
COLORS = ["#1a56db", "#e3a008", "#0e9f6e", "#e02424", "#7e3af2", "#ff8a4c"]

# ── Global style ──────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#d1d5db",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#f3f4f6",
    "grid.linewidth": 0.8,
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
    "text.color": "#111827",
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _apply_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.yaxis.set_tick_params(pad=4)


def _format_billions(x: float, _pos=None) -> str:
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}T"
    if abs(x) >= 1:
        return f"${x:.0f}B"
    return f"${x:.2f}B"


def _format_pct(x: float, _pos=None) -> str:
    return f"{x:.1f}%"


def _save(fig: plt.Figure, name: str) -> str:
    path = ARTIFACTS_DIR / f"{name}_{uuid.uuid4().hex[:6]}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"artifacts/{path.name}"


def _add_footer(fig: plt.Figure, source: str = "Source: SEC EDGAR XBRL") -> None:
    fig.text(0.02, -0.02, source, ha="left", va="top",
             fontsize=7.5, color="#9ca3af", style="italic")


# ── Public chart functions ────────────────────────────────────────────────────

def line_chart(
    series_data: dict[str, dict[str, float]],
    title: str,
    subtitle: str = "",
    y_format: str = "billions",
    source: str = "Source: SEC EDGAR XBRL",
    filename: str = "line_chart",
) -> str:
    """
    Multi-series line chart. Professional consulting style.

    Args:
        series_data: {series_name: {x_label: y_value, ...}}
                     e.g. {"AAPL": {"2019": 260.17, "2020": 274.52, ...}}
        title: Bold chart title
        subtitle: Gray subtitle shown below title
        y_format: "billions" | "pct" | "raw"
        source: Source attribution shown at footer
        filename: Base filename (uuid suffix added automatically)

    Returns:
        Relative path to saved PNG (e.g. "artifacts/line_chart_abc123.png")
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        _apply_style(ax)

        formatter = {"billions": _format_billions, "pct": _format_pct}.get(y_format)

        all_x: list[str] = []
        for values in series_data.values():
            all_x.extend(values.keys())
        x_labels = sorted(set(all_x))
        x_pos = list(range(len(x_labels)))
        x_map = {lbl: i for i, lbl in enumerate(x_labels)}

        for idx, (name, values) in enumerate(series_data.items()):
            color = COLORS[idx % len(COLORS)]
            xs = [x_map[k] for k in sorted(values.keys()) if k in x_map]
            ys = [values[k] for k in sorted(values.keys()) if k in x_map]
            ax.plot(xs, ys, color=color, linewidth=2.2, marker="o",
                    markersize=5, zorder=3, label=name)
            # End-of-line label
            if xs:
                ax.annotate(
                    f"{name}",
                    xy=(xs[-1], ys[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=color, fontweight="bold",
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=0)
        if formatter:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))

        # Title block
        fig.suptitle(title, x=0.02, y=1.02, ha="left",
                     fontsize=13, fontweight="bold", color="#111827")
        if subtitle:
            ax.set_title(subtitle, loc="left", fontsize=9.5, color="#6b7280", pad=6)

        ax.set_xlim(-0.3, len(x_labels) - 0.3)
        _add_footer(fig, source)
        fig.tight_layout()
        return _save(fig, filename)


def bar_chart(
    series_data: dict[str, dict[str, float]],
    title: str,
    subtitle: str = "",
    y_format: str = "pct",
    source: str = "Source: SEC EDGAR XBRL",
    filename: str = "bar_chart",
) -> str:
    """
    Grouped bar chart. Good for comparing metrics across companies per year.

    Args:
        series_data: {series_name: {x_label: y_value}}
        title, subtitle, y_format, source, filename: same as line_chart
    """
    with plt.rc_context(STYLE):
        all_x: list[str] = []
        for v in series_data.values():
            all_x.extend(v.keys())
        x_labels = sorted(set(all_x))
        n_groups = len(x_labels)
        n_series = len(series_data)
        width = 0.7 / max(n_series, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.2), 5))
        _apply_style(ax)

        formatter = {"billions": _format_billions, "pct": _format_pct}.get(y_format)

        for i, (name, values) in enumerate(series_data.items()):
            color = COLORS[i % len(COLORS)]
            offset = (i - (n_series - 1) / 2) * width
            xs = [j + offset for j in range(n_groups)]
            ys = [values.get(lbl, 0) for lbl in x_labels]
            bars = ax.bar(xs, ys, width=width * 0.88, color=color,
                          alpha=0.88, label=name, zorder=3)
            # Value labels on bars
            for bar, y in zip(bars, ys):
                if y != 0:
                    label = formatter(y) if formatter else f"{y:.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(bar.get_height()) * 0.02,
                            label, ha="center", va="bottom",
                            fontsize=7.5, color="#374151")

        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(x_labels)
        if formatter:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
        ax.legend(loc="upper left")

        fig.suptitle(title, x=0.02, y=1.02, ha="left",
                     fontsize=13, fontweight="bold", color="#111827")
        if subtitle:
            ax.set_title(subtitle, loc="left", fontsize=9.5, color="#6b7280", pad=6)

        _add_footer(fig, source)
        fig.tight_layout()
        return _save(fig, filename)


def waterfall_chart(
    values: dict[str, float],
    title: str,
    subtitle: str = "",
    y_format: str = "billions",
    source: str = "Source: SEC EDGAR XBRL",
    filename: str = "waterfall",
) -> str:
    """
    Waterfall / bridge chart. Good for showing composition or YoY change.

    Args:
        values: OrderedDict of {label: value} where positive=up, negative=down.
                First and last items treated as totals (solid bars).
    """
    with plt.rc_context(STYLE):
        labels = list(values.keys())
        amounts = list(values.values())
        n = len(labels)

        running = 0.0
        bottoms, heights, colors = [], [], []
        for i, amt in enumerate(amounts):
            if i == 0 or i == n - 1:
                bottoms.append(0)
                heights.append(amt)
                colors.append(COLORS[0])
            else:
                bottoms.append(running)
                heights.append(amt)
                colors.append(COLORS[1] if amt >= 0 else COLORS[3])
            if i != n - 1:
                running += amt

        fig, ax = plt.subplots(figsize=(max(7, n * 0.9), 5))
        _apply_style(ax)
        formatter = {"billions": _format_billions, "pct": _format_pct}.get(y_format)

        bars = ax.bar(range(n), heights, bottom=bottoms, color=colors,
                      width=0.55, zorder=3, alpha=0.9)
        # Connector lines
        for i in range(n - 1):
            top = bottoms[i] + heights[i]
            ax.plot([i + 0.275, i + 0.725], [top, top],
                    color="#9ca3af", linewidth=0.8, linestyle="--")
        # Value labels
        for bar, b, h in zip(bars, bottoms, heights):
            y_pos = b + h + abs(h) * 0.03 if h >= 0 else b + h - abs(h) * 0.08
            label = formatter(h) if formatter else f"{h:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    label, ha="center", va="bottom", fontsize=8, color="#374151")

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        if formatter:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))

        fig.suptitle(title, x=0.02, y=1.02, ha="left",
                     fontsize=13, fontweight="bold", color="#111827")
        if subtitle:
            ax.set_title(subtitle, loc="left", fontsize=9.5, color="#6b7280", pad=6)

        _add_footer(fig, source)
        fig.tight_layout()
        return _save(fig, filename)
