"""Draw the README charts from results/.

    uv run python scripts/plots.py

Writes docs/figures/<name>-light.svg and <name>-dark.svg; the README picks one with
<picture> so each matches GitHub's theme.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
OUT = Path("docs/figures")

# Categorical slots 1-3 and ink, stepped separately for each surface.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#383835",
    },
}


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())["results"]


def tok_s(results, mode, k=None, rtt_ms=0.0):
    for r in results:
        c = r["config"]
        if c["mode"] == mode and c["k"] == k and c["rtt_ms"] == rtt_ms:
            return r["summary"]["tok_s_median"]
    raise KeyError((mode, k, rtt_ms))


def speedups(results, base, mode, ks, rtt_ms=0.0):
    return [tok_s(results, mode, k, rtt_ms) / base for k in ks]


def styled_axes(theme, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=theme["text"])
    ax.set_xlabel(xlabel, color=theme["muted"])
    ax.set_ylabel(ylabel, color=theme["muted"])
    ax.tick_params(colors=theme["muted"], length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["muted"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def baseline(ax, theme, label="target alone"):
    ax.axhline(1.0, color=theme["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(label, (0.0, 1.0), xycoords=("axes fraction", "data"), xytext=(2, 4),
                textcoords="offset points", ha="left", va="bottom", fontsize=9,
                color=theme["muted"])


def line(ax, x, y, color, label, dashed=False):
    ax.plot(x, y, color=color, linewidth=2, linestyle=(0, (5, 3)) if dashed else "-",
            marker="o", markersize=6, markeredgecolor="none", label=label)


def end_label(ax, x, y, text, theme, dy=0):
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center",
                fontsize=9, color=theme["text"])


def legend(ax, theme, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=theme["text"], **kw)
    return leg


def fast_mode(theme):
    """Local GPU: the same sweep before and after the static cache + CUDA graphs."""
    ks = [1, 3, 5, 8]
    eager, fast = load("bench-3b-eager"), load("bench-3b-fast")
    e0, f0 = tok_s(eager, "target"), tok_s(fast, "target")
    fig, ax = styled_axes(
        theme,
        "Fast mode turned a slowdown into a speedup",
        "draft length k (tokens guessed per round)",
        "speedup over target alone",
    )
    blue, orange = theme["series"][:2]
    line(ax, ks, speedups(fast, f0, "colocated", ks), blue, "co-located, fast")
    line(ax, ks, speedups(fast, f0, "split", ks), orange, "split, fast")
    line(ax, ks, speedups(eager, e0, "colocated", ks), blue, "co-located, eager", dashed=True)
    line(ax, ks, speedups(eager, e0, "split", ks), orange, "split, eager", dashed=True)
    baseline(ax, theme)
    ax.annotate("1.34×", (3, tok_s(fast, "colocated", 3) / f0), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color=theme["text"])
    ax.set_xticks(ks)
    ax.set_ylim(0.5, 1.45)
    legend(ax, theme, loc="lower left", ncol=2)
    ax.set_title("Qwen2.5 0.5B → 3B, RTX 5070 Ti, bf16. Each mode vs its own target-only baseline.",
                 loc="left", fontsize=9, color=theme["muted"])
    return fig


def a100_k(theme):
    """Cloud: A100 target, draft co-located or on its own GPU in the same datacenter."""
    ks = [1, 3, 5, 8]
    base_results = load("p3-a100-baselines")
    base = tok_s(base_results, "target")
    l4 = load("p3-split-l4")
    # k=5 hit a network spike in the sweep (37 ms/round instead of ~2.7); use the rerun.
    l4 = [r for r in l4 if r["config"]["k"] != 5] + load("p3-split-l4-k5-rerun")
    fig, ax = styled_axes(
        theme,
        "Split mode needs a fast draft GPU",
        "draft length k",
        "speedup over target alone",
    )
    blue, orange, aqua = theme["series"]
    series = [
        (speedups(base_results, base, "colocated", ks), blue, "co-located on the A100"),
        (speedups(load("p3-split-4090"), base, "split", ks), orange, "split, RTX 4090 draft"),
        (speedups(l4, base, "split", ks), aqua, "split, L4 draft"),
    ]
    for y, color, label in series:
        line(ax, ks, y, color, label)
    baseline(ax, theme)
    ax.set_xticks(ks)
    ax.set_ylim(0.3, 1.4)
    legend(ax, theme, loc="lower left")
    ax.set_title(f"Llama 3.2 1B → 3.1 8B, A100 target ({base:.0f} tok/s alone), same datacenter.",
                 loc="left", fontsize=9, color=theme["muted"])
    return fig


def delay(theme):
    """Injected delay per gRPC call (two calls per round), split mode at k=3."""
    fig, ax = styled_axes(
        theme,
        "Split mode stops paying off at a few ms of network delay",
        "added delay per network call (ms)",
        "speedup over target alone",
    )
    blue, orange, aqua = theme["series"]
    a100 = tok_s(load("p3-a100-baselines"), "target")
    local = load("bench-3b-fast-rtt")
    local_base = tok_s(local, "target")
    series = [
        (local, local_base, blue, "local: 0.5B → 3B, one RTX 5070 Ti"),
        (load("p3-split-4090-delay"), a100, orange, "cloud: 4090 draft → A100 target"),
        (load("p3-split-l4-delay"), a100, aqua, "cloud: L4 draft → A100 target"),
    ]
    for results, base, color, label in series:
        rtts = [r["config"]["rtt_ms"] for r in results if r["config"]["mode"] == "split"]
        line(ax, rtts, [tok_s(results, "split", 3, d) / base for d in rtts], color, label)
    baseline(ax, theme)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 10, 20])
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0.3, 1.4)
    legend(ax, theme, loc="upper right")
    ax.set_title("Draft length k = 3. A same-datacenter call already costs about 1 ms.",
                 loc="left", fontsize=9, color=theme["muted"])
    return fig


def regions(theme):
    """Best k at each draft location, A100 target fixed in EUR-IS-1."""
    base_results = load("p3-a100-baselines")
    base = tok_s(base_results, "target")
    rows = [("co-located on the A100", base_results, "colocated", "no network")]
    for name, where in [("p3-split-4090", "4090, same datacenter"),
                        ("p3-split-4090-EUR-IS-2", "4090 in EUR-IS-2"),
                        ("p3-split-4090-US-IL-1", "4090 in US-IL-1")]:
        rows.append((where, load(name), "split", None))
    labels, values, notes = [], [], []
    for label, results, mode, note in rows:
        best = max((r for r in results if r["config"]["mode"] == mode), key=lambda r: r["summary"]["tok_s_median"])
        s = best["summary"]
        net = s.get("network_ms_per_round")
        labels.append(label)
        values.append(s["tok_s_median"] / base)
        notes.append(f"k={best['config']['k']} · {note or f'{net:.{1 if net < 10 else 0}f} ms/round'}"
                     f" · ${s['usd_per_1m_tokens']:.2f}/1M tok")
    fig, ax = styled_axes(
        theme,
        "Splitting across regions is 2–4× slower than the target alone",
        "speedup over target alone (best k)",
        "",
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, height=0.55, color=theme["series"][0])
    for yi, v, note in zip(y, values, notes):
        ax.annotate(f"{v:.2f}×   {note}", (v, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=theme["text"])
    ax.set_yticks(list(y), labels, color=theme["text"])
    ax.axvline(1.0, color=theme["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate("target alone", (1.0, 0.0), xycoords=("data", "axes fraction"), xytext=(4, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=9, color=theme["muted"])
    ax.set_xlim(0, 2.75)
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_title(f"Llama 1B → 8B, A100 target in EUR-IS-1 ({base:.0f} tok/s, $8.09/1M tok alone).",
                 loc="left", fontsize=9, color=theme["muted"])
    return fig


CHARTS = {"fast-mode": fast_mode, "a100-k": a100_k, "delay": delay, "regions": regions}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans",
                         "svg.hashsalt": "specwire"})
    for name, draw in CHARTS.items():
        for mode, theme in THEMES.items():
            fig = draw(theme)
            fig.tight_layout()
            fig.savefig(OUT / f"{name}-{mode}.svg", metadata={"Date": None})
            plt.close(fig)
            print(OUT / f"{name}-{mode}.svg")
