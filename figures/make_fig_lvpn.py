#!/usr/bin/env python3
"""
Figure fig:lvpn (RQ2): per-database Q-error_50 of GRACEFUL, the per-node encoding LV-PN, and LAMP-W on
the loop-bearing workload (actual cardinalities, 19 held-out databases, seed 0). Same style and scale as
fig_rq1_perdb_lines (line + marker, grey connectors, dashed geometric means, bold Times labels).
Q-error floored at 0.01 s (GRACEFUL convention).

    python figures/make_fig_lvpn.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(REPO, "figures", "out")]
os.makedirs(OUT_DIRS[0], exist_ok=True)
# (config, label, colour, marker, z-order); GRACEFUL red and LAMP blue as in the other figures
SERIES = [("graceful", "GRACEFUL", "#E41A1C", "o", 3), ("lv_pn", "LV-PN", "#4DAF4A", "^", 2),
          ("lamp_w", "LAMP", "#1F78D1", "s", 4)]


def gm(values):
    return float(np.exp(np.mean(np.log(values))))


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman TT", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8})
    df = pd.read_csv(os.path.join(REPO, "data", "results", "loo_loops_act_seed0", "per_query_act.csv"))
    p, l = df.pred.clip(lower=0.01), df.label.clip(lower=0.01)
    df["q"] = np.maximum(p / l, l / p)
    q50 = df.groupby(["config", "db"]).q.quantile(0.5)
    dbs = sorted(q50.loc["graceful"].index)
    x = np.arange(len(dbs))
    vals = {c: np.array([q50.loc[c][d] for d in dbs]) for c, *_ in SERIES}

    fig, ax = plt.subplots(figsize=(6.95, 2.05))
    lo = np.minimum.reduce([vals[c] for c, *_ in SERIES])
    hi = np.maximum.reduce([vals[c] for c, *_ in SERIES])
    for i in x:
        ax.plot([i, i], [lo[i], hi[i]], color="#A8A8A8", linewidth=0.75, zorder=1)
    g = vals["graceful"]
    for c, lab, col, mk, z in SERIES:
        v = vals[c]
        ax.plot(x, v, color=col, marker=mk, markersize=3.8, linewidth=1.2, zorder=z,
                label=f"{lab} (GM={gm(v):.3f})")
        ax.axhline(gm(v), color=col, linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
    ax.set_ylim(1.0, hi.max() * 1.22)
    ax.set_xlim(-0.5, len(dbs) - 0.5)
    ax.set_ylabel(r"Q-error$_{50}$", fontsize=8.4, fontweight="bold", labelpad=3)
    ax.yaxis.grid(True, color="#D7D7D7", linestyle="--", linewidth=0.45, zorder=0)
    ax.tick_params(axis="y", labelsize=7.8, length=2.5, width=0.7, pad=1.5)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=True, fancybox=False, framealpha=0.95,
              edgecolor="#DDDDDD", fontsize=7.0, borderpad=0.25, handlelength=1.4, handletextpad=0.3,
              columnspacing=0.8)
    def signed(v):  # typographic minus for negative reductions
        return f"{v:+.1f}".replace("-", "\u2212")
    summary = "   ".join(f"{lab} {signed(100 * (1 - gm(vals[c]) / gm(g)))}%, better on {int((vals[c] < g).sum())}/{len(dbs)}"
                         for c, lab, *_ in SERIES[1:])
    ax.text(0.995, 0.93, summary, transform=ax.transAxes, ha="right", va="top", fontsize=7.2, fontweight="bold",
            color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("tpc_h", "tpc-h") for d in dbs], rotation=60, ha="right", rotation_mode="anchor",
                       fontsize=8.2, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.99, bottom=0.3)
    for out in OUT_DIRS:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"LV_PN.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=300)
    for c, lab, *_ in SERIES:
        print(f"{lab:<18} GM {gm(vals[c]):.3f}  better than GRACEFUL on {int((vals[c] < g).sum())}/19")


if __name__ == "__main__":
    main()
