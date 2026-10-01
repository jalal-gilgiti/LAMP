#!/usr/bin/env python3
"""
Per-database Q-error_50 of GRACEFUL and LAMP-W on the loop-bearing workload (19 held-out databases),
one panel per cardinality source (each model trained and tested with that source), in the style of
Graceful/new generation/make_table8_est_act_compact.py (line + marker, grey connectors, dashed
geometric means). Values from the per-query result files, Q-error
floored at 0.01 s (GRACEFUL convention).

    python figures/make_fig_perdb_lines.py
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
SOURCES = [("Actual", "loo_loops_act_seed0", "act"), ("DeepDB", "loo_loops_dd_seed0", "dd"),
           ("WanderJoin", "loo_loops_wj_seed0", "wj"), ("DuckDB", "loo_loops_est_seed0", "est")]
C_G, C_L = "#E41A1C", "#1F78D1"  # colours of the original figure (GRACEFUL red, LAMP blue)


def per_db(res, card):
    df = pd.read_csv(os.path.join(REPO, "data", "results", res, f"per_query_{card}.csv"))
    df = df[df.config.isin(["graceful", "lamp_w"])].copy()
    p, l = df.pred.clip(lower=0.01), df.label.clip(lower=0.01)
    df["q"] = np.maximum(p / l, l / p)
    q50 = df.groupby(["config", "db"]).q.quantile(0.5)
    return q50.loc["graceful"], q50.loc["lamp_w"]


def gm(values):
    return float(np.exp(np.mean(np.log(values))))


def main():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman TT", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8})
    data = {lab: per_db(res, card) for lab, res, card in SOURCES}
    dbs = sorted(data["Actual"][0].index)
    x = np.arange(len(dbs))
    fig, axes = plt.subplots(len(SOURCES), 1, figsize=(6.95, 5.0), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, (lab, _, _) in zip(axes, SOURCES):
        g = np.array([data[lab][0][d] for d in dbs])
        w = np.array([data[lab][1][d] for d in dbs])
        cap = 3.3  # common vertical range; larger values are drawn at the panel edge and labelled
        top = min(max(g.max(), w.max()), cap) * 1.24
        g_plot, w_plot = np.minimum(g, top * 0.93), np.minimum(w, top * 0.93)
        for i in x:
            ax.plot([i, i], [w_plot[i], g_plot[i]], color="#A8A8A8", linewidth=0.75, zorder=1)
        ax.plot(x, g_plot, color=C_G, marker="o", markersize=3.6, linewidth=1.2, zorder=3,
                label=f"GRACEFUL (GM={gm(g):.3f})")
        ax.plot(x, w_plot, color=C_L, marker="s", markersize=3.6, linewidth=1.2, zorder=4,
                label=f"LAMP (GM={gm(w):.3f})")
        for i in x:
            for v, vp, col in ((g[i], g_plot[i], C_G), (w[i], w_plot[i], C_L)):
                if v > vp:
                    ax.text(i + 0.22, vp, f"\u2191{v:.2f}", fontsize=7.0, fontweight="bold", color=col,
                            ha="left", va="center")
        ax.axhline(gm(g), color=C_G, linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
        ax.axhline(gm(w), color=C_L, linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
        ax.set_ylim(1.0, top)
        ax.set_ylabel(f"{lab}\n" + r"Q-error$_{50}$", fontsize=8.4, fontweight="bold", labelpad=3)
        ax.yaxis.grid(True, color="#D7D7D7", linestyle="--", linewidth=0.45, zorder=0)
        ax.tick_params(axis="y", labelsize=7.8, length=2.5, width=0.7, pad=1.5)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        wins = int((w < g).sum())
        ax.legend(loc="upper center", bbox_to_anchor=(0.47, 1.0), ncol=2, frameon=True, fancybox=False, framealpha=0.95, edgecolor="#DDDDDD",
                  fontsize=7.0, borderpad=0.25, handlelength=1.4, handletextpad=0.3, columnspacing=0.8)
        ax.text(0.995, 0.93, f"{100 * (1 - gm(w) / gm(g)):+.1f}%, better on {wins}/{len(dbs)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2, fontweight="bold", color="#222222")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([d.replace("tpc_h", "tpc-h") for d in dbs], rotation=60, ha="right",
                             rotation_mode="anchor", fontsize=8.2, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.99, bottom=0.17)
    for out in OUT_DIRS:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"fig_rq1_perdb_lines.{ext}"), bbox_inches="tight", pad_inches=0.03,
                        dpi=300)
    for lab in data:
        g, w = data[lab]
        print(f"{lab:<11} GM {gm(g):.3f} -> {gm(w):.3f}  wins {int((w < g).sum())}/19")


if __name__ == "__main__":
    main()
