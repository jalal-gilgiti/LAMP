#!/usr/bin/env python3
"""
RQ3 figure: per-pattern Q-error_50 / Q-error_95 on the measured dynamic-loop benchmark, full-workload
models (5 held-out databases, actual cardinalities), predictions calibrated on the static control T1 of
the other folds (analyze_loo_dynamic.calibrate_t1). Series: GRACEFUL, LAMP with a fixed fallback bound
instead of Auto-Bound, LAMP with Auto-Bound. Style of make_fig_perdb_lines.py.

    python figures/make_fig_dynamic_patterns.py [--lamp_config lamp_w_bw] [--results_dir ...]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import analyze_loo_dynamic as A  # noqa: E402

OUT_DIRS = [os.path.join(REPO, "figures", "out")]
os.makedirs(OUT_DIRS[0], exist_ok=True)
PATTERNS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
NAMES = {"T1": "static", "T2": "literal", "T3": "nested", "T4": "tuple", "T5": "coll.",
         "T6": "conv.", "T7": "seq.", "T8": "branch"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "data", "results", "dyn_zs_act_full"))
    ap.add_argument("--lamp_config", default="lamp_w")
    ap.add_argument("--noab_config", default="lamp_w_noab")
    ap.add_argument("--card", default="act")
    ap.add_argument("--out_name", default="fig_rq3_patterns")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman TT", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    d = A.calibrate_t1(A.load(args.results_dir, args.card))
    series = [("graceful", "GRACEFUL", "#E41A1C", "o", "-"),
              (args.noab_config, "LAMP, fixed bound", "#8C8C8C", "^", "--"),
              (args.lamp_config, "LAMP, Auto-Bound", "#1F78D1", "s", "-")]
    vals = {}
    for cfg, *_ in series:
        for m, qq in (("q50", 0.5), ("q95", 0.95)):
            vals[(cfg, m)] = []
            for t in PATTERNS:
                part = d[(d.config == cfg) & (d.pattern == t)]
                per = part.groupby("db").qerr.quantile(qq)
                vals[(cfg, m)].append(A.gm(per.values))
    x = np.arange(len(PATTERNS))
    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.15))
    for ax, m, ylab in zip(axes, ("q50", "q95"), (r"Q-error$_{50}$", r"Q-error$_{95}$")):
        g = np.array(vals[("graceful", m)])
        l = np.array(vals[(args.lamp_config, m)])
        for i in x:
            ax.plot([i, i], [l[i], g[i]], color="#B5B5B5", linewidth=0.8, zorder=1)
        for cfg, lab, col, mk, ls in series:
            ax.plot(x, vals[(cfg, m)], color=col, marker=mk, markersize=3.8, linewidth=1.2, linestyle=ls,
                    zorder=3 if cfg != "graceful" else 2, label=lab)
        ax.axvspan(-0.45, 0.45, color="#F1F1F1", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n{NAMES[t]}" for t in PATTERNS], fontsize=6.4, fontweight="bold", linespacing=0.95)
        ax.set_ylabel(ylab, fontsize=8.6, fontweight="bold", labelpad=2)
        ax.set_ylim(1.0, max(max(v) for (c, mm), v in vals.items() if mm == m) * 1.18)
        ax.yaxis.grid(True, color="#D7D7D7", linestyle="--", linewidth=0.45, zorder=0)
        ax.tick_params(axis="y", labelsize=7.4, length=2.2, width=0.7, pad=1.2)
        ax.tick_params(axis="x", length=0, pad=2)
        for t in ax.get_yticklabels():
            t.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_color("#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False,
               fontsize=7.4, handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(pad=0.25, w_pad=1.2)
    for out in OUT_DIRS:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"{args.out_name}.{ext}"), bbox_inches="tight", pad_inches=0.03, dpi=300)
    for m in ("q50", "q95"):
        print(m, {t: (round(vals[("graceful", m)][i], 2), round(vals[(args.noab_config, m)][i], 2),
                      round(vals[(args.lamp_config, m)][i], 2)) for i, t in enumerate(PATTERNS)})


if __name__ == "__main__":
    main()
