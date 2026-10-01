#!/usr/bin/env python3
"""
Figure 4 (fig4_card_sources_compact.pdf): GRACEFUL vs LAMP-W on the loop-bearing workload,
19 leave-one-database-out folds, under four cardinality sources. Each model is trained and
tested with the same source (Table tab:rq1-card, upper block + actual row).

Values are the geometric mean over folds of the per-fold Q-error_50 / Q-error_95 / Q-error_max,
recomputed from the per-query result files; the plotting style is the one of
Graceful/test.py (the previous version of this figure).

    python figures/make_fig4_card_sources.py
"""
import os


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman TT", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIRS = [os.path.join(REPO, "figures", "out")]
os.makedirs(OUT_DIRS[0], exist_ok=True)
MIN_VAL = 0.01
PROTOCOLS = {  # label, result dir, card
    # each model trained and tested with the same cardinality source
    "same": [("Actual", "loo_loops_act_seed0", "act"), ("DuckDB", "loo_loops_est_seed0", "est"),
             ("DeepDB", "loo_loops_dd_seed0", "dd"), ("WanderJoin", "loo_loops_wj_seed0", "wj")],
    # GRACEFUL's protocol: trained on actual cardinalities, deployed with estimates
    "graceful": [("Actual", "loo_loops_act_seed0", "act"), ("DuckDB", "loo_loops_act_seed0", "est"),
                 ("DeepDB", "loo_loops_act_seed0_ddwj", "dd"), ("WanderJoin", "loo_loops_act_seed0_ddwj", "wj")],
}


def gm(values):
    return float(np.exp(np.mean(np.log(values))))


def card_data(protocol):
    rows = []
    for label, res, card in PROTOCOLS[protocol]:
        df = pd.read_csv(os.path.join(REPO, "data", "results", res, f"per_query_{card}.csv"))
        # GRACEFUL's QError convention (models/training/metrics.py): labels and predictions floored at
        # min_val = 0.01 s; only changes the two negative LAMP-W predictions under WanderJoin (Q-error_max)
        pred, runtime = df["pred"].clip(lower=MIN_VAL), df["label"].clip(lower=MIN_VAL)
        df["qerr"] = np.maximum(pred / runtime, runtime / pred)
        per = df.groupby(["config", "db"])["qerr"].agg(
            q50=lambda s: s.quantile(0.5), q95=lambda s: s.quantile(0.95), qmax="max")
        g, w = per.loc["graceful"], per.loc["lamp_w"]
        assert len(g) == len(w) == 19, (label, len(g), len(w))
        rows.append((label, gm(g.q50), gm(w.q50), gm(g.q95), gm(w.q95), gm(g.qmax), gm(w.qmax)))
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="same", choices=list(PROTOCOLS))
    ap.add_argument("--no_max", action="store_true", help="only Q-error_50 and Q-error_95 panels")
    ap.add_argument("--out_name", default="fig4_card_sources_compact")
    args = ap.parse_args()
    data = card_data(args.protocol)
    for d in data:
        print("{:<11} Q50 {:.3f} {:.3f}  Q95 {:.2f} {:.2f}  Qmax {:.2f} {:.2f}".format(*d))
    sources = [d[0] for d in data]
    grace = np.array([[d[1], d[3], d[5]] for d in data])
    lamp = np.array([[d[2], d[4], d[6]] for d in data])
    metric_labels = [r"Q-error$_{50}$", r"Q-error$_{95}$", r"Q-error$_{\mathrm{max}}$"]
    n_panels = 2 if args.no_max else 3
    grace, lamp, metric_labels = grace[:, :n_panels], lamp[:, :n_panels], metric_labels[:n_panels]

    fig, axes = plt.subplots(1, n_panels, figsize=(6.95 * n_panels / 3 + 0.2, 2.02))
    card_x = np.arange(len(sources))
    card_bw = 0.34

    for mi, (ax, ylabel) in enumerate(zip(axes, metric_labels)):
        ax.bar(card_x - card_bw / 2, grace[:, mi], card_bw, color="#E41A1C", edgecolor="white",
               linewidth=0.25, label="GRACEFUL", zorder=3)
        ax.bar(card_x + card_bw / 2, lamp[:, mi], card_bw, color="#377EB8", edgecolor="white",
               linewidth=0.25, label="LAMP", zorder=3)

        ymax = max(grace[:, mi].max(), lamp[:, mi].max())
        for xi, (gv, lv) in enumerate(zip(grace[:, mi], lamp[:, mi])):
            gain = (gv - lv) / gv * 100  # positive: LAMP lower (better); negative: LAMP higher
            ax.text(xi + card_bw / 2, lv + ymax * 0.025,
                    f"{gain:.1f}%" if gain >= 0 else f"−{-gain:.1f}%",
                    ha="center", va="bottom", fontsize=6.9, color="black", fontweight="bold")
        ax.set_ylim(0, ymax * 1.16)

        ax.set_xticks(card_x)
        ax.set_xticklabels(sources, rotation=18, ha="right", rotation_mode="anchor", fontsize=8.0,
                           color="black", fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.9, labelpad=2.0, color="black", fontweight="bold")
        ax.yaxis.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.55, zorder=0)
        ax.xaxis.grid(False)
        ax.tick_params(axis="y", labelsize=7.6, length=2.2, width=0.75, pad=1.2, labelcolor="black",
                       colors="black")
        ax.tick_params(axis="x", length=2.0, width=0.7, pad=1.2, labelcolor="black", colors="black")
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")
            tick_label.set_color("black")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.85)
            spine.set_color("#444444")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=True,
               fancybox=False, framealpha=0.95, edgecolor="#777777", borderpad=0.26, handlelength=1.0,
               handletextpad=0.34, columnspacing=0.85, fontsize=7.2)
    fig.tight_layout(pad=0.22, w_pad=1.08)
    out_dirs = OUT_DIRS
    for out in out_dirs:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"{args.out_name}.{ext}"), bbox_inches="tight", pad_inches=0.025)
    print("written to", [os.path.join(o, args.out_name + ".pdf") for o in out_dirs])


def main_two_rows(n_panels: int = 3, out_name: str = "fig4_card_sources_2rows") -> None:
    """Both protocols in one figure: row 1 same-source training, row 2 trained on actual."""
    rows_spec = [("same", "(a) Trained on same source"), ("graceful", "(b) Trained on actual")]
    metric_labels = [r"Q-error$_{50}$", r"Q-error$_{95}$", r"Q-error$_{\mathrm{max}}$"][:n_panels]
    fig, axes = plt.subplots(2, n_panels, figsize=(6.95 * n_panels / 3 + 0.3, 3.85))
    card_bw = 0.34
    for ri, (protocol, row_label) in enumerate(rows_spec):
        data = card_data(protocol)
        sources = [d[0] for d in data]
        grace = np.array([[d[1], d[3], d[5]] for d in data])[:, :n_panels]
        lamp = np.array([[d[2], d[4], d[6]] for d in data])[:, :n_panels]
        card_x = np.arange(len(sources))
        for mi in range(n_panels):
            ax = axes[ri, mi]
            ax.bar(card_x - card_bw / 2, grace[:, mi], card_bw, color="#E41A1C", edgecolor="white",
                   linewidth=0.25, label="GRACEFUL", zorder=3)
            ax.bar(card_x + card_bw / 2, lamp[:, mi], card_bw, color="#377EB8", edgecolor="white",
                   linewidth=0.25, label="LAMP", zorder=3)
            ymax = max(grace[:, mi].max(), lamp[:, mi].max())
            for xi, (gv, lv) in enumerate(zip(grace[:, mi], lamp[:, mi])):
                gain = (gv - lv) / gv * 100
                ax.text(xi + card_bw / 2, lv + ymax * 0.025,
                        f"{gain:.1f}%" if gain >= 0 else f"−{-gain:.1f}%",
                        ha="center", va="bottom", fontsize=6.6, color="black", fontweight="bold")
            ax.set_ylim(0, ymax * 1.18)
            ax.set_xticks(card_x)
            if ri == 1:
                ax.set_xticklabels(sources, rotation=18, ha="right", rotation_mode="anchor", fontsize=8.0,
                                   color="black", fontweight="bold")
            else:
                ax.set_xticklabels([])
            if mi == 0:
                ax.annotate(row_label, xy=(-0.36, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=8.0, fontweight="bold", color="black")
            ax.set_ylabel(metric_labels[mi], fontsize=8.4, labelpad=2.0, color="black", fontweight="bold")
            ax.yaxis.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.55, zorder=0)
            ax.tick_params(axis="y", labelsize=7.4, length=2.2, width=0.75, pad=1.2, colors="black")
            ax.tick_params(axis="x", length=2.0, width=0.7, pad=1.2, colors="black")
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.85)
                spine.set_color("#444444")
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=True,
               fancybox=False, framealpha=0.95, edgecolor="#777777", borderpad=0.26, handlelength=1.0,
               handletextpad=0.34, columnspacing=0.85, fontsize=7.2)
    fig.tight_layout(pad=0.22, w_pad=1.0, h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIRS[0], f"{out_name}.{ext}"), bbox_inches="tight", pad_inches=0.025)
    print("written", os.path.join(OUT_DIRS[0], out_name + ".pdf"))


def main_one_row(out_name: str = "fig4_card_sources_1x4") -> None:
    """Q50 and Q95 of both protocols in one row: (a) same source, (b) trained on actual."""
    panels = [("same", 0, "(a) Same source"), ("same", 1, "(a) Same source"),
              ("graceful", 0, "(b) Trained on actual"), ("graceful", 1, "(b) Trained on actual")]
    metric_labels = [r"Q-error$_{50}$", r"Q-error$_{95}$"]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.95))
    card_bw = 0.36
    cache = {}
    for ax, (protocol, mi, title) in zip(axes, panels):
        data = cache.setdefault(protocol, card_data(protocol))
        sources = [d[0] for d in data]
        grace = np.array([[d[1], d[3]] for d in data])[:, mi]
        lamp = np.array([[d[2], d[4]] for d in data])[:, mi]
        x = np.arange(len(sources))
        ax.bar(x - card_bw / 2, grace, card_bw, color="#E41A1C", edgecolor="white", linewidth=0.25,
               label="GRACEFUL", zorder=3)
        ax.bar(x + card_bw / 2, lamp, card_bw, color="#377EB8", edgecolor="white", linewidth=0.25,
               label="LAMP", zorder=3)
        ymax = max(grace.max(), lamp.max())
        for xi, (gv, lv) in enumerate(zip(grace, lamp)):
            gain = (gv - lv) / gv * 100
            ax.text(xi + card_bw / 2, lv + ymax * 0.025,
                    f"{gain:.1f}%" if gain >= 0 else f"−{-gain:.1f}%",
                    ha="center", va="bottom", fontsize=5.6, color="black", fontweight="bold")
        ax.set_ylim(0, ymax * 1.2)
        ax.set_title(title, fontsize=7.4, fontweight="bold", pad=2.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=25, ha="right", rotation_mode="anchor", fontsize=6.8,
                           color="black", fontweight="bold")
        ax.set_ylabel(metric_labels[mi], fontsize=7.8, labelpad=1.5, color="black", fontweight="bold")
        ax.yaxis.grid(True, color="#D0D0D0", linestyle="--", linewidth=0.55, zorder=0)
        ax.tick_params(axis="y", labelsize=6.6, length=2.0, width=0.7, pad=1.0, colors="black")
        ax.tick_params(axis="x", length=2.0, width=0.7, pad=1.0, colors="black")
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.85)
            spine.set_color("#444444")
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(pad=0.2, w_pad=0.7)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=True,
               fancybox=False, framealpha=0.95, edgecolor="#777777", borderpad=0.26, handlelength=1.0,
               handletextpad=0.34, columnspacing=0.85, fontsize=7.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIRS[0], f"{out_name}.{ext}"), bbox_inches="tight", pad_inches=0.025)
    print("written", os.path.join(OUT_DIRS[0], out_name + ".pdf"))


if __name__ == "__main__":
    import sys
    # without arguments: the 1x4 figure of the paper; with arguments: the configurable variant (see main)
    main() if len(sys.argv) > 1 else main_one_row()
