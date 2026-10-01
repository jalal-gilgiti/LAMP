#!/usr/bin/env python3
"""
RQ1 initialization robustness: GRACEFUL vs LAMP-W on the loop-bearing workload (19 held-out databases),
models trained on actual cardinalities with seeds 0, 1, 2 and evaluated with actual cardinalities and with the
DeepDB, WanderJoin and DuckDB estimates (protocol (b)). Per seed: geometric mean over databases of the per-database
Q-error_50 / Q-error_95 (labels and predictions floored at 0.01 s, GRACEFUL convention), relative reduction, number
of databases improved, and a paired Wilcoxon test over databases; then mean and standard deviation over seeds.

Seed 0 uses the evaluation files of the paper (DeepDB / WanderJoin from loo_loops_act_seed0_ddwj, the re-evaluation
with the corrected cardinality routing); seeds 1 and 2 were trained after that fix and evaluate all sources in training.

    python experiments/analyze_seeds.py [--seeds 0 1 2]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, 'data', 'results')
SOURCES = [('Actual', 'act'), ('DeepDB', 'dd'), ('WanderJoin', 'wj'), ('DuckDB', 'est')]


def files(seed, card):
    if seed == 0 and card in ('dd', 'wj'):
        base = os.path.join(RES, 'loo_loops_act_seed0_ddwj')
    else:
        base = os.path.join(RES, f'loo_loops_act_seed{seed}')
    out = {}
    for config in ('graceful', 'lamp_w'):
        for path in glob.glob(os.path.join(base, 'models', config, '*', '**', f'*_pushdown_{card}_per_query.csv'),
                              recursive=True):
            db = os.path.relpath(path, os.path.join(base, 'models', config)).split(os.sep)[0]
            out[(config, db)] = path
    return out


def per_db(seed, card):
    rows = []
    for (config, db), path in files(seed, card).items():
        df = pd.read_csv(path)
        p, l = df['pred'].clip(lower=0.01), df['label'].clip(lower=0.01)
        q = np.maximum(p / l, l / p)
        rows.append(dict(config=config, db=db, q50=q.quantile(0.5), q95=q.quantile(0.95), n=len(df)))
    t = pd.DataFrame(rows).pivot(index='db', columns='config')
    return t.dropna()


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    args = ap.parse_args()
    lines, summary = [], []
    for label, card in SOURCES:
        lines.append(f'== {label} (trained on actual cardinalities, evaluated with {card})')
        for seed in args.seeds:
            t = per_db(seed, card)
            if len(t) < 19:
                lines.append(f'  seed {seed}: only {len(t)}/19 databases complete, skipped')
                continue
            g50, w50 = t[('q50', 'graceful')], t[('q50', 'lamp_w')]
            g95, w95 = t[('q95', 'graceful')], t[('q95', 'lamp_w')]
            r50, r95 = 100 * (1 - gm(w50) / gm(g50)), 100 * (1 - gm(w95) / gm(g95))
            p50 = wilcoxon(np.log(w50), np.log(g50)).pvalue
            p95 = wilcoxon(np.log(w95), np.log(g95)).pvalue
            wins50, wins95 = int((w50 < g50).sum()), int((w95 < g95).sum())
            n = int(t[('n', 'lamp_w')].sum())
            summary.append(dict(source=label, seed=seed, g50=gm(g50), w50=gm(w50), r50=r50, wins50=wins50, p50=p50,
                                g95=gm(g95), w95=gm(w95), r95=r95, wins95=wins95, p95=p95, n=n))
            lines.append(f'  seed {seed}: Q50 {gm(g50):.3f} -> {gm(w50):.3f} ({r50:+.1f}%, {wins50}/19, p={p50:.1e})  '
                         f'Q95 {gm(g95):.2f} -> {gm(w95):.2f} ({r95:+.1f}%, {wins95}/19, p={p95:.1e})  n={n}')
        s = pd.DataFrame([r for r in summary if r['source'] == label])
        if len(s) > 1:
            lines.append(f'  mean over {len(s)} seeds: Q50 {s.g50.mean():.3f} -> {s.w50.mean():.3f}  reduction '
                         f'{s.r50.mean():+.1f} +- {s.r50.std(ddof=1):.1f}% (min {s.r50.min():+.1f}, max {s.r50.max():+.1f}); '
                         f'Q95 reduction {s.r95.mean():+.1f} +- {s.r95.std(ddof=1):.1f}% (min {s.r95.min():+.1f}, '
                         f'max {s.r95.max():+.1f}); wins Q50 {s.wins50.min()}-{s.wins50.max()}/19')
    report = '\n'.join(lines)
    print(report)
    out = os.path.join(RES, 'rq1_seeds')
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'report.txt'), 'w').write(report + '\n')
    pd.DataFrame(summary).to_csv(os.path.join(out, 'per_seed.csv'), index=False)


if __name__ == '__main__':
    main()
