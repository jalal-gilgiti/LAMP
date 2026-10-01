#!/usr/bin/env python3
"""
Per-pattern comparison on the dynamic-loop LOO runs (experiments/run_loo_dynamic.py), plus
Auto-Bound accuracy (data/dyn_runs/autobound_<T>_<db>.json).

    python experiments/analyze_loo_dynamic.py --card_type act [--eval_card est]
    python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act --calibrate_t1

--calibrate_t1 (models trained on labels from another machine): fold <db>'s predictions are scaled by
one factor per model, the median label/pred ratio on the static control pattern T1 of the *other*
folds; T1 itself is then no longer an independent test and is reported for reference only.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = ['graceful', 'lamp_w', 'lamp_w_noab', 'lamp_w_oracle', 'lamp_w_bwtrain', 'graceful_ablocal', 'graceful_oraclelocal']
PATTERNS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']


def gm(values):
    values = [v for v in values if v == v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def latest_files(paths, root):
    """One result file per (config, db, file name): a job rerun after an interruption can leave an earlier, complete
    copy of the same predictions; keep the most recent one."""
    best = {}
    for path in paths:
        rel = os.path.relpath(path, root).split(os.sep)
        key = (rel[0], rel[1], os.path.basename(path).split('workload')[-1])
        if key not in best or os.path.getmtime(path) > os.path.getmtime(best[key]):
            best[key] = path
    return sorted(best.values())


def load(res_dir, card):
    rows = []
    for path in latest_files(glob.glob(os.path.join(res_dir, 'models', '*', '*', '**', f'*_pushdown_T?_{card}_per_query.csv'),
                                       recursive=True), os.path.join(res_dir, 'models')):
        rel = os.path.relpath(path, os.path.join(res_dir, 'models')).split(os.sep)
        df = pd.read_csv(path)
        df['config'], df['db'] = rel[0], rel[1]
        df['idx'] = np.arange(len(df))
        df['pattern'] = re.search(rf'_pushdown_(T\d)_{card}_per_query\.csv$', path).group(1)
        pred = np.maximum(df['pred'].values, 1e-9)
        df['qerr'] = np.maximum(pred / df['label'].values, df['label'].values / pred)
        rows.append(df[['config', 'db', 'pattern', 'idx', 'label', 'pred', 'qerr']])
    return pd.concat(rows, ignore_index=True) if rows else None


def calibrate_t1(data):
    """Per model and held-out database: rescale predictions by the median label / prediction ratio on T1 of the
    other databases. Factors are computed from the uncalibrated predictions (order-independent)."""
    data = data.copy()
    raw = data['pred'].copy()
    factors = {}
    for c in data['config'].unique():
        for db in data['db'].unique():
            ref = (data['config'] == c) & (data['db'] != db) & (data['pattern'] == 'T1')
            factors[(c, db)] = float(np.median(data.loc[ref, 'label'] / raw[ref]))
    for (c, db), factor in factors.items():
        m = (data['config'] == c) & (data['db'] == db)
        data.loc[m, 'pred'] = raw[m] * factor
    pred = np.maximum(data['pred'].values, 1e-9)
    data['qerr'] = np.maximum(pred / data['label'].values, data['label'].values / pred)
    return data


def autobound_accuracy():
    print('\n== Auto-Bound trip-count estimates vs truth (all dynamic loops) ==')
    print(f'{"pattern":<8}{"loops":>7}{"strategy":>26}{"median ratio":>14}{"within 2x":>11}')
    for t in PATTERNS:
        s = [x for f in glob.glob(os.path.join(REPO, 'data', 'dyn_runs', f'autobound_{t}_*.json'))
             for x in json.load(open(f)) if x.get('true_trips')]
        if not s:
            continue
        ratio = [max(x['est_trips'], x['true_trips']) / min(x['est_trips'], x['true_trips']) for x in s]
        strat = pd.Series([x['strategy'] for x in s]).value_counts().index[0]
        print(f'{t:<8}{len(s):>7}{strat:>26}{np.median(ratio):>14.2f}{np.mean(np.array(ratio) <= 2):>10.0%}')


def print_dynamic_summary(data, configs):
    """Reduction over GRACEFUL of the GM over databases of per-database Q50 / Q95, databases improved (Q50), and a
    paired Wilcoxon test over queries on log Q-error, for T2-T8 (loops with a run-time bound) and per pattern."""
    print('\nvs GRACEFUL (T2-T8: loops with a run-time bound)')
    pairs = [('graceful', c) for c in configs if c != 'graceful']
    pairs += [(b, c) for b, c in [('graceful_ablocal', 'lamp_w'), ('graceful_oraclelocal', 'lamp_w_oracle')]
              if b in configs and c in configs]
    for t in ['T2-T8'] + PATTERNS:
        part = data[data['pattern'] != 'T1'] if t == 'T2-T8' else data[data['pattern'] == t]
        if part.empty:
            continue
        per = part.groupby(['config', 'db'])['qerr'].agg(q50=lambda s: s.quantile(.5), q95=lambda s: s.quantile(.95))
        for b, c in pairs:
            b50, b95 = gm(per.loc[b]['q50']), gm(per.loc[b]['q95'])
            c50, c95 = gm(per.loc[c]['q50']), gm(per.loc[c]['q95'])
            wins = int((per.loc[c]['q50'] < per.loc[b]['q50']).sum())
            base = part[part['config'] == b].set_index(['db', 'pattern', 'idx'])['qerr']
            other = part[part['config'] == c].set_index(['db', 'pattern', 'idx'])['qerr']
            joined = pd.concat([base, other], axis=1, join='inner').dropna()
            p = wilcoxon(np.log(joined.iloc[:, 1]), np.log(joined.iloc[:, 0])).pvalue
            print(f'  {t:<6}{c:<14} vs {b:<20} {b50:.3f}/{b95:.2f} -> {c50:.3f}/{c95:.2f}  Q50 {100 * (1 - c50 / b50):+.1f}%  '
                  f'Q95 {100 * (1 - c95 / b95):+.1f}%  better on {wins}/{len(per.loc[c])} dbs  p={p:.1e}  n={len(joined)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='act')
    ap.add_argument('--eval_card', default=None)
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--calibrate_t1', action='store_true')
    args = ap.parse_args()
    card = args.eval_card or args.card_type
    res_dir = args.results_dir or os.path.join(REPO, 'data', 'results', f'dyn_{args.card_type}_seed0')
    data = load(res_dir, card)
    if data is None:
        raise SystemExit(f'no results under {res_dir}')
    configs = [c for c in CONFIGS if c in set(data['config'])]
    dbs = sorted(db for db in set(data['db']) if all(((data['config'] == c) & (data['db'] == db)).any()
                                                      for c in configs))
    data = data[data['db'].isin(dbs)]
    if args.calibrate_t1:
        data = calibrate_t1(data)
    print(f'results: {res_dir}  eval_card={card}  calibrate_t1={args.calibrate_t1}  folds complete for all configs: {len(dbs)}  ({", ".join(dbs)})')
    header = f'{"pattern":<9}{"n":>6}' + ''.join(f'{c:>22}' for c in configs)
    print('\nGM over folds of per-fold Q50 / Q95 (lower is better)')
    print(header)
    summary = {}
    for t in ['all'] + PATTERNS:
        part = data if t == 'all' else data[data['pattern'] == t]
        if part.empty:
            continue
        per = part.groupby(['config', 'db'])['qerr'].agg(q50=lambda s: s.quantile(.5), q95=lambda s: s.quantile(.95))
        cells, entry = [], {}
        for c in configs:
            q50 = gm(per.loc[c]['q50']) if c in per.index.get_level_values(0) else float('nan')
            q95 = gm(per.loc[c]['q95']) if c in per.index.get_level_values(0) else float('nan')
            entry[c] = dict(q50=q50, q95=q95)
            cells.append(f'{q50:>13.3f} / {q95:<6.2f}')
        n = int((part['config'] == configs[0]).sum())
        print(f'{t:<9}{n:>6}' + ''.join(cells))
        if 'graceful' in entry:
            g = entry['graceful']
            gains = '  '.join(f'{c}: Q50 {100 * (1 - e["q50"] / g["q50"]):+.1f}% Q95 {100 * (1 - e["q95"] / g["q95"]):+.1f}%'
                              for c, e in entry.items() if c != 'graceful')
            print(f'{"":<15}vs GRACEFUL  {gains}')
        summary[t] = entry
    print_dynamic_summary(data, configs)
    json.dump(summary, open(os.path.join(res_dir, f'summary_dyn_{card}{"_cal" if args.calibrate_t1 else ""}.json'), 'w'), indent=2)
    autobound_accuracy()


if __name__ == '__main__':
    main()
