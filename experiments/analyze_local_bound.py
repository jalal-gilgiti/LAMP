#!/usr/bin/env python3
"""
Bound signal vs representation on the dynamic-loop benchmark (RQ3 protocol: models trained on static loops,
per-model calibration factor fitted on T1 of the other folds, Q-error as in analyze_loo_dynamic.py).

    model                    bound signal    representation
    graceful                 missing         local
    graceful_ablocal         Auto-Bound      local (GRACEFUL's no_iter)
    graceful_oraclelocal     oracle          local
    lamp_w_noab              fixed fallback  scope-composed
    lamp_w                   Auto-Bound      scope-composed
    lamp_w_oracle            oracle          scope-composed

Reports T2-T8 and per-pattern GM-over-databases Q50/Q95, per-database Q50, paired Wilcoxon tests over queries, and
a diagnostic split of the dynamic queries by the range of their Auto-Bound estimates relative to the no_iter values
GRACEFUL saw in training ([8, 15] in every fold): IN-RANGE (all estimates in [8, 15]), LOW-OOD (some estimate < 8,
none > 15), HIGH-OOD (some estimate > 15). The split is diagnostic only; the main results keep all queries.

    python experiments/analyze_local_bound.py --results_dir data/results/dyn_zs_act_full --card act
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'experiments'))
from analyze_loo_dynamic import calibrate_t1, gm, latest_files  # noqa: E402

MODELS = ['graceful', 'graceful_ablocal', 'graceful_oraclelocal', 'lamp_w_noab', 'lamp_w', 'lamp_w_oracle']
PATTERNS = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
TRAIN_RANGE = (8.0, 15.0)


def load(res_dir, card):
    rows = []
    for config in MODELS:
        for path in latest_files(glob.glob(os.path.join(res_dir, 'models', config, '*', '**',
                                                     f'*_pushdown_T?_{card}_per_query.csv'), recursive=True),
                                 os.path.join(res_dir, 'models')):
            db = os.path.relpath(path, os.path.join(res_dir, 'models', config)).split(os.sep)[0]
            df = pd.read_csv(path)
            df['config'], df['db'] = config, db
            df['pattern'] = re.search(rf'_pushdown_(T\d)_{card}_per_query\.csv$', path).group(1)
            df['row'] = np.arange(len(df))
            df['udf'] = df['sql'].str.extract(r'\b(func_\d+)\s*\(', expand=False)
            rows.append(df[['config', 'db', 'pattern', 'row', 'udf', 'label', 'pred']])
    data = pd.concat(rows, ignore_index=True)
    pred = np.maximum(data['pred'].values, 1e-9)  # same Q-error as analyze_loo_dynamic.load
    data['qerr'] = np.maximum(pred / data['label'].values, data['label'].values / pred)
    return data


def range_class():
    """(db, pattern, udf) -> IN-RANGE / LOW-OOD / HIGH-OOD from the Auto-Bound estimates of its dynamic loops."""
    out = {}
    for f in glob.glob(os.path.join(REPO, 'data', 'dyn_runs', 'autobound_T*_*.json')):
        m = re.search(r'autobound_(T\d)_(.+)\.json$', f)
        pattern, db = m.group(1), m.group(2)
        est = {}
        for s in json.load(open(f)):
            est.setdefault(s['udf'], []).append(s['est_trips'])
        for udf, vals in est.items():
            lo, hi = min(vals), max(vals)
            out[(db, pattern, udf)] = 'HIGH-OOD' if hi > TRAIN_RANGE[1] else 'LOW-OOD' if lo < TRAIN_RANGE[0] else 'IN-RANGE'
    return out


def gm_table(part, models):
    per = part.groupby(['config', 'db'])['qerr'].agg(q50=lambda s: s.quantile(.5), q95=lambda s: s.quantile(.95))
    return {c: (gm(per.loc[c]['q50']), gm(per.loc[c]['q95'])) for c in models if c in per.index.get_level_values(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--card', default='act')
    args = ap.parse_args()
    data = load(args.results_dir, args.card)
    models = [c for c in MODELS if c in set(data['config'])]
    dbs = sorted(db for db in set(data['db']) if all(((data['config'] == c) & (data['db'] == db)).any() for c in models))
    data = calibrate_t1(data[data['db'].isin(dbs)])
    dyn = data[data['pattern'] != 'T1']
    lines = [f'{args.results_dir}  card={args.card}  calibrated on T1  databases: {len(dbs)} ({", ".join(dbs)})',
             f'models: {models}', '',
             'GM over databases of per-database Q50 / Q95 (reduction vs GRACEFUL in parentheses)']
    base = None
    for name, part in [('T2-T8', dyn)] + [(t, dyn[dyn['pattern'] == t]) for t in PATTERNS]:
        if part.empty:
            continue
        tab = gm_table(part, models)
        base = tab['graceful']
        n = int((part['config'] == 'graceful').sum())
        cells = '  '.join(f'{c} {q50:.3f}/{q95:.2f} ({100 * (1 - q50 / base[0]):+.1f}%/{100 * (1 - q95 / base[1]):+.1f}%)'
                          for c, (q50, q95) in tab.items())
        lines.append(f'{name:<6} n={n:<5} {cells}')

    # paired tests over queries (T2-T8)
    piv = dyn.pivot_table(index=['db', 'pattern', 'row'], columns='config', values='qerr')
    lines.append('')
    lines.append('Paired Wilcoxon over T2-T8 queries (share of queries where the first model is better):')
    for a, b in [('graceful_ablocal', 'graceful'), ('lamp_w', 'graceful_ablocal'), ('lamp_w_oracle', 'graceful_oraclelocal'),
                 ('graceful_oraclelocal', 'graceful_ablocal'), ('lamp_w', 'graceful')]:
        if a in piv and b in piv:
            s = piv[[a, b]].dropna()
            lines.append(f'  {a} vs {b}: n={len(s)}  better on {100 * (s[a] < s[b]).mean():.1f}%  '
                         f'p={wilcoxon(np.log(s[a]), np.log(s[b])).pvalue:.1e}')

    # per-database Q50 (T2-T8)
    lines.append('')
    lines.append('Per-database Q50 over T2-T8:')
    per = dyn.groupby(['db', 'config'])['qerr'].median().unstack()[models]
    lines.append(per.round(3).to_string())

    # in-range / OOD diagnostic (pooled over queries: per-database class counts are small)
    cls = range_class()
    dyn = dyn.copy()
    dyn['range'] = [cls.get((d, p, u), 'no dynamic loop') for d, p, u in zip(dyn['db'], dyn['pattern'], dyn['udf'])]
    lines.append('')
    lines.append('Diagnostic split by Auto-Bound range vs GRACEFUL training no_iter [8, 15] '
                 '(pooled over queries: Q50 / Q95, n):')
    for scope, part in [('T2-T8', dyn)] + [(t, dyn[dyn['pattern'] == t]) for t in ('T3', 'T4', 'T7')]:
        for r in ['IN-RANGE', 'LOW-OOD', 'HIGH-OOD', 'no dynamic loop']:
            sub = part[part['range'] == r]
            if sub.empty:
                continue
            n = int((sub['config'] == 'graceful').sum())
            cells = '  '.join(f'{c} {sub[sub.config == c].qerr.quantile(.5):.3f}/{sub[sub.config == c].qerr.quantile(.95):.2f}'
                              for c in models)
            lines.append(f'  {scope:<6}{r:<16} n={n:<5} {cells}')
    report = '\n'.join(lines)
    print(report)
    out = os.path.join(args.results_dir, f'local_bound_{args.card}.txt')
    open(out, 'w').write(report + '\n')


if __name__ == '__main__':
    main()
