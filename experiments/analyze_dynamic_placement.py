#!/usr/bin/env python3
"""
Placement speedups on the dynamic-loop benchmark (run_dynamic_placement.py).

Advisor (identical for all models, GRACEFUL protocol): pull up iff tau * pred_pull < pred_push, tau = 1.25.
Runtimes are the back-to-back measurements of data/dyn_place/pairs.json. Per held-out database and
pattern group: speedup = sum(T_push) / sum(T_selected); reported as the geometric mean over databases,
with always-pull-up and the oracle, the number of wrong decisions, and a paired sign count per query.

    python experiments/analyze_dynamic_placement.py --models full|loops [--card act]
"""
import argparse
import glob
import json
import math
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = ['graceful', 'lamp_w_noab', 'lamp_w']


def gm(v):
    v = [x for x in v if x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float('nan')


def load(res_dir, card):
    preds = {}
    for path in glob.glob(os.path.join(res_dir, 'models', '*', '*', '**', f'*workload_pairs_*_T?_{card}_per_query.csv'),
                          recursive=True):
        rel = os.path.relpath(path, os.path.join(res_dir, 'models')).split(os.sep)
        config, db = rel[0], rel[1]
        m = re.search(r'workload_pairs_(pullup|pushdown)_(T\d)_', path)
        variant, pattern = ('pull' if m.group(1) == 'pullup' else 'push'), m.group(2)
        df = pd.read_csv(path)
        df['udf'] = df['sql'].str.extract(r'\b(func_\d+)\s*\(', expand=False)
        for udf, pred in zip(df['udf'], df['pred']):
            preds[(config, db, pattern, udf, variant)] = max(float(pred), 0.01)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='full', choices=['full', 'loops'])
    ap.add_argument('--card', default='act')
    ap.add_argument('--tau', type=float, default=1.25)
    ap.add_argument('--root', default=None, help='benchmark root holding dyn_place/pairs.json (e.g. data/dyn_placement)')
    args = ap.parse_args()
    tag = '_' + os.path.basename(os.path.normpath(args.root)) if args.root else ''
    res_dir = os.path.join(REPO, 'data', 'results', f'dyn_place{tag}_{args.card}_{args.models}')
    pairs = json.load(open(os.path.join(REPO, args.root or 'data', 'dyn_place', 'pairs.json')))
    preds = load(res_dir, args.card)
    configs = [c for c in CONFIGS if any(k[0] == c for k in preds)]
    rows = []
    for pattern, dbs in pairs.items():
        for db, udfs in dbs.items():
            for udf, rt in udfs.items():
                r = dict(pattern=pattern, db=db, udf=udf, push=rt['push_ms'], pull=rt['pull_ms'])
                ok = True
                for c in configs:
                    pu, pd_ = preds.get((c, db, pattern, udf, 'pull')), preds.get((c, db, pattern, udf, 'push'))
                    if pu is None or pd_ is None:
                        ok = False
                        break
                    r[c] = args.tau * pu < pd_
                if ok:
                    rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no complete pairs yet')
    df['dyn'] = df.pattern != 'T1'
    lines = [f'placement pairs with predictions for all models: {len(df)} over {df.db.nunique()} databases '
             f'({args.models} models, card={args.card}, tau={args.tau})']

    def block(sub, name):
        per = []
        for db, d in sub.groupby('db'):
            e = dict(db=db, oracle=d.push.sum() / d[['push', 'pull']].min(axis=1).sum(),
                     always_pull=d.push.sum() / d.pull.sum())
            for c in configs:
                chosen = d.pull.where(d[c].astype(bool), d.push)
                e[c] = d.push.sum() / chosen.sum()
                e[c + '_wrong'] = int((d[c].astype(bool) != (d.pull < d.push)).sum())
            per.append(e)
        per = pd.DataFrame(per)
        cells = '  '.join(f'{c} {gm(per[c]):.3f}x' for c in ['always_pull'] + configs + ['oracle'])
        wrong = '  '.join(f'{c} {int(per[c + "_wrong"].sum())}' for c in configs)
        better = ''
        if 'graceful' in configs and 'lamp_w' in configs:
            better = (f'  | LAMP-W vs GRACEFUL {100 * (gm(per.lamp_w) / gm(per.graceful) - 1):+.1f}% '
                      f'(better on {int((per.lamp_w > per.graceful + 1e-9).sum())}, worse on '
                      f'{int((per.lamp_w < per.graceful - 1e-9).sum())} of {len(per)} dbs)')
        lines.append(f'{name:<10} n={len(sub):>4}  {cells}  | wrong: {wrong}{better}')
        return per

    block(df, 'all')
    block(df[~df.dyn], 'T1 static')
    per_dyn = block(df[df.dyn], 'T2-T8')
    for t in sorted(df.pattern.unique()):
        if t != 'T1':
            block(df[df.pattern == t], t)
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(res_dir, f'report_{args.card}.txt'), 'w').write(report + '\n')
    per_dyn.to_csv(os.path.join(res_dir, f'per_db_dyn_{args.card}.csv'), index=False)


if __name__ == '__main__':
    main()
