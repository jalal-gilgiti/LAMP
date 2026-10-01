#!/usr/bin/env python3
"""
Push-down / pull-up placement with GRACEFUL vs LAMP-W predictions (run_placement_loops.py).

Advisor (identical for both models): pull up iff tau * pred_pullup < pred_pushdown, tau = 1.25.
Per database: speedup = sum(T_push) / sum(T_selected) over its pairs (measured runtimes), the
paper's definition; reported with always-push (1.0), always-pull-up and the oracle (min per pair).
Aggregates: geometric mean over databases, #databases >= 1.0x, wrong decisions, and the
runtime lost by wrong decisions relative to the oracle.

    python experiments/analyze_placement.py --card_type act [--tau 1.25]
"""
import argparse
import glob
import json
import math
import os
import re

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gm(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_predictions(res_dir, card):
    preds = {}
    for path in glob.glob(os.path.join(res_dir, 'models', '*', '*', '**', f'*workload_pairs_*_{card}_per_query.csv'),
                          recursive=True):
        rel = os.path.relpath(path, os.path.join(res_dir, 'models')).split(os.sep)
        config, db = rel[0], rel[1]
        variant = 'pullup' if '_pairs_pullup_' in path else 'pushdown'
        df = pd.read_csv(path)
        df['udf'] = df['sql'].str.extract(r'\b(func_\d+)\s*\(', expand=False)
        for udf, pred in zip(df['udf'], df['pred']):
            preds[(config, db, udf, variant)] = float(pred)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', required=True)
    ap.add_argument('--tau', type=float, default=1.25)
    ap.add_argument('--seed', type=int, default=0, help='training seed of the checkpoints')
    args = ap.parse_args()
    res_dir = os.path.join(REPO, 'data', 'results',
                           f'placement_loops_{args.card_type}' + (f'_seed{args.seed}' if args.seed else ''))
    pairs = json.load(open(os.path.join(REPO, 'data', 'placement', 'pairs.json')))
    preds = load_predictions(res_dir, args.card_type)
    configs = sorted({k[0] for k in preds})

    rows = []
    for db, udfs in pairs.items():
        for udf, rt in udfs.items():
            row = dict(db=db, udf=udf, push=rt['push_ms'], pull=rt['pull_ms'])
            for c in configs:
                pu, pd_ = preds.get((c, db, udf, 'pullup')), preds.get((c, db, udf, 'pushdown'))
                row[c] = None if pu is None or pd_ is None else (args.tau * pu < pd_)
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=configs)
    dbs = sorted(df['db'].unique())

    per_db = []
    for db in dbs:
        d = df[df['db'] == db]
        push = d['push'].sum()
        entry = dict(db=db, pairs=len(d), oracle=push / d[['push', 'pull']].min(axis=1).sum(),
                     always_pull=push / d['pull'].sum())
        for c in configs:
            chosen = d['pull'].where(d[c].astype(bool), d['push'])
            entry[c] = push / chosen.sum()
            entry[f'{c}_wrong'] = int(((d[c].astype(bool)) != (d['pull'] < d['push'])).sum())
        per_db.append(entry)
    res = pd.DataFrame(per_db)

    cols = ['always_pull'] + configs + ['oracle']
    print(f'placement pairs: {len(df)} over {len(dbs)} held-out databases, tau={args.tau}, card={args.card_type}\n')
    print(f'{"db":<15}{"pairs":>6}' + ''.join(f'{c:>13}' for c in cols) + ''.join(f'{c + " wrong":>15}' for c in configs))
    for _, r in res.iterrows():
        print(f'{r["db"]:<15}{r["pairs"]:>6}' + ''.join(f'{r[c]:>13.3f}' for c in cols)
              + ''.join(f'{int(r[c + "_wrong"]):>15}' for c in configs))
    print('-' * 100)
    print(f'{"GM speedup":<21}' + ''.join(f'{gm(res[c]):>13.3f}' for c in cols))
    print(f'{"DBs >= 1.0x":<21}' + ''.join(f'{int((res[c] >= 1.0 - 1e-9).sum()):>13}' for c in cols))
    for c in configs:
        print(f'{c}: wrong decisions {int(res[c + "_wrong"].sum())}/{len(df)}')
    if 'graceful' in configs and 'lamp_w' in configs:
        g, w = gm(res['graceful']), gm(res['lamp_w'])
        print(f'LAMP-W vs GRACEFUL placement speedup: {100 * (w / g - 1):+.1f}%  '
              f'(LAMP-W better on {int((res["lamp_w"] > res["graceful"] + 1e-9).sum())}, '
              f'worse on {int((res["lamp_w"] < res["graceful"] - 1e-9).sum())} of {len(res)} databases)')
    res.to_csv(os.path.join(res_dir, f'placement_tau{args.tau}.csv'), index=False)


if __name__ == '__main__':
    main()
