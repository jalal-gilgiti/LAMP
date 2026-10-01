#!/usr/bin/env python3
"""
Placement with seed ensembles: for each model (GRACEFUL, LAMP-W), the predictions of the checkpoints trained with
seeds 0, 1, 2 are averaged in log space (geometric mean) per candidate plan; the unchanged advisor
(pull up iff tau * pred_pullup < pred_pushdown, tau = 1.25) then decides every pair. Identical treatment for both
models; same pairs, runtimes and metric as analyze_placement.py. The per-seed results are printed for reference.

    python experiments/analyze_placement_ensemble.py --card_type est
"""
import argparse
import json
import math
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'experiments'))
from analyze_placement import gm, load_predictions  # noqa: E402

CONFIGS = ['graceful', 'lamp_w']


def evaluate(pairs, preds, tau):
    rows = []
    for db, udfs in pairs.items():
        for udf, rt in udfs.items():
            row = dict(db=db, udf=udf, push=rt['push_ms'], pull=rt['pull_ms'])
            for c in CONFIGS:
                pu, pd_ = preds.get((c, db, udf, 'pullup')), preds.get((c, db, udf, 'pushdown'))
                row[c] = None if pu is None or pd_ is None else (tau * pu < pd_)
            rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=CONFIGS)
    per_db = []
    for db, d in df.groupby('db'):
        push = d['push'].sum()
        e = dict(db=db, oracle=push / d[['push', 'pull']].min(axis=1).sum())
        for c in CONFIGS:
            e[c] = push / d['pull'].where(d[c].astype(bool), d['push']).sum()
            e[c + '_wrong'] = int((d[c].astype(bool) != (d['pull'] < d['push'])).sum())
        per_db.append(e)
    res = pd.DataFrame(per_db)
    g, w = gm(res['graceful']), gm(res['lamp_w'])
    return dict(n=len(df), g=g, w=w, gain=100 * (w / g - 1), oracle=gm(res['oracle']),
                better=int((res['lamp_w'] > res['graceful'] + 1e-9).sum()),
                worse=int((res['lamp_w'] < res['graceful'] - 1e-9).sum()),
                g_wrong=int(res['graceful_wrong'].sum()), w_wrong=int(res['lamp_w_wrong'].sum()),
                g_ge1=int((res['graceful'] >= 1 - 1e-9).sum()), w_ge1=int((res['lamp_w'] >= 1 - 1e-9).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='est')
    ap.add_argument('--eval_card', default=None, help='cardinalities the candidates are costed with (default: card_type)')
    ap.add_argument('--tau', type=float, default=1.25)
    args = ap.parse_args()
    eval_card = args.eval_card or args.card_type
    pairs = json.load(open(os.path.join(REPO, 'data', 'placement', 'pairs.json')))
    seeds = {}
    for s in (0, 1, 2):
        res_dir = os.path.join(REPO, 'data', 'results', f'placement_loops_{args.card_type}' + (f'_seed{s}' if s else ''))
        seeds[s] = load_predictions(res_dir, eval_card)
    fmt = lambda r: (f"n={r['n']}  GRACEFUL {r['g']:.3f}x  LAMP {r['w']:.3f}x  ({r['gain']:+.1f}%, LAMP better on "
                     f"{r['better']}, worse on {r['worse']} DBs)  wrong {r['g_wrong']} vs {r['w_wrong']}  "
                     f">=1x DBs {r['g_ge1']} vs {r['w_ge1']}  oracle {r['oracle']:.3f}x")
    for s, p in seeds.items():
        print(f'seed {s}:   ' + fmt(evaluate(pairs, p, args.tau)))
    common = set(seeds[0]) & set(seeds[1]) & set(seeds[2])
    ens = {k: math.exp(sum(math.log(max(seeds[s][k], 1e-9)) for s in seeds) / 3) for k in common}
    print(f'ensemble: ' + fmt(evaluate(pairs, ens, args.tau)))


if __name__ == '__main__':
    main()
