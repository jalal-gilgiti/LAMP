#!/usr/bin/env python3
"""
Compare GRACEFUL and LAMP on the loop-only LOO runs (experiments/run_loo_loops.py).

For every test database reads the per-query CSV of the requested evaluation card,
assigns each plan its loop class from the UDF graph (single / sequential / nested),
and reports Q-error per database and class:
  * per-database Q50 / Q95 and their geometric mean over databases (paper convention),
  * databases where each LAMP variant beats GRACEFUL,
  * pooled Q50 / Q95 per loop class over all test plans (nested and sequential are
    too small per database for stable medians).

Usage:
    python experiments/analyze_loo_loops.py --card_type est [--eval_card est]
"""
import argparse
import glob
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, 'data', 'loop_runs', 'duckdb_pushdown')
CLASSES = ['no_loop', 'any_loop', 'single', 'sequential', 'nested']


def loop_class(graph):
    heads = [n for n, a in graph.nodes(data=True) if a['type'] == 'LOOP_HEAD']
    if not heads:
        return 'no_loop'
    if any(str(graph.nodes[h].get('loop_part')) == 'True' for h in heads):
        return 'nested'
    return 'single' if len(heads) == 1 else 'sequential'


def load_loop_classes(base):
    cache = os.path.join(base, 'loop_classes.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    classes = {}
    for db in sorted(os.listdir(os.path.join(base, 'dbs'))):
        graphs = glob.glob(os.path.join(base, 'dbs', db, 'created_graphs', '*.loopend.gpickle'))
        with ThreadPoolExecutor(16) as ex:
            labels = ex.map(lambda p: loop_class(nx.read_gpickle(p)), graphs)
        classes[db] = {os.path.basename(p).split('.')[0]: c for p, c in zip(graphs, labels)}
    json.dump(classes, open(cache, 'w'))
    return classes


def qerr(label, pred):
    pred = np.maximum(pred, 1e-9)
    return np.maximum(pred / label, label / pred)


def gm(values):
    values = [v for v in values if v == v]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='est', help='cardinality the models were trained with')
    ap.add_argument('--eval_card', default=None, help='evaluation pass to read (default: same as card_type)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--results_dir', default=None)
    ap.add_argument('--wl_base', default=os.path.dirname(BASE),
                    help='workload root the runs used (data/full_runs for the full corpus)')
    args = ap.parse_args()
    eval_card = args.eval_card or args.card_type
    res_dir = args.results_dir or os.path.join(REPO, 'data', 'results', f'loo_loops_{args.card_type}_seed{args.seed}')
    classes = load_loop_classes(os.path.join(args.wl_base, 'duckdb_pushdown'))

    rows = []
    for path in glob.glob(os.path.join(res_dir, 'models', '*', '*', '**', f'*_pushdown_{eval_card}_per_query.csv'),
                          recursive=True):
        rel = os.path.relpath(path, os.path.join(res_dir, 'models')).split(os.sep)
        config, db = rel[0], rel[1]
        df = pd.read_csv(path)
        df['udf'] = df['sql'].str.extract(r'\b(func_\d+)\s*\(', expand=False)
        df['loop_class'] = df['udf'].map(classes.get(db, {}))
        df['qerr'] = qerr(df['label'].values, df['pred'].values)
        df['config'], df['db'] = config, db
        rows.append(df[['config', 'db', 'udf', 'loop_class', 'label', 'pred', 'qerr']])
    if not rows:
        raise SystemExit(f'no per-query results found under {res_dir}')
    data = pd.concat(rows, ignore_index=True)
    unmapped = data['loop_class'].isna().sum()
    if unmapped:
        print(f'warning: {unmapped} predictions without a loop class')

    configs = [c for c in ['graceful', 'lamp_w', 'lamp_w_bw', 'abl_noscope', 'abl_noops', 'abl_nooplam', 'abl_noedge', 'lv_pn'] if c in set(data['config'])]
    dbs = sorted(set(data['db']))
    complete = [db for db in dbs if all(((data['config'] == c) & (data['db'] == db)).any() for c in configs)]
    data = data[data['db'].isin(complete)]
    print(f'results: {res_dir}  eval_card={eval_card}  folds complete for all configs: {len(complete)}/{len(dbs)}\n')

    summary = {}
    for subset in ['all'] + CLASSES:
        if subset == 'all':
            part = data
        elif subset == 'any_loop':
            part = data[data['loop_class'].isin(['single', 'sequential', 'nested'])]
        else:
            part = data[data['loop_class'] == subset]
        if part.empty:
            continue
        per_db = part.groupby(['config', 'db'])['qerr'].agg(
            q50=lambda s: s.quantile(0.5), q95=lambda s: s.quantile(0.95), n='size').reset_index()
        print(f'== {subset} loops ==')
        header = f'{"db":<16}{"n":>6}' + ''.join(f'{c + " Q50":>15}' for c in configs)
        print(header)
        for db in complete:
            r = per_db[per_db['db'] == db].set_index('config')
            if r.empty:
                continue
            print(f'{db:<16}{int(r["n"].iloc[0]):>6}' + ''.join(f'{r.loc[c, "q50"]:>15.3f}' if c in r.index else f'{"-":>15}' for c in configs))
        entry = {}
        for c in configs:
            d = per_db[per_db['config'] == c].set_index('db')
            entry[c] = {'gm_q50': gm(d['q50']), 'gm_q95': gm(d['q95']),
                        'pooled_q50': float(part[part['config'] == c]['qerr'].quantile(0.5)),
                        'pooled_q95': float(part[part['config'] == c]['qerr'].quantile(0.95)),
                        'n': int((part['config'] == c).sum())}
        base = per_db[per_db['config'] == 'graceful'].set_index('db')['q50']
        for c in configs:
            if c == 'graceful':
                continue
            mine = per_db[per_db['config'] == c].set_index('db')['q50']
            common = base.index.intersection(mine.index)
            entry[c]['wins_q50'] = f'{int((mine[common] < base[common]).sum())}/{len(common)}'
            entry[c]['gain_gm_q50_pct'] = 100 * (1 - entry[c]['gm_q50'] / entry['graceful']['gm_q50'])
            entry[c]['gain_gm_q95_pct'] = 100 * (1 - entry[c]['gm_q95'] / entry['graceful']['gm_q95'])
        print('-' * len(header))
        for c in configs:
            e = entry[c]
            extra = '' if c == 'graceful' else (f'  gain Q50 {e["gain_gm_q50_pct"]:+.1f}%  Q95 {e["gain_gm_q95_pct"]:+.1f}%'
                                                f'  wins {e["wins_q50"]}')
            print(f'{c:<10} GM Q50={e["gm_q50"]:.3f}  GM Q95={e["gm_q95"]:.3f}  pooled Q50={e["pooled_q50"]:.3f} '
                  f'Q95={e["pooled_q95"]:.3f}  n={e["n"]}{extra}')
        print()
        summary[subset] = entry

    out = os.path.join(res_dir, f'summary_{eval_card}.json')
    json.dump(summary, open(out, 'w'), indent=2)
    data.to_csv(os.path.join(res_dir, f'per_query_{eval_card}.csv'), index=False)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
