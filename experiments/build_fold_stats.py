#!/usr/bin/env python3
"""
Fold-local feature statistics for leave-one-database-out runs.

Fits every scaler on the training databases only (the held-out database contributes
nothing), for one cardinality type:
  1. UDF-graph features: each training plan's graph is annotated with
     compute_lambda_v(graph, card_type) and all node attributes are collected.
  2. Plan-level features: gather_feature_statistics over the training workload JSONs.
The merged file is passed to train.py through LAMP_STATISTICS_FILE.

Plans are selected exactly as the loader does: runtime >= min_runtime_ms and UDF not
on the col-col comparison blacklist (created_graphs/udf_w_col_col_comparison.json).

Usage:
    python experiments/build_fold_stats.py --exclude imdb --card_type est \
        --wl_base_path data/loop_runs --out data/fold_stats/loops_est/imdb.json
"""
import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preprocessing.feature_statistics import gather_feature_statistics  # noqa: E402
from udf_graph.annotate_graph_info import compute_lambda_v  # noqa: E402

DATABASES = [
    'accidents', 'airline', 'baseball', 'basketball', 'carcinogenesis', 'consumer', 'credit',
    'employee', 'fhnk', 'financial', 'geneea', 'genome', 'hepatitis', 'imdb', 'movielens',
    'seznam', 'ssb', 'tournament', 'tpc_h', 'walmart',
]

NUMERICAL = [
    'in_rows_act', 'in_rows_est', 'in_rows_deepdb', 'in_rows_wj', 'no_params', 'no_iter', 'lambda_v',
    'has_dynamic_loop',
    'max_log_lambda_s', 'frac_amplified_s', 'mean_log_lambda_s', 'log_loop_work', 'log_total_work',
    'n_numpy', 'n_math', 'n_string', 'n_arith', 'w_numpy', 'w_math', 'w_string', 'w_arith',
]
CATEGORICAL = ['in_dts', 'ops', 'loop_part', 'cmops', 'cmdtypes', 'loop_type', 'fixed_iter', 'out_dts',
               'lib_onehot']
VECTOR = ['lib_embedding']

# Structural loop features use fixed zero-centred scaling (see the paper), not fitted values,
# so they are identical across folds and cardinality types.
FIXED_SCALING = {
    'has_dynamic_loop': 1.0,
    'max_log_lambda_s': 4.0, 'frac_amplified_s': 1.0, 'mean_log_lambda_s': 2.0,
}


def training_plans(wl_base, variant, exclude, min_runtime_ms, extra_wl=(), only_extra=False):
    """Workload paths and graph paths of every training plan the loader would use.
    extra_wl: additional training workload.json paths (e.g. dynamic-loop workloads)."""
    wl_paths, graph_paths = [], []
    candidates = [os.path.join(wl_base, variant, 'parsed_plans', db, 'workload.json')
                  for db in DATABASES if db != exclude and not only_extra] + list(extra_wl)
    for wl_path in candidates:
        if not os.path.exists(wl_path):
            continue
        db = os.path.basename(os.path.dirname(wl_path))
        graph_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(wl_path))), 'dbs', db,
                                 'created_graphs')
        blacklist_path = os.path.join(graph_dir, 'udf_w_col_col_comparison.json')
        blacklist = set(json.load(open(blacklist_path))) if os.path.exists(blacklist_path) else set()
        wl_paths.append(wl_path)
        for plan in json.load(open(wl_path))['parsed_plans']:
            name = plan['udf']['udf_name']
            if plan['plan_runtime_ms'] < min_runtime_ms or name in blacklist:
                continue
            graph_paths.append(os.path.join(graph_dir, name + '.loopend.gpickle'))
    return wl_paths, graph_paths


def udf_feature_stats(graph_paths, card_type):
    values = defaultdict(list)
    tracked = set(NUMERICAL + CATEGORICAL + VECTOR)

    def collect(path):
        graph = nx.read_gpickle(path)
        compute_lambda_v(graph, card_type=card_type)
        out = defaultdict(list)
        for _, attrs in graph.nodes(data=True):
            for key, val in attrs.items():
                if key in tracked and val is not None:
                    out[key].extend(val if isinstance(val, list) else [val])
        return out

    with ThreadPoolExecutor(16) as ex:
        for out in ex.map(collect, graph_paths):
            for key, vals in out.items():
                values[key].extend(vals)

    stats = {}
    for key in NUMERICAL:
        if key in FIXED_SCALING:
            stats[key] = {'center': 0.0, 'scale': FIXED_SCALING[key], 'max': None, 'type': 'numeric'}
        elif values.get(key):
            arr = np.array(values[key], dtype=np.float32).reshape(-1, 1)
            scaler = RobustScaler().fit(arr)
            stats[key] = {'max': float(arr.max()), 'scale': float(scaler.scale_.item()),
                          'center': float(scaler.center_.item()), 'type': 'numeric'}
    for key in CATEGORICAL:
        unique = sorted(set(values.get(key, [])), key=str)  # deterministic ids across processes
        stats[key] = {'value_dict': {v: i for i, v in enumerate(unique)}, 'no_vals': len(unique),
                      'type': 'categorical'}
    for key in VECTOR:
        if values.get(key):
            stats[key] = {'vec_len': len(values[key][0]), 'type': 'vector'}
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', required=True, help='held-out test database')
    ap.add_argument('--card_type', required=True, choices=['act', 'est', 'dd', 'wj'])
    ap.add_argument('--wl_base_path', required=True)
    ap.add_argument('--variant', default='duckdb_pushdown')
    ap.add_argument('--min_runtime_ms', type=int, default=50)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extra_wl', default='', help='comma-separated extra training workload.json paths')
    ap.add_argument('--only_extra', action='store_true', help='fit on the --extra_wl workloads only')
    args = ap.parse_args()

    wl_paths, graph_paths = training_plans(args.wl_base_path, args.variant, args.exclude, args.min_runtime_ms,
                                           [p for p in args.extra_wl.split(',') if p], args.only_extra)
    print(f'fold exclude={args.exclude} card={args.card_type}: {len(wl_paths)} training DBs, '
          f'{len(graph_paths)} training plans', flush=True)

    udf_stats = udf_feature_stats(graph_paths, args.card_type)

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        gather_feature_statistics(wl_paths, tmp_path, udf_stats_loc=None)
        plan_stats = json.load(open(tmp_path))
    finally:
        os.unlink(tmp_path)

    merged = {**plan_stats, **udf_stats}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(merged, f, indent=2)
    for key in ('in_rows_est', 'log_loop_work', 'log_total_work'):
        if key in merged:
            print(f'  {key:<16} center={merged[key]["center"]:.4f} scale={merged[key]["scale"]:.4f}', flush=True)
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
