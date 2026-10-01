#!/usr/bin/env python3
"""
Planning-time overhead of LAMP (single thread, wall clock).

  per query  Auto-Bound       annotate_graph on each dynamic query (column statistics preloaded)
  per query  LAMP features    compute_lambda_v (scope composition, work summaries) + loop-scope edges,
                              on the loop-bearing benchmark queries and the dynamic queries
  per DB     column stats     extract_col_stats, collected once offline (like ANALYZE)

Graph loading and copying are excluded from the timed regions. Reported against the measured
query runtimes of the same queries.

    CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/measure_overhead.py
"""
import copy
import glob
import json
import os
import sys
import time

import networkx as nx
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data_preparation.gen_dynamic_loops import DBS, OUT, PATTERNS, db_path, pattern_root  # noqa: E402
from models.dataset.plan_graph_batching.dd_plan_batching import add_loop_scope_edges  # noqa: E402
from udf_graph.annotate_graph_info import compute_lambda_v  # noqa: E402


def timed(fn, *args, repeat=3, **kw):
    best = float('inf')
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def lamp_features(g):
    compute_lambda_v(g, 'act')
    add_loop_scope_edges(g)


def summary(name, ms, runtimes_ms=None):
    ms = np.asarray(ms)
    line = f'{name:<44}{len(ms):>7}{np.median(ms):>10.3f}{np.percentile(ms, 95):>10.3f}{ms.max():>10.3f}'
    if runtimes_ms is not None:
        line += f'{100 * np.median(ms / np.asarray(runtimes_ms)):>12.4f}%'
    return line


def main():
    import duckdb
    from lamp_dynamic.auto_bound import annotate_graph
    from lamp_dynamic.col_stats import extract_col_stats

    rows = []
    # loop-bearing benchmark queries (labels in seconds)
    import pandas as pd
    q = pd.read_csv(os.path.join(REPO, 'data', 'results', 'loo_loops_act_seed0', 'per_query_act.csv'))
    q = q[q.config == 'graceful']
    feat_ms, rt_ms = [], []
    for db, udf, label in zip(q.db, q.udf, q.label):
        g = nx.read_gpickle(os.path.join(REPO, 'data', 'loop_runs', 'duckdb_pushdown', 'dbs', db, 'created_graphs',
                                         f'{udf}.loopend.gpickle'))
        best = float('inf')
        for _ in range(3):
            h = copy.deepcopy(g)
            t0 = time.perf_counter()
            lamp_features(h)
            best = min(best, time.perf_counter() - t0)
        feat_ms.append(best * 1000.0)
        rt_ms.append(label * 1000.0)
    rows.append(summary('LAMP features, benchmark loop queries', feat_ms, rt_ms))
    print(rows[-1], flush=True)

    # dynamic queries: Auto-Bound + LAMP features; column statistics once per database
    ab_ms, dfeat_ms, drt_ms, stats_s = [], [], [], {}
    for db in DBS:
        t0 = time.perf_counter()
        col_stats = {db: extract_col_stats(db_path(db))}
        stats_s[db] = time.perf_counter() - t0
        con = duckdb.connect(db_path(db), read_only=True)
        tables = [r[0] for r in con.execute('SELECT table_name FROM duckdb_tables').fetchall()]
        con.close()
        for pattern in PATTERNS:
            wl_path = os.path.join(pattern_root(pattern), 'parsed_plans', db, 'workload.json')
            if not os.path.exists(wl_path):
                continue
            for plan in json.load(open(wl_path))['parsed_plans']:
                name, sql = plan['udf']['udf_name'], plan['query']
                gpath = os.path.join(pattern_root(pattern), 'dbs', db, 'created_graphs', f'{name}.loopend.gpickle')
                if not os.path.exists(gpath):
                    continue
                g = nx.read_gpickle(gpath)
                best_ab = best_f = float('inf')
                for _ in range(3):
                    h = copy.deepcopy(g)
                    t0 = time.perf_counter()
                    annotate_graph(h, sql, name, col_stats, db, known_tables=tables)
                    best_ab = min(best_ab, time.perf_counter() - t0)
                    h = copy.deepcopy(g)
                    t0 = time.perf_counter()
                    lamp_features(h)
                    best_f = min(best_f, time.perf_counter() - t0)
                ab_ms.append(best_ab * 1000.0)
                dfeat_ms.append(best_f * 1000.0)
                drt_ms.append(plan['plan_runtime_ms'])
        print(f'{db} done', flush=True)
    rows.append(summary('Auto-Bound, dynamic queries', ab_ms, drt_ms))
    rows.append(summary('LAMP features, dynamic queries', dfeat_ms, drt_ms))
    rows.append(summary('Auto-Bound + features, dynamic queries', np.add(ab_ms, dfeat_ms), drt_ms))
    head = f'{"":<44}{"n":>7}{"p50 ms":>10}{"p95 ms":>10}{"max ms":>10}{"p50 / query":>13}'
    stats = ', '.join(f'{db} {s:.1f}s' for db, s in stats_s.items())
    report = (f'{head}\n' + '\n'.join(rows) +
              f'\n\nOne-time column statistics per database (offline, like ANALYZE): {stats}\n'
              f'median query runtime: benchmark loops {np.median(rt_ms):.0f} ms, dynamic {np.median(drt_ms):.0f} ms\n')
    print(report)
    open(os.path.join(REPO, 'data', 'results', 'overhead.txt'), 'w').write(report)


if __name__ == '__main__':
    main()
