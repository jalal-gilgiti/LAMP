#!/usr/bin/env python3
"""
Auto-Bound annotation and true trip counts for the dynamic-loop workload (gen_dynamic_loops.py).

For every dynamic LOOP_HEAD the planning-time estimate `lambda_est` = log N_hat is set by
Auto-Bound (lamp_dynamic.auto_bound) from planner-visible information only: the SQL call,
catalog statistics and the UDF source. Auto-Bound never sees the pattern label; the tiers
that fire on this workload are

  T2        SQL call-site literal                              (tier: sql_literal)
  T3/T4/T7  column median of the argument, clipped to C when the source bounds the
            trip-count variable with min(..., C)               (tier: col_stat_p50_capped)
  T5        collection loop over a parameter: median value length of its column
                                                                (tier: col_len_p50)
  T6        convergence loop, no derivable bound -> fallback    (tier: fallback)
  T1/T8     static bounds                                       (tier: static_exact)

The pattern label is used only to compute the true trip counts (ground truth for analysis).

`lambda_true` = log of the true mean trip count over the argument column (analysis and the
LAMP_AUTOBOUND=oracle upper bound only; never used by the deployed model).
A per-query summary is written to data/dyn_runs/autobound_<T>_<db>.json.

    python data_preparation/annotate_dynamic_loops.py [--dbs ...] [--patterns ...]
"""
import argparse
import json
import math
import os
import re
import sys

import duckdb
import networkx as nx
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data_preparation.gen_dynamic_loops import DBS, OUT, PATTERNS, db_path, pattern_root  # noqa: E402
from lamp_dynamic.auto_bound import annotate_graph  # noqa: E402
from lamp_dynamic.auto_bound.sql_arg_parser import parse_call_args  # noqa: E402
from lamp_dynamic.col_stats import extract_col_stats  # noqa: E402

CAP_RE = re.compile(r'n_dyn = max\(1, min\(int\(abs\((\w+)\)\), (\d+)\)\)')
SAMPLE_ROWS = 200_000
FALLBACK_ITERS = 10.0


def newton_iterations(v: float) -> int:
    t = abs(float(v)) + 2.0
    y, prev, n = t, 0.0, 0
    while abs(y - prev) > 1e-6:
        prev, y, n = y, 0.5 * (y + t / y), n + 1
    return n


def column_values(con, table, col):
    return [r[0] for r in con.execute(
        f'SELECT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL USING SAMPLE {SAMPLE_ROWS} ROWS').fetchall()]


def true_trips(pattern, values, cap=None):
    if not values:
        return None
    if pattern in ('T3', 'T4', 'T7'):
        return float(np.mean([max(1, min(int(abs(float(v))), cap)) for v in values]))
    if pattern == 'T5':
        return float(np.mean([len(str(v)) for v in values]))
    if pattern == 'T6':
        return float(np.mean([newton_iterations(v) for v in values]))
    return None


def annotate(db, patterns):
    con = duckdb.connect(db_path(db), read_only=True)
    col_stats = {db: extract_col_stats(db_path(db))}
    tables = [r[0] for r in con.execute('SELECT table_name FROM duckdb_tables').fetchall()]
    length_cache = {}

    def value_length(table, col):
        """Median value length of a column: a catalog statistic, collected once per column."""
        if (table, col) not in length_cache:
            vals = column_values(con, table, col)
            length_cache[(table, col)] = float(np.median([len(str(v)) for v in vals])) if vals else None
        return length_cache[(table, col)]

    for pattern in patterns:
        root = pattern_root(pattern)
        wl_path = os.path.join(root, 'parsed_plans', db, 'workload.json')
        if not os.path.exists(wl_path):
            continue
        man = json.load(open(os.path.join(OUT, 'manifest', f'{pattern}_{db}.json')))
        entry = {e['udf_name']: e for e in man['entries']}
        code = man['udf_code']
        summary = []
        for plan in json.load(open(wl_path))['parsed_plans']:
            name, sql = plan['udf']['udf_name'], plan['query']
            gpath = os.path.join(root, 'dbs', db, 'created_graphs', f'{name}.loopend.gpickle')
            if not os.path.exists(gpath):
                continue
            g = nx.read_gpickle(gpath)
            e = entry[name]
            src = ''.join(code[name])
            annotate_graph(g, sql, name, col_stats, db, known_tables=tables, udf_source=src,
                           value_length=value_length)
            param_names = [p.split(':')[0] for p in re.match(r'def \w+\((.*)\)', src).group(1).split(',') if p]
            arg = None
            if e['param'] in param_names and e['param'] != 'n_dyn':
                pos = param_names.index(e['param'])
                args = [a for a in parse_call_args(sql, name, tables) if a['pos'] == pos]
                arg = args[0] if args and args[0]['type'] == 'col_ref' and args[0].get('table') else None
            values = column_values(con, arg['table'], arg['col']) if arg else []
            cap_m = CAP_RE.search(src)
            cap = int(cap_m.group(2)) if cap_m else None

            for node, a in g.nodes(data=True):
                if a['type'] != 'LOOP_HEAD' or str(a.get('fixed_iter')) == 'True':
                    continue
                est, strategy = math.exp(a['lambda_est']), a['bound_strategy']
                # ground truth only (analysis / oracle); the estimate above is label-free
                truth = float(e['literal']) if pattern == 'T2' else true_trips(pattern, values, cap)
                if truth:
                    a['lambda_true'] = math.log(max(1.0, truth))
                summary.append(dict(udf=name, loop_type=a.get('loop_type'), strategy=strategy,
                                    est_trips=est, true_trips=truth,
                                    column=f"{arg['table']}.{arg['col']}" if arg else None))
            nx.write_gpickle(g, gpath)
        json.dump(summary, open(os.path.join(OUT, f'autobound_{pattern}_{db}.json'), 'w'), indent=1)
        ok = [s for s in summary if s['true_trips']]
        within2 = sum(1 for s in ok if max(s['est_trips'], s['true_trips']) / min(s['est_trips'], s['true_trips']) <= 2)
        print(f'[annotate] {db} {pattern}: {len(summary)} dynamic loops, estimate within 2x of truth: '
              f'{within2}/{len(ok)}', flush=True)
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dbs', nargs='+', default=DBS)
    ap.add_argument('--patterns', nargs='+', default=PATTERNS)
    args = ap.parse_args()
    for db in args.dbs:
        annotate(db, args.patterns)


if __name__ == '__main__':
    main()
