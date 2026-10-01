#!/usr/bin/env python3
"""
Auto-Bound annotation of the pull-up graphs of the dynamic-loop placement pairs
(gen_dynamic_placement.py). A UDF call binds its arguments to the same columns in both placements,
so Auto-Bound resolves each pull-up loop from the push-down call of the same query (the pull-up SQL
refers to the columns through the nested subquery). The true mean trip count (analysis / oracle only)
is copied from the push-down graph of the same UDF.

    python data_preparation/annotate_placement_pullup.py [--dbs ...]
"""
import json
import os
import sys

import duckdb
import networkx as nx
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data_preparation.annotate_dynamic_loops import column_values  # noqa: E402
from data_preparation.gen_dynamic_loops import DBS, OUT, PATTERNS, db_path, pattern_root  # noqa: E402
from data_preparation.gen_dynamic_placement import PLACE, pull_root  # noqa: E402
from lamp_dynamic.auto_bound import annotate_graph  # noqa: E402
from lamp_dynamic.col_stats import extract_col_stats  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dbs', nargs='+', default=DBS)
    for db in ap.parse_args().dbs:
        con = duckdb.connect(db_path(db), read_only=True)
        col_stats = {db: extract_col_stats(db_path(db))}
        tables = [r[0] for r in con.execute('SELECT table_name FROM duckdb_tables').fetchall()]
        cache = {}

        def value_length(table, col):
            if (table, col) not in cache:
                vals = column_values(con, table, col)
                cache[(table, col)] = float(np.median([len(str(v)) for v in vals])) if vals else None
            return cache[(table, col)]

        for pattern in PATTERNS:
            wl = os.path.join(pull_root(pattern), 'parsed_plans', db, 'workload.json')
            raw = os.path.join(PLACE, 'raw', f'{pattern}_{db}.json')
            if not (os.path.exists(wl) and os.path.exists(raw)):
                continue
            push_sql = {r['udf_name']: r['push_sql'] for r in json.load(open(raw))}
            code = json.load(open(os.path.join(OUT, 'manifest', f'{pattern}_{db}.json')))['udf_code']
            done = 0
            for plan in json.load(open(wl))['parsed_plans']:
                name = plan['udf']['udf_name']
                gpath = os.path.join(pull_root(pattern), 'dbs', db, 'created_graphs', f'{name}.loopend.gpickle')
                ppath = os.path.join(pattern_root(pattern), 'dbs', db, 'created_graphs', f'{name}.loopend.gpickle')
                if not os.path.exists(gpath) or name not in push_sql:
                    continue
                g = nx.read_gpickle(gpath)
                annotate_graph(g, push_sql[name], name, col_stats, db, known_tables=tables,
                               udf_source=''.join(code[name]), value_length=value_length)
                if os.path.exists(ppath):
                    truth = {a.get('lineno'): a.get('lambda_true') for _, a in nx.read_gpickle(ppath).nodes(data=True)
                             if a['type'] == 'LOOP_HEAD'}
                    for _, a in g.nodes(data=True):
                        if a['type'] == 'LOOP_HEAD' and truth.get(a.get('lineno')) is not None:
                            a['lambda_true'] = truth[a.get('lineno')]
                nx.write_gpickle(g, gpath)
                done += 1
            print(f'[annotate-pullup] {pattern} {db}: {done} graphs', flush=True)
        con.close()


if __name__ == '__main__':
    main()
