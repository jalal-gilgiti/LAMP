#!/usr/bin/env python3
"""
Push-down / pull-up placement pairs for the loop-bearing workload.

A pair is the same UDF query measured with the UDF pushed down (duckdb_pushdown) and pulled up
(duckdb_pullup), both with runtimes in [50 ms, 30 s], the UDF not on the col-col blacklist, and
runtimes differing by more than 5% (the GRACEFUL advisor protocol, pull_push_advisor/utils.py).
Writes, per database, workload_pairs.json next to each variant's workload.json (so the
variant's UDF graphs are found), plus data/placement/pairs.json.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP = os.path.join(REPO, 'data', 'loop_runs')
MIN_RT, MAX_RT, MIN_DIFF = 50.0, 30_000.0, 1.05


def main():
    pairs = {}
    for db in sorted(os.listdir(os.path.join(LOOP, 'duckdb_pullup', 'parsed_plans'))):
        paths = {v: os.path.join(LOOP, f'duckdb_{v}', 'parsed_plans', db, 'workload.json') for v in ('pullup', 'pushdown')}
        if not all(os.path.isfile(p) for p in paths.values()):
            continue
        wls = {v: json.load(open(p)) for v, p in paths.items()}
        plans = {v: {p['udf']['udf_name']: p for p in wl['parsed_plans'] if MIN_RT <= p['plan_runtime_ms'] <= MAX_RT}
                 for v, wl in wls.items()}
        blacklist = set()
        for v in ('pullup', 'pushdown'):
            f = os.path.join(LOOP, f'duckdb_{v}', 'dbs', db, 'created_graphs', 'udf_w_col_col_comparison.json')
            if os.path.exists(f):
                blacklist |= set(json.load(open(f)))
        names = sorted(n for n in plans['pullup'] if n in plans['pushdown'] and n not in blacklist and
                       max(plans['pullup'][n]['plan_runtime_ms'], plans['pushdown'][n]['plan_runtime_ms']) /
                       min(plans['pullup'][n]['plan_runtime_ms'], plans['pushdown'][n]['plan_runtime_ms']) > MIN_DIFF)
        if not names:
            continue
        for v, wl in wls.items():
            out = dict(wl, parsed_plans=[plans[v][n] for n in names])
            json.dump(out, open(os.path.join(os.path.dirname(paths[v]), 'workload_pairs.json'), 'w'))
        pairs[db] = {n: dict(push_ms=plans['pushdown'][n]['plan_runtime_ms'], pull_ms=plans['pullup'][n]['plan_runtime_ms'])
                     for n in names}
        print(f'{db:<15} {len(names):>4} pairs, pull-up faster in {sum(p["pull_ms"] < p["push_ms"] for p in pairs[db].values())}')
    os.makedirs(os.path.join(REPO, 'data', 'placement'), exist_ok=True)
    json.dump(pairs, open(os.path.join(REPO, 'data', 'placement', 'pairs.json'), 'w'), indent=1)
    print('total pairs', sum(len(v) for v in pairs.values()))


if __name__ == '__main__':
    main()
