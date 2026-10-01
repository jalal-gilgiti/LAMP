#!/usr/bin/env python3
"""
Dynamic-loop placement benchmark built from the loop-bearing placement queries.

Seeds are the loop-bearing filter queries of the benchmark (data/loop_runs/duckdb_pushdown):
first the push-down / pull-up placement pairs (workload_pairs.json, high placement headroom),
then further loop queries, up to PER_PATTERN queries per database and pattern. Each seed UDF is
rewritten into the taxonomy patterns T1-T8 (dynamic_loop_rewrites.py) exactly as in
gen_dynamic_loops.py, the pull-up variant is produced with GRACEFUL's pullup_udf_in_sql, and both
placements are executed back to back (gen_dynamic_placement.execute). Parsing, graph creation and
Auto-Bound annotation reuse the existing pipeline modules with the roots redirected to data/dyn_placement:

    data/dyn_placement/dyn_runs/<T>/duckdb_pushdown/...    push-down side (manifest/ holds the rewrites)
    data/dyn_placement/dyn_place/<T>/duckdb_pullup/...     pull-up side, raw/ paired runs, pairs.json

    python data_preparation/gen_placement_dynamic.py --phase generate|execute|parse|graphs|annotate|pairs
"""
import argparse
import json
import os
import random
import shutil
import sys
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import data_preparation.gen_dynamic_loops as G  # noqa: E402
import data_preparation.gen_dynamic_placement as P  # noqa: E402
from data_preparation.dynamic_loop_rewrites import (_numeric_param, add_sql_literal, find_loops,  # noqa: E402
                                                    loop_structure, parse_def, rewrite)

ROOT = os.path.join(REPO, 'data', 'dyn_placement')
G.OUT = os.path.join(ROOT, 'dyn_runs')
P.PLACE = os.path.join(ROOT, 'dyn_place')
COPY_DIR = os.path.join(ROOT, 'db_copies')
PER_PATTERN = 80
SEED_RT_MS = (50.0, 10_000.0)


def generate(db):
    import duckdb
    from cross_db_benchmark.benchmark_tools.utils import pullup_udf_in_sql
    from lamp_dynamic.auto_bound.sql_arg_parser import parse_call_args
    wl, code, blacklist = G.load_seed_data(db)
    pair_seeds = {p['udf']['udf_name'] for p in json.load(open(
        os.path.join(G.SRC, 'parsed_plans', db, 'workload_pairs.json')))['parsed_plans']}
    known_tables = [t['table_name'] for t in wl['database_stats']['table_stats']]
    con = duckdb.connect(G.db_path(db), read_only=True)
    rng = random.Random(0)
    by_structure = {}
    for plan in wl['parsed_plans']:
        name = plan['udf']['udf_name']
        if name in blacklist:
            continue
        if name not in pair_seeds and not (SEED_RT_MS[0] <= plan['plan_runtime_ms'] <= SEED_RT_MS[1]):
            continue
        if not pullup_udf_in_sql(plan['query'])[1]:
            continue
        by_structure.setdefault(loop_structure([ln.rstrip('\n') for ln in code[name]]), []).append(plan)
    for plans in by_structure.values():  # placement-pair seeds first, then the rest in random order
        rng.shuffle(plans)
        plans.sort(key=lambda p: p['udf']['udf_name'] not in pair_seeds)

    for pattern in G.PATTERNS:
        entries, udf_code = [], {}
        for plan in by_structure.get(G.STRUCTURE[pattern], []):
            if len(entries) >= PER_PATTERN:
                break
            name, sql = plan['udf']['udf_name'], plan['query']
            if name in udf_code:
                continue
            lines = [ln.rstrip('\n') for ln in code[name]]
            kw = {}
            if pattern == 'T2':
                kw['literal'] = max(1, int(round(find_loops(lines)[0]['bound']
                                                 * G.T2_MULTIPLIERS[len(entries) % len(G.T2_MULTIPLIERS)])))
            if pattern == 'T8':
                _, params, _ = parse_def(lines[0])
                x = _numeric_param(params)
                if x is None:
                    continue
                arg = [a for a in parse_call_args(sql, name, known_tables) if a['pos'] == [p for p, _ in params].index(x)]
                if not arg or arg[0]['type'] != 'col_ref' or not arg[0].get('table'):
                    continue
                kw['threshold'] = G.column_median(con, arg[0]['table'], arg[0]['col'])
                if kw['threshold'] is None:
                    continue
            r = rewrite(lines, pattern, **kw)
            if r is None:
                continue
            new_sql = add_sql_literal(sql, name, r['extra_arg']) if r['extra_arg'] is not None else sql
            src = [ln for ln in r['lines'] if not ln.startswith('db_conn.')]
            udf_code[name] = [ln + '\n' for ln in src] + [f"db_conn.create_function('{name}',{name})\n"]
            entries.append(dict(udf_name=name, sql=new_sql, seed_sql=sql, seed_runtime_ms=plan['plan_runtime_ms'],
                                pattern=pattern, param=r['param'], literal=r.get('extra_arg'),
                                threshold=r.get('threshold'), cap=r.get('cap'), placement_seed=name in pair_seeds))
        os.makedirs(os.path.join(G.OUT, 'manifest'), exist_ok=True)
        json.dump(dict(db=db, pattern=pattern, entries=entries, udf_code=udf_code),
                  open(os.path.join(G.OUT, 'manifest', f'{pattern}_{db}.json'), 'w'), indent=1)
        print(f'[generate] {db} {pattern}: {len(entries)} queries '
              f'({sum(e["placement_seed"] for e in entries)} from placement pairs)', flush=True)
    con.close()


def parse(task):
    """Both sides from the same paired runs: push-down via gen_dynamic_loops.parse, pull-up via gen_dynamic_placement.parse."""
    pattern, db = task
    raw = json.load(open(os.path.join(P.PLACE, 'raw', f'{pattern}_{db}.json')))
    push_raw = [dict(sql=r['push_sql'], udf_name=r['udf_name'], analyze_plans=r['push'], timeout=False)
                for r in raw if r['push'] is not None and r['pull'] is not None]
    os.makedirs(os.path.join(G.OUT, 'raw'), exist_ok=True)
    json.dump(push_raw, open(os.path.join(G.OUT, 'raw', f'{pattern}_{db}.json'), 'w'))
    return G.parse(task) + ' | ' + P.parse(task)


def graphs(task):
    """One private database copy per (db, pattern, side): graph creation writes temporary views."""
    db, pattern, side = task
    from udf_graph.create_graph import prepareGraphs
    from udf_graph.dbms_wrapper import DBMSWrapper
    root = G.pattern_root(pattern) if side == 'push' else P.pull_root(pattern)
    if not os.path.exists(os.path.join(root, 'parsed_plans', db, 'workload.json')):
        return f'[graphs] {db} {pattern} {side}: no workload'
    graph_dir = os.path.join(root, 'dbs', db, 'created_graphs')
    if os.path.exists(os.path.join(graph_dir, 'udf_w_col_col_comparison.json')):
        return f'[graphs] {db} {pattern} {side}: already done'
    copy_dir = os.path.join(COPY_DIR, f'{db}_{pattern}_{side}')
    os.makedirs(copy_dir, exist_ok=True)
    shutil.copy(G.db_path(db), os.path.join(copy_dir, f'{db}_10_1.db'))
    try:
        kw = {'dir': copy_dir, 'version': '0.10.1'}
        wrapper = DBMSWrapper(dbms='duckdb', dbms_kwargs=kw, db_name=db)
        prepareGraphs(code_location=os.path.join(root, 'dbs', db, 'sql_scripts'), graph_location=graph_dir,
                      exp_folder=root, func_tab_map=os.path.join(root, 'dbs', db, 'func_table_dict.csv'),
                      db_name=db, dbms_wrapper=wrapper, duckdb_kwargs=kw,
                      graph_kwargs={'add_loop_end_node': True}, card_est_assume_lazy_eval=False,
                      pullup_udf=side == 'pull', udf_intermed_pos=False, skip_wj=True, skip_deepdb=True,
                      deepdb_rel_ensemble_location='', deepdb_single_ensemble_location='')
    finally:
        shutil.rmtree(copy_dir, ignore_errors=True)
    return f'[graphs] {db} {pattern} {side}: {len(os.listdir(graph_dir)) - 1} graphs'


def safe(fn):
    def run(task):
        try:
            return fn(task)
        except Exception as ex:
            return f'[{fn.__name__}] {task}: FAILED {type(ex).__name__}: {ex}'
    return run


def _safe_graphs(task):
    return safe(graphs)(task)


def _safe_annotate(db):
    import data_preparation.annotate_dynamic_loops as A
    import data_preparation.annotate_placement_pullup as AP
    A.OUT = G.OUT
    AP.OUT, AP.PLACE = G.OUT, P.PLACE
    try:
        A.annotate(db, G.PATTERNS)
        sys.argv = ['annotate_placement_pullup', '--dbs', db]
        AP.main()
        return f'[annotate] {db}: done'
    except Exception as ex:
        return f'[annotate] {db}: FAILED {type(ex).__name__}: {ex}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['generate', 'execute', 'parse', 'graphs', 'annotate', 'pairs'])
    ap.add_argument('--dbs', nargs='+', default=G.DBS)
    ap.add_argument('--patterns', nargs='+', default=G.PATTERNS)
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    tasks = [(p, db) for db in args.dbs for p in args.patterns]
    if args.phase == 'generate':
        for db in args.dbs:
            generate(db)
    elif args.phase == 'execute':
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(P.safe_execute, tasks):
                print(msg, flush=True)
    elif args.phase == 'parse':
        for t in tasks:
            print(safe(parse)(t), flush=True)
    elif args.phase == 'graphs':
        gt = [(db, p, s) for db in args.dbs for p in args.patterns for s in ('push', 'pull')]
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(_safe_graphs, gt):
                print(msg, flush=True)
    elif args.phase == 'annotate':
        with Pool(min(args.workers, len(args.dbs))) as pool:
            for msg in pool.imap_unordered(_safe_annotate, args.dbs):
                print(msg, flush=True)
    else:
        P.pairs()


if __name__ == '__main__':
    main()
