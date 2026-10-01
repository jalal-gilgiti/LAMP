#!/usr/bin/env python3
"""
Measured dynamic-loop workload (taxonomy patterns T1-T8, see dynamic_loop_rewrites.py).

For each database and pattern, loop-bearing benchmark queries of the full GRACEFUL corpus
are taken as seeds; the UDF is rewritten so that its trip count becomes input dependent,
the query is executed on the real DuckDB database (1 warm-up + 3 timed EXPLAIN ANALYZE
runs, as in GRACEFUL), parsed with GRACEFUL's parse_dd_plans and turned into UDF graphs
with GRACEFUL's prepareGraphs. Output mirrors the benchmark layout, one root per pattern:

    data/dyn_runs/<T>/duckdb_pushdown/parsed_plans/<db>/workload.json
    data/dyn_runs/<T>/duckdb_pushdown/dbs/<db>/{sql_scripts/udfs.json, func_table_dict.csv, created_graphs/}
    data/dyn_runs/manifest/<T>_<db>.json      seed plan, driving parameter, literal / threshold / cap
    data/dyn_runs/raw/<T>_<db>.json           EXPLAIN ANALYZE output of every executed query

Phases (resumable, run in order): generate, execute, parse, graphs.
Databases are copies under data/dyn_dbs (graph annotation creates temporary views).

    python data_preparation/gen_dynamic_loops.py --phase generate
    python data_preparation/gen_dynamic_loops.py --phase execute --workers 6
    python data_preparation/gen_dynamic_loops.py --phase parse
    python data_preparation/gen_dynamic_loops.py --phase graphs --workers 8
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data_preparation.dynamic_loop_rewrites import add_sql_literal, loop_structure, rewrite  # noqa: E402

SRC = os.path.join(REPO, 'data', 'loop_runs', 'duckdb_pushdown')
DB_DIR = os.path.join(REPO, 'data', 'dyn_dbs')
OUT = os.path.join(REPO, 'data', 'dyn_runs')
DBS = ['financial', 'geneea', 'baseball', 'movielens', 'ssb', 'credit', 'tpc_h', 'walmart']
PATTERNS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
STRUCTURE = {'T1': 'single', 'T2': 'single', 'T3': 'nested', 'T4': 'single', 'T5': 'single',
             'T6': 'single', 'T7': 'sequential', 'T8': 'single'}
T2_MULTIPLIERS = [0.5, 1, 2, 3]
SEEDS_PER_PATTERN = 40
SEED_RT_MS = (100.0, 10_000.0)   # seed runtime window, leaves room for the rewritten trip count
MIN_RT_MS, MAX_RT_MS = 50.0, 30_000.0
QUERY_TIMEOUT_S = 40
WARMUP, TIMED = 1, 3


def pattern_root(pattern):
    return os.path.join(OUT, pattern, 'duckdb_pushdown')


def db_path(db):
    return os.path.join(DB_DIR, f'{db}_10_1.db')


def load_seed_data(db):
    wl = json.load(open(os.path.join(SRC, 'parsed_plans', db, 'workload.json')))
    code = json.load(open(os.path.join(SRC, 'dbs', db, 'sql_scripts', 'udfs.json')))
    blacklist = set(json.load(open(os.path.join(SRC, 'dbs', db, 'created_graphs', 'udf_w_col_col_comparison.json'))))
    return wl, code, blacklist


# generate
def column_median(con, table, col):
    try:
        v = con.execute(f'SELECT median(TRY_CAST("{col}" AS DOUBLE)) FROM "{table}"').fetchone()[0]
        return None if v is None or not math.isfinite(v) else float(v)
    except Exception:
        return None


def generate(db):
    import duckdb
    from lamp_dynamic.auto_bound.sql_arg_parser import parse_call_args
    wl, code, blacklist = load_seed_data(db)
    known_tables = [t['table_name'] for t in wl['database_stats']['table_stats']]
    con = duckdb.connect(db_path(db), read_only=True)
    rng = random.Random(0)
    seeds_by_structure = {}
    for plan in wl['parsed_plans']:
        name = plan['udf']['udf_name']
        if name in blacklist or not (SEED_RT_MS[0] <= plan['plan_runtime_ms'] <= SEED_RT_MS[1]):
            continue
        lines = [ln.rstrip('\n') for ln in code[name]]
        seeds_by_structure.setdefault(loop_structure(lines), []).append(plan)

    for pattern in PATTERNS:
        seeds = list(seeds_by_structure.get(STRUCTURE[pattern], []))
        rng.shuffle(seeds)
        entries, udf_code = [], {}
        for plan in seeds:
            if len(entries) >= SEEDS_PER_PATTERN:
                break
            name, sql = plan['udf']['udf_name'], plan['query']
            lines = [ln.rstrip('\n') for ln in code[name]]
            kw = {}
            if pattern == 'T2':
                from data_preparation.dynamic_loop_rewrites import find_loops
                bound = find_loops(lines)[0]['bound']
                kw['literal'] = max(1, int(round(bound * T2_MULTIPLIERS[len(entries) % len(T2_MULTIPLIERS)])))
            if pattern == 'T8':
                from data_preparation.dynamic_loop_rewrites import _numeric_param, parse_def
                _, params, _ = parse_def(lines[0])
                x = _numeric_param(params)
                if x is None:
                    continue
                arg = [a for a in parse_call_args(sql, name, known_tables) if a['pos'] == [p for p, _ in params].index(x)]
                if not arg or arg[0]['type'] != 'col_ref' or not arg[0].get('table'):
                    continue
                kw['threshold'] = column_median(con, arg[0]['table'], arg[0]['col'])
                if kw['threshold'] is None:
                    continue
            r = rewrite(lines, pattern, **kw)
            if r is None:
                continue
            new_sql = add_sql_literal(sql, name, r['extra_arg']) if r['extra_arg'] is not None else sql
            # the UDF source ends with the create_function line, as in the benchmark udfs.json
            src = [ln for ln in r['lines'] if not ln.startswith('db_conn.')]
            udf_code[name] = [ln + '\n' for ln in src] + [f"db_conn.create_function('{name}',{name})\n"]
            entries.append(dict(udf_name=name, sql=new_sql, seed_sql=sql, seed_runtime_ms=plan['plan_runtime_ms'],
                                pattern=pattern, param=r['param'], literal=r.get('extra_arg'),
                                threshold=r.get('threshold'), cap=r.get('cap')))
        os.makedirs(os.path.join(OUT, 'manifest'), exist_ok=True)
        json.dump(dict(db=db, pattern=pattern, entries=entries, udf_code=udf_code),
                  open(os.path.join(OUT, 'manifest', f'{pattern}_{db}.json'), 'w'), indent=1)
        print(f'[generate] {db} {pattern}: {len(entries)} queries', flush=True)
    con.close()


# execute
def execute(task):
    pattern, db = task
    import duckdb
    raw_path = os.path.join(OUT, 'raw', f'{pattern}_{db}.json')
    if os.path.exists(raw_path):
        return f'[execute] {pattern} {db}: already done'
    man = json.load(open(os.path.join(OUT, 'manifest', f'{pattern}_{db}.json')))
    con = duckdb.connect(db_path(db), read_only=True)
    con.execute('PRAGMA threads=4')
    # same namespace GRACEFUL's runner (dbms/run_workload.py) uses to load UDF code
    import datetime
    ns = {'math': math, 'numpy': numpy, 'db_conn': con, 'date': datetime.date, 'time': datetime.time,
          'VARCHAR': duckdb.typing.VARCHAR, 'INTEGER': duckdb.typing.INTEGER, 'DOUBLE': duckdb.typing.DOUBLE,
          'TIMESTAMP': duckdb.typing.TIMESTAMP, 'DATE': duckdb.typing.DATE, 'TIME': duckdb.typing.TIME,
          'BOOLEAN': duckdb.typing.BOOLEAN, 'BIGINT': duckdb.typing.BIGINT, 'FLOAT': duckdb.typing.FLOAT,
          'HUGEINT': duckdb.typing.HUGEINT}
    failed_udfs = {}
    for name, src in man['udf_code'].items():
        try:
            exec(''.join(src), ns)
        except Exception as ex:
            failed_udfs[name] = str(ex)[:200]
    results, t0 = [], time.time()
    for e in man['entries']:
        if e['udf_name'] in failed_udfs:
            results.append(dict(sql=e['sql'], udf_name=e['udf_name'], analyze_plans=None, timeout=False,
                                error='udf registration: ' + failed_udfs[e['udf_name']]))
            continue
        timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
        timer.start()
        try:
            for _ in range(WARMUP):
                con.execute(e['sql']).fetchall()
            con.execute("PRAGMA enable_profiling = 'json';")
            plans = [json.loads(con.execute(f"EXPLAIN ANALYZE {e['sql']}").fetchall()[0][1]) for _ in range(TIMED)]
            con.execute('PRAGMA disable_profiling;')
            results.append(dict(sql=e['sql'], udf_name=e['udf_name'], analyze_plans=plans, timeout=False))
        except Exception as ex:
            try:
                con.execute('PRAGMA disable_profiling;')
            except Exception:
                pass
            results.append(dict(sql=e['sql'], udf_name=e['udf_name'], analyze_plans=None, timeout=True,
                                error=str(ex)[:200]))
        finally:
            timer.cancel()
    os.makedirs(os.path.join(OUT, 'raw'), exist_ok=True)
    json.dump(results, open(raw_path, 'w'))
    ok = sum(r['analyze_plans'] is not None for r in results)
    return f'[execute] {pattern} {db}: {ok}/{len(results)} ok ({(time.time() - t0) / 60:.1f} min)'


def safe_execute(task):
    try:
        return execute(task)
    except Exception as ex:
        return f'[execute] {task}: FAILED {ex}'


# parse
def _elapsed_s(plan_json):
    if 'timing' in plan_json:
        return float(plan_json['timing'])
    for child in plan_json.get('children', []):
        t = _elapsed_s(child)
        if t > 0:
            return t
    return 0.0


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(x) for x in obj]
    return obj


def parse(task):
    pattern, db = task
    from cross_db_benchmark.benchmark_tools.dbms.parse_dd_plan import parse_dd_plans
    from cross_db_benchmark.benchmark_tools.dbms.parse_filter import PredicateNode
    raw = json.load(open(os.path.join(OUT, 'raw', f'{pattern}_{db}.json')))
    man = json.load(open(os.path.join(OUT, 'manifest', f'{pattern}_{db}.json')))
    seed_wl = json.load(open(os.path.join(SRC, 'parsed_plans', db, 'workload.json')))

    query_list = []
    for r in raw:
        if r['analyze_plans'] is None:
            continue
        plans = []
        for pj in r['analyze_plans']:
            p = _ns(pj)
            p.result = str(_elapsed_s(pj))
            plans.append(p)
        query_list.append(SimpleNamespace(sql=r['sql'], analyze_plans=plans, timeout=False))
    run_stats = SimpleNamespace(
        database_stats=SimpleNamespace(
            column_stats=[SimpleNamespace(**c) for c in seed_wl['database_stats']['column_stats']],
            table_stats=[SimpleNamespace(**t) for t in seed_wl['database_stats']['table_stats']]),
        query_list=query_list, run_kwargs={'hardware': 'lamp-dyn'}, total_time_secs=0.0)
    code_dict = man['udf_code']
    parsed, _ = parse_dd_plans(run_stats, min_runtime_ms=MIN_RT_MS, max_runtime_ms=MAX_RT_MS,
                               parse_baseline=False, parse_join_conds=True, udf_code_dict=code_dict)

    def default(obj):
        if isinstance(obj, SimpleNamespace):
            return vars(obj)
        if isinstance(obj, PredicateNode):
            return obj.to_dict()
        return str(obj)

    root = pattern_root(pattern)
    os.makedirs(os.path.join(root, 'parsed_plans', db), exist_ok=True)
    with open(os.path.join(root, 'parsed_plans', db, 'workload.json'), 'w') as f:
        json.dump(parsed, f, default=default)
    scripts = os.path.join(root, 'dbs', db, 'sql_scripts')
    os.makedirs(scripts, exist_ok=True)
    json.dump(code_dict, open(os.path.join(scripts, 'udfs.json'), 'w'))
    shutil.copy(os.path.join(SRC, 'dbs', db, 'func_table_dict.csv'), os.path.join(root, 'dbs', db, 'func_table_dict.csv'))
    n = len(parsed['parsed_plans']) if isinstance(parsed, dict) else len(vars(parsed)['parsed_plans'])
    return f'[parse] {pattern} {db}: {n} plans in [{MIN_RT_MS:.0f} ms, {MAX_RT_MS / 1000:.0f} s]'


# graphs
def graphs(db):
    """One process per database: annotation writes a temporary view into the database copy."""
    from udf_graph.create_graph import prepareGraphs
    from udf_graph.dbms_wrapper import DBMSWrapper
    duckdb_kwargs = {'dir': DB_DIR, 'version': '0.10.1'}
    msgs = []
    for pattern in PATTERNS:
        root = pattern_root(pattern)
        if not os.path.exists(os.path.join(root, 'parsed_plans', db, 'workload.json')):
            continue
        graph_dir = os.path.join(root, 'dbs', db, 'created_graphs')
        if os.path.exists(os.path.join(graph_dir, 'udf_w_col_col_comparison.json')):
            msgs.append(f'{pattern}: already done')
            continue
        wrapper = DBMSWrapper(dbms='duckdb', dbms_kwargs=duckdb_kwargs, db_name=db)
        prepareGraphs(code_location=os.path.join(root, 'dbs', db, 'sql_scripts'), graph_location=graph_dir,
                      exp_folder=root, func_tab_map=os.path.join(root, 'dbs', db, 'func_table_dict.csv'),
                      db_name=db, dbms_wrapper=wrapper, duckdb_kwargs=duckdb_kwargs,
                      graph_kwargs={'add_loop_end_node': True}, card_est_assume_lazy_eval=False,
                      pullup_udf=False, udf_intermed_pos=False, skip_wj=True, skip_deepdb=True,
                      deepdb_rel_ensemble_location='', deepdb_single_ensemble_location='')
        msgs.append(f'{pattern}: {len(os.listdir(graph_dir)) - 1} graphs')
    return f'[graphs] {db}: ' + ', '.join(msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['generate', 'execute', 'parse', 'graphs'])
    ap.add_argument('--dbs', nargs='+', default=DBS)
    ap.add_argument('--patterns', nargs='+', default=PATTERNS)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    tasks = [(p, db) for db in args.dbs for p in args.patterns]
    if args.phase == 'generate':
        for db in args.dbs:
            generate(db)
    elif args.phase == 'execute':
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(safe_execute, tasks):
                print(msg, flush=True)
    elif args.phase == 'parse':
        for t in tasks:
            try:
                print(parse(t), flush=True)
            except Exception as ex:
                print(f'[parse] {t}: FAILED {ex}', flush=True)
    elif args.phase == 'graphs':
        with Pool(min(args.workers, len(args.dbs))) as pool:
            for msg in pool.imap_unordered(graphs, args.dbs):
                print(msg, flush=True)


if __name__ == '__main__':
    main()
