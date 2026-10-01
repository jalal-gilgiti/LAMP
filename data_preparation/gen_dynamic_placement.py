#!/usr/bin/env python3
"""
Push-down / pull-up placement pairs for the measured dynamic-loop workload (gen_dynamic_loops.py).

For every dynamic query whose UDF is a filter, the pull-up variant is produced with GRACEFUL's
pullup_udf_in_sql (for T2 from the seed query, then the SQL literal is appended as in the
push-down variant). Both variants are executed back to back in the same session (1 warm-up +
3 timed EXPLAIN ANALYZE runs each), so the pair is measured under the same machine load.

Outputs (the push-down side reuses the graphs of data/dyn_runs):
    data/dyn_place/raw/<T>_<db>.json                          paired EXPLAIN ANALYZE output
    data/dyn_place/<T>/duckdb_pullup/parsed_plans/<db>/workload.json, dbs/<db>/created_graphs/
    data/dyn_runs/<T>/duckdb_pushdown/parsed_plans/<db>/workload_pairs.json   paired push-down plans
    data/dyn_place/<T>/duckdb_pullup/parsed_plans/<db>/workload_pairs.json    paired pull-up plans
    data/dyn_place/pairs.json                                  {T: {db: {udf: {push_ms, pull_ms}}}}

    python data_preparation/gen_dynamic_placement.py --phase execute --workers 3
    python data_preparation/gen_dynamic_placement.py --phase parse
    python data_preparation/gen_dynamic_placement.py --phase graphs --workers 4
    python data_preparation/gen_dynamic_placement.py --phase pairs
"""
import argparse
import json
import math
import os
import shutil
import sys
import threading
import time
from multiprocessing import Pool
from types import SimpleNamespace

import numpy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import data_preparation.gen_dynamic_loops as G  # noqa: E402
from data_preparation.dynamic_loop_rewrites import add_sql_literal  # noqa: E402

PLACE = os.path.join(REPO, 'data', 'dyn_place')
MIN_DIFF = 1.05  # runtimes must differ by more than 5% (GRACEFUL advisor protocol)


def pull_root(pattern):
    return os.path.join(PLACE, pattern, 'duckdb_pullup')


def pullup_sql(entry):
    from cross_db_benchmark.benchmark_tools.utils import pullup_udf_in_sql
    if entry['literal'] is not None:  # T2: convert the seed query, then append the literal argument
        q, ok = pullup_udf_in_sql(entry['seed_sql'])
        return (add_sql_literal(q, entry['udf_name'], entry['literal']), ok) if ok else (None, False)
    q, ok = pullup_udf_in_sql(entry['sql'])
    return (q, ok) if ok else (None, False)


def _run(con, sql):
    for _ in range(G.WARMUP):
        con.execute(sql).fetchall()
    con.execute("PRAGMA enable_profiling = 'json';")
    plans = [json.loads(con.execute(f'EXPLAIN ANALYZE {sql}').fetchall()[0][1]) for _ in range(G.TIMED)]
    con.execute('PRAGMA disable_profiling;')
    return plans


def execute(task):
    pattern, db = task
    import datetime
    import duckdb
    raw_path = os.path.join(PLACE, 'raw', f'{pattern}_{db}.json')
    if os.path.exists(raw_path):
        return f'[execute] {pattern} {db}: already done'
    man = json.load(open(os.path.join(G.OUT, 'manifest', f'{pattern}_{db}.json')))
    con = duckdb.connect(G.db_path(db), read_only=True)
    con.execute('PRAGMA threads=4')
    ns = {'math': math, 'numpy': numpy, 'db_conn': con, 'date': datetime.date, 'time': datetime.time,
          'VARCHAR': duckdb.typing.VARCHAR, 'INTEGER': duckdb.typing.INTEGER, 'DOUBLE': duckdb.typing.DOUBLE,
          'TIMESTAMP': duckdb.typing.TIMESTAMP, 'DATE': duckdb.typing.DATE, 'TIME': duckdb.typing.TIME,
          'BOOLEAN': duckdb.typing.BOOLEAN, 'BIGINT': duckdb.typing.BIGINT, 'FLOAT': duckdb.typing.FLOAT,
          'HUGEINT': duckdb.typing.HUGEINT}
    failed = set()
    for name, src in man['udf_code'].items():
        try:
            exec(''.join(src), ns)
        except Exception:
            failed.add(name)
    results, t0 = [], time.time()
    for e in man['entries']:
        pull, ok = pullup_sql(e)
        if not ok or e['udf_name'] in failed:
            continue
        rec = dict(udf_name=e['udf_name'], push_sql=e['sql'], pull_sql=pull, push=None, pull=None)
        for side, sql in (('push', e['sql']), ('pull', pull)):
            timer = threading.Timer(G.QUERY_TIMEOUT_S, con.interrupt)
            timer.start()
            try:
                rec[side] = _run(con, sql)
            except Exception as ex:
                try:
                    con.execute('PRAGMA disable_profiling;')
                except Exception:
                    pass
                rec[f'{side}_error'] = str(ex)[:200]
            finally:
                timer.cancel()
        results.append(rec)
    os.makedirs(os.path.join(PLACE, 'raw'), exist_ok=True)
    json.dump(results, open(raw_path, 'w'))
    ok = sum(r['push'] is not None and r['pull'] is not None for r in results)
    return f'[execute] {pattern} {db}: {ok}/{len(results)} pairs measured ({(time.time() - t0) / 60:.1f} min)'


def safe_execute(task):
    try:
        return execute(task)
    except Exception as ex:
        return f'[execute] {task}: FAILED {ex}'


def runtime_ms(plans):
    return 1000.0 * float(numpy.mean([G._elapsed_s(p) for p in plans]))


def parse(task):
    """Parse the pull-up side into a GRACEFUL workload (the push-down side exists in data/dyn_runs)."""
    pattern, db = task
    from cross_db_benchmark.benchmark_tools.dbms.parse_dd_plan import parse_dd_plans
    from cross_db_benchmark.benchmark_tools.dbms.parse_filter import PredicateNode
    raw = json.load(open(os.path.join(PLACE, 'raw', f'{pattern}_{db}.json')))
    man = json.load(open(os.path.join(G.OUT, 'manifest', f'{pattern}_{db}.json')))
    seed_wl = json.load(open(os.path.join(G.SRC, 'parsed_plans', db, 'workload.json')))
    query_list = []
    for r in raw:
        if r['pull'] is None or r['push'] is None:
            continue
        plans = []
        for pj in r['pull']:
            p = G._ns(pj)
            p.result = str(G._elapsed_s(pj))
            plans.append(p)
        query_list.append(SimpleNamespace(sql=r['pull_sql'], analyze_plans=plans, timeout=False))
    run_stats = SimpleNamespace(
        database_stats=SimpleNamespace(
            column_stats=[SimpleNamespace(**c) for c in seed_wl['database_stats']['column_stats']],
            table_stats=[SimpleNamespace(**t) for t in seed_wl['database_stats']['table_stats']]),
        query_list=query_list, run_kwargs={'hardware': 'lamp-dyn'}, total_time_secs=0.0)
    parsed, _ = parse_dd_plans(run_stats, min_runtime_ms=G.MIN_RT_MS, max_runtime_ms=G.MAX_RT_MS,
                               parse_baseline=False, parse_join_conds=True, udf_code_dict=man['udf_code'])

    def default(obj):
        if isinstance(obj, SimpleNamespace):
            return vars(obj)
        if isinstance(obj, PredicateNode):
            return obj.to_dict()
        return str(obj)

    root = pull_root(pattern)
    os.makedirs(os.path.join(root, 'parsed_plans', db), exist_ok=True)
    with open(os.path.join(root, 'parsed_plans', db, 'workload.json'), 'w') as f:
        json.dump(parsed, f, default=default)
    scripts = os.path.join(root, 'dbs', db, 'sql_scripts')
    os.makedirs(scripts, exist_ok=True)
    json.dump(man['udf_code'], open(os.path.join(scripts, 'udfs.json'), 'w'))
    shutil.copy(os.path.join(G.SRC, 'dbs', db, 'func_table_dict.csv'), os.path.join(root, 'dbs', db, 'func_table_dict.csv'))
    n = len(parsed['parsed_plans']) if isinstance(parsed, dict) else len(vars(parsed)['parsed_plans'])
    return f'[parse] {pattern} {db}: {n} pull-up plans'


def graphs(db):
    from udf_graph.create_graph import prepareGraphs
    from udf_graph.dbms_wrapper import DBMSWrapper
    duckdb_kwargs = {'dir': G.DB_DIR, 'version': '0.10.1'}
    msgs = []
    for pattern in G.PATTERNS:
        root = pull_root(pattern)
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
                      pullup_udf=True, udf_intermed_pos=False, skip_wj=True, skip_deepdb=True,
                      deepdb_rel_ensemble_location='', deepdb_single_ensemble_location='')
        msgs.append(f'{pattern}: {len(os.listdir(graph_dir)) - 1} graphs')
    return f'[graphs] {db}: ' + ', '.join(msgs)


def pairs():
    """Pairs measured back to back, both runtimes in [50 ms, 30 s], differing by > 5%, with plans on both sides."""
    out = {}
    for pattern in G.PATTERNS:
        for db in G.DBS:
            raw_path = os.path.join(PLACE, 'raw', f'{pattern}_{db}.json')
            push_wl_path = os.path.join(G.pattern_root(pattern), 'parsed_plans', db, 'workload.json')
            pull_wl_path = os.path.join(pull_root(pattern), 'parsed_plans', db, 'workload.json')
            if not all(os.path.exists(p) for p in (raw_path, push_wl_path, pull_wl_path)):
                continue
            graph_dir = os.path.join(pull_root(pattern), 'dbs', db, 'created_graphs')
            timed = {r['udf_name']: (runtime_ms(r['push']), runtime_ms(r['pull']))
                     for r in json.load(open(raw_path)) if r['push'] is not None and r['pull'] is not None}
            wls = {v: json.load(open(p)) for v, p in (('push', push_wl_path), ('pull', pull_wl_path))}
            plans = {v: {p['udf']['udf_name']: p for p in wl['parsed_plans']} for v, wl in wls.items()}
            names = sorted(n for n, (pu, pl) in timed.items()
                           if n in plans['push'] and n in plans['pull']
                           and os.path.exists(os.path.join(graph_dir, f'{n}.loopend.gpickle'))
                           and G.MIN_RT_MS <= min(pu, pl) and max(pu, pl) <= G.MAX_RT_MS
                           and max(pu, pl) / min(pu, pl) > MIN_DIFF)
            if not names:
                continue
            for v, path in (('push', push_wl_path), ('pull', pull_wl_path)):
                out_wl = dict(wls[v], parsed_plans=[plans[v][n] for n in names])
                json.dump(out_wl, open(os.path.join(os.path.dirname(path), 'workload_pairs.json'), 'w'))
            out.setdefault(pattern, {})[db] = {n: dict(push_ms=timed[n][0], pull_ms=timed[n][1]) for n in names}
            print(f'[pairs] {pattern} {db}: {len(names)} pairs, pull-up faster in '
                  f'{sum(timed[n][1] < timed[n][0] for n in names)}', flush=True)
    json.dump(out, open(os.path.join(PLACE, 'pairs.json'), 'w'), indent=1)
    print('total pairs', sum(len(u) for t in out.values() for u in t.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['execute', 'parse', 'graphs', 'pairs'])
    ap.add_argument('--dbs', nargs='+', default=G.DBS)
    ap.add_argument('--patterns', nargs='+', default=G.PATTERNS)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()
    tasks = [(p, db) for db in args.dbs for p in args.patterns]
    if args.phase == 'execute':
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
    else:
        pairs()


if __name__ == '__main__':
    main()
