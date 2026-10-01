#!/usr/bin/env python3
"""
Extract the loop-bearing subset of the GRACEFUL workload runs into this project.

For every workload variant and database, keeps only plans whose runtime lies in
[MIN_RT_MS, MAX_RT_MS] and whose UDF contains at least one explicit for/while
loop, then copies the UDF graphs those plans reference. The output mirrors the
layout expected by utils/hyperparams_utils.compile_train_test_filenames, so it can
be passed directly as --wl_base_path:

    <out>/<variant>/parsed_plans/<db>/workload.json      (loop plans only)
    <out>/<variant>/dbs/<db>/created_graphs/<udf>.*       (graphs of those plans)
    <out>/<variant>/dbs/<db>/sql_scripts/                 (UDF sources)
    <out>/manifest.json                                   (counts + selection rule)

Loop count: plan.udf.udf_num_loops when present, otherwise the number of source
lines of the UDF that start with `for` / `while` (udf_num_loops is absent from
duckdb_pushdown / duckdb_pullup metadata).

The source tree is only read, never written.

Usage:
    python3 data_preparation/extract_loop_subset.py \
        --src <GRACEFUL_DATA>/workload_runs \
        --out data/loop_runs
"""
import argparse
import json
import os
import re
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

MIN_RT_MS = 50
MAX_RT_MS = 30000
VARIANTS = ['duckdb_pushdown', 'duckdb_intermed_pullup', 'duckdb_pullup']
LOOP_LINE = re.compile(r'\s*(for|while)\b')


def count_loops(plan, udf_sources):
    udf = plan.get('udf', {})
    n = udf.get('udf_num_loops')
    if n is not None:
        return n, 'meta'
    code = udf_sources.get(udf.get('udf_name'), '')
    if isinstance(code, list):
        code = '\n'.join(code)
    return sum(1 for ln in code.split('\n') if LOOP_LINE.match(ln)), 'src'


def copy_graphs(src_db, dst_db, udf_names):
    src_dir = os.path.join(src_db, 'created_graphs')
    dst_dir = os.path.join(dst_db, 'created_graphs')
    os.makedirs(dst_dir, exist_ok=True)
    wanted = set(udf_names)

    # graphs extracted on disk
    on_disk = []
    if os.path.isdir(src_dir):
        for fn in os.listdir(src_dir):
            if fn.split('.', 1)[0] in wanted:
                on_disk.append(fn)
    found = {fn.split('.', 1)[0] for fn in on_disk}

    def _cp(fn):
        dst = os.path.join(dst_dir, fn)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(src_dir, fn), dst)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(_cp, on_disk))

    # graphs only available inside created_graphs.zip
    from_zip = 0
    missing = wanted - found
    zpath = os.path.join(src_db, 'created_graphs.zip')
    if missing and os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                base = os.path.basename(member)
                if base.split('.', 1)[0] in missing and not member.endswith('/'):
                    dst = os.path.join(dst_dir, base)
                    if not os.path.exists(dst):
                        with z.open(member) as fsrc, open(dst, 'wb') as fdst:
                            shutil.copyfileobj(fsrc, fdst)
                    from_zip += 1
                    found.add(base.split('.', 1)[0])

    return len(on_disk), from_zip, sorted(wanted - found)


def process_db(src_variant, dst_variant, db):
    wl_src = os.path.join(src_variant, 'parsed_plans', db, 'workload.json')
    with open(wl_src) as f:
        wl = json.load(f)

    src_db = os.path.join(src_variant, 'dbs', db)
    udf_src_path = os.path.join(src_db, 'sql_scripts', 'udfs.json')
    udf_sources = {}
    if os.path.exists(udf_src_path):
        with open(udf_src_path) as f:
            udf_sources = json.load(f)

    kept, n_single, n_multi, how = [], 0, 0, set()
    n_in_range = 0
    for plan in wl['parsed_plans']:
        rt = plan.get('plan_runtime_ms', plan.get('plan_runtime', 0))
        if not (MIN_RT_MS <= rt <= MAX_RT_MS):
            continue
        n_in_range += 1
        n_loops, src = count_loops(plan, udf_sources)
        how.add(src)
        if n_loops >= 1:
            kept.append(plan)
            n_single += n_loops == 1
            n_multi += n_loops >= 2

    stats = dict(plans_total=len(wl['parsed_plans']), plans_in_range=n_in_range,
                 loop_plans=len(kept), single_loop=n_single, multi_loop=n_multi,
                 loop_count_source=sorted(how))
    if not kept:
        return stats

    out_wl = dict(wl)
    out_wl['parsed_plans'] = kept
    dst_wl_dir = os.path.join(dst_variant, 'parsed_plans', db)
    os.makedirs(dst_wl_dir, exist_ok=True)
    with open(os.path.join(dst_wl_dir, 'workload.json'), 'w') as f:
        json.dump(out_wl, f)

    dst_db = os.path.join(dst_variant, 'dbs', db)
    os.makedirs(dst_db, exist_ok=True)
    if os.path.isdir(os.path.join(src_db, 'sql_scripts')):
        shutil.copytree(os.path.join(src_db, 'sql_scripts'), os.path.join(dst_db, 'sql_scripts'),
                        dirs_exist_ok=True)
    for fn in ('func_table_dict.csv',):
        if os.path.exists(os.path.join(src_db, fn)):
            shutil.copy2(os.path.join(src_db, fn), os.path.join(dst_db, fn))

    udf_names = [p['udf']['udf_name'] for p in kept]
    n_disk, n_zip, missing = copy_graphs(src_db, dst_db, udf_names)
    stats.update(graph_files_from_disk=n_disk, graph_files_from_zip=n_zip,
                 udfs_without_graph=missing)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variants', nargs='+', default=VARIANTS)
    args = ap.parse_args()

    manifest = dict(source=args.src, runtime_range_ms=[MIN_RT_MS, MAX_RT_MS],
                    selection='udf_num_loops >= 1 (metadata) else count of source lines matching '
                              r'^\s*(for|while)\b >= 1',
                    variants={})
    for variant in args.variants:
        src_variant = os.path.join(args.src, variant)
        dst_variant = os.path.join(args.out, variant)
        dbs = sorted(d for d in os.listdir(os.path.join(src_variant, 'parsed_plans'))
                     if os.path.isfile(os.path.join(src_variant, 'parsed_plans', d, 'workload.json')))
        manifest['variants'][variant] = {}
        for db in dbs:
            stats = process_db(src_variant, dst_variant, db)
            manifest['variants'][variant][db] = stats
            print(f'{variant:<24}{db:<16}loops={stats["loop_plans"]:>5}  '
                  f'disk={stats.get("graph_files_from_disk", 0):>5}  zip={stats.get("graph_files_from_zip", 0):>5}  '
                  f'missing={len(stats.get("udfs_without_graph", []))}', flush=True)
        # plan-level feature statistics of the full variant (kept for reference;
        # fold-local statistics must be rebuilt from the training DBs)
        for fn in ('statistics_workload_combined.json', 'udf_stats.json'):
            p = os.path.join(src_variant, 'parsed_plans', fn)
            if os.path.exists(p):
                os.makedirs(os.path.join(dst_variant, 'parsed_plans'), exist_ok=True)
                shutil.copy2(p, os.path.join(dst_variant, 'parsed_plans', f'full_corpus_{fn}'))

    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print('manifest written', flush=True)


if __name__ == '__main__':
    main()
