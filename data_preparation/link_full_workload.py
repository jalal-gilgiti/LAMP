#!/usr/bin/env python3
"""
Link the complete GRACEFUL push-down workload (queries with and without loops) into data/full_runs,
the workload root used for the full-workload models (run_loo_loops.py --wl_base data/full_runs).

    python data_preparation/link_full_workload.py --workload_runs /path/to/graceful_data/extracted/workload_runs

Creates
    data/full_runs/duckdb_pushdown/parsed_plans/<db>/workload.json
    data/full_runs/duckdb_pushdown/dbs/<db>
as symbolic links into <workload_runs>/duckdb_pushdown.
"""
import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workload_runs', required=True, help='GRACEFUL workload_runs directory')
    ap.add_argument('--out', default=os.path.join(REPO, 'data', 'full_runs'))
    args = ap.parse_args()

    src = os.path.join(os.path.abspath(args.workload_runs), 'duckdb_pushdown')
    dst = os.path.join(args.out, 'duckdb_pushdown')
    dbs = sorted(d for d in os.listdir(os.path.join(src, 'parsed_plans'))
                 if os.path.exists(os.path.join(src, 'parsed_plans', d, 'workload.json')))
    for db in dbs:
        link(os.path.join(src, 'parsed_plans', db, 'workload.json'),
             os.path.join(dst, 'parsed_plans', db, 'workload.json'))
        link(os.path.join(src, 'dbs', db), os.path.join(dst, 'dbs', db))
    print(f'linked {len(dbs)} databases into {dst}')


if __name__ == '__main__':
    main()
