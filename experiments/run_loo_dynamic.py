#!/usr/bin/env python3
"""
Leave-one-database-out on the measured dynamic-loop workload (data/dyn_runs, T1-T8).

Fold <db> (one of the 8 dynamic databases):
  train  dynamic-loop plans (all patterns, incl. T1 static loops) of the other dynamic databases;
         every label is measured on the same machine (the original benchmark labels come from
         different hardware and are therefore not mixed in)
  test   dynamic-loop plans of <db>, one workload per pattern

Models (same GNN, loss and training; LAMP_AUTOBOUND selects the trip-count source):
  graceful       GRACEFUL baseline (dynamic loops appear with no_iter = -1)
  lamp_w         LAMP-W with Auto-Bound estimates                       (deployment)
  lamp_w_noab    LAMP-W with the fixed fallback bound instead of Auto-Bound (ablation)
  lamp_w_oracle  LAMP-W with the true mean trip count                     (upper bound)

Fold statistics are fitted on the training plans only, per trip-count mode.

    python experiments/run_loo_dynamic.py --card_type act --gpus 6 7 --jobs_per_gpu 3
    python experiments/analyze_loo_dynamic.py --card_type act
"""
import argparse
import glob
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL_BASE = os.path.join(REPO, 'data', 'loop_runs')
DYN = os.path.join(REPO, 'data', 'dyn_runs')
DYN_DBS = ['financial', 'geneea', 'baseball', 'movielens', 'ssb', 'credit', 'tpc_h', 'walmart']
PATTERNS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
LVW = 'ddactfonudfgraphlvw_liboh_gradnorm_mldupl_loopend_loopedge'
CONFIGS = {  # name -> (model config, LAMP_AUTOBOUND mode)
    'graceful': ('ddactfonudfnolv_liboh_gradnorm_mldupl_loopend_loopedge', 'on'),
    'lamp_w': (LVW, 'on'),
    'lamp_w_noab': (LVW, 'off'),
    'lamp_w_oracle': (LVW, 'oracle'),
}


def dyn_workloads(dbs):
    return [p for db in dbs for t in PATTERNS
            for p in [os.path.join(DYN, t, 'duckdb_pushdown', 'parsed_plans', db, 'workload.json')]
            if os.path.exists(p)]


def done(out_dir, config, db, card):
    return len(glob.glob(os.path.join(out_dir, 'models', config, db, '**', f'*_T?_{card}_per_query.csv'),
                         recursive=True)) >= len(dyn_workloads([db]))


def build_stats(db, card, mode):
    stats_dir = os.path.join(REPO, 'data', 'fold_stats', f'dyn_{card}_{mode}')
    os.makedirs(stats_dir, exist_ok=True)
    out = os.path.join(stats_dir, f'{db}.json')
    if not os.path.exists(out):
        extra = ','.join(dyn_workloads([d for d in DYN_DBS if d != db]))
        with open(os.path.join(stats_dir, f'{db}.log'), 'w') as log:
            subprocess.run([sys.executable, os.path.join(REPO, 'experiments', 'build_fold_stats.py'),
                            '--exclude', db, '--card_type', card, '--wl_base_path', WL_BASE,
                            '--extra_wl', extra, '--only_extra', '--out', out],
                           stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO,
                           env=dict(os.environ, LAMP_AUTOBOUND=mode))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='act', choices=['act', 'est'])
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--dbs', nargs='+', default=DYN_DBS)
    ap.add_argument('--gpus', nargs='+', type=int, required=True)
    ap.add_argument('--jobs_per_gpu', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(REPO, 'data', 'results', f'dyn_{args.card_type}_seed{args.seed}')
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)

    modes = sorted({CONFIGS[c][1] for c in args.configs})
    print(f'[stats] fold statistics for {len(args.dbs)} folds x modes {modes}', flush=True)
    with ThreadPoolExecutor(4) as ex:
        keys = [(db, m) for db in args.dbs for m in modes]
        stats = dict(zip(keys, ex.map(lambda k: build_stats(k[0], args.card_type, k[1]), keys)))

    jobs = queue.Queue()
    for db in args.dbs:
        for config in args.configs:
            if done(out_dir, config, db, args.card_type):
                print(f'[skip] {config}/{db} already done', flush=True)
            else:
                jobs.put((config, db))
    total = jobs.qsize()
    print(f'[train] {total} jobs on GPUs {args.gpus} x {args.jobs_per_gpu}', flush=True)
    lock, counter = threading.Lock(), [0]

    def worker(gpu):
        while True:
            try:
                config, db = jobs.get_nowait()
            except queue.Empty:
                return
            model_config, mode = CONFIGS[config]
            cmd = [sys.executable, os.path.join(REPO, 'train.py'),
                   '--wl_base_path', WL_BASE, '--out_base_path', os.path.join(out_dir, 'models', config, db),
                   '--device', 'cuda:0', '--model_config', model_config, '--data_keyword', 'complex_dd',
                   '--database', 'duckdb', '--test_against', db, '--include_pushdown_data',
                   '--card_type', args.card_type, '--min_runtime_ms', '50', '--max_runtime', '30',
                   '--epochs', str(args.epochs), '--seed', str(args.seed), '--num_workers', '4',
                   '--train_wl', ','.join(dyn_workloads([d for d in DYN_DBS if d != db])),
                   '--test_wl', ','.join(dyn_workloads([db]))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), LAMP_STATISTICS_FILE=stats[(db, mode)],
                       LAMP_AUTOBOUND=mode, WANDB_MODE='disabled')
            log_path = os.path.join(out_dir, 'logs', f'{config}_{db}.log')
            t0 = time.time()
            with open(log_path, 'w') as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
            ok = rc == 0 and done(out_dir, config, db, args.card_type)
            with lock:
                counter[0] += 1
                print(f'[{counter[0]}/{total}] gpu{gpu} {config}/{db} {"ok" if ok else f"FAILED rc={rc}"} '
                      f'({(time.time() - t0) / 60:.1f} min) log={log_path}', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'[done] results in {out_dir}', flush=True)


if __name__ == '__main__':
    main()
