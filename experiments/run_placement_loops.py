#!/usr/bin/env python3
"""
Placement (push-down vs pull-up) predictions for the loop-bearing pairs
(data_preparation/build_placement_pairs.py), using the trained loop-only LOO models:
fold <db>'s GRACEFUL and LAMP-W checkpoints predict both candidates of <db>'s pairs.

Inference only, on CPU (no GPU is visible to the jobs), at low priority.

  --card_type act : models trained on actual cardinalities   (loo_loops_act_seed0)
  --card_type est : models trained on DuckDB estimates       (loo_loops_est_seed0)

    python experiments/run_placement_loops.py --card_type act
    python experiments/analyze_placement.py --card_type act
"""
import argparse
import glob
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOP = os.path.join(REPO, 'data', 'loop_runs')
CONFIGS = {'graceful': 'ddactfonudfnolv_liboh_gradnorm_mldupl_loopend_loopedge',
           'lamp_w': 'ddactfonudfgraphlvw_liboh_gradnorm_mldupl_loopend_loopedge'}
TEST_DBS = ['accidents', 'airline', 'baseball', 'carcinogenesis', 'consumer', 'credit', 'employee', 'fhnk',
            'financial', 'geneea', 'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb', 'tournament',
            'tpc_h', 'walmart']


def pair_workloads(db):
    return [os.path.join(LOOP, f'duckdb_{v}', 'parsed_plans', db, 'workload_pairs.json') for v in ('pullup', 'pushdown')]


def checkpoint(card, config, db, seed=0):
    hits = glob.glob(os.path.join(REPO, 'data', 'results', f'loo_loops_{card}_seed{seed}', 'models', config, db, '**', '*.pt'),
                     recursive=True)
    assert len(hits) == 1, f'{card}/{config}/{db}: {hits}'
    return os.path.dirname(hits[0]), os.path.basename(hits[0])[:-3]


def done(out_dir, config, db, card):
    return len(glob.glob(os.path.join(out_dir, 'models', config, db, '**', f'*workload_pairs_*_{card}_per_query.csv'),
                         recursive=True)) >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', required=True, choices=['act', 'est', 'dd', 'wj'])
    ap.add_argument('--dbs', nargs='+', default=TEST_DBS)
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0, help='training seed of the checkpoints')
    args = ap.parse_args()
    out_dir = os.path.join(REPO, 'data', 'results',
                           f'placement_loops_{args.card_type}' + (f'_seed{args.seed}' if args.seed else ''))
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    stats_dir = os.path.join(REPO, 'data', 'fold_stats', f'loops_{args.card_type}')

    jobs = queue.Queue()
    for db in args.dbs:
        if not all(os.path.exists(p) for p in pair_workloads(db)):
            continue
        for config in CONFIGS:
            if not done(out_dir, config, db, args.card_type):
                jobs.put((config, db))
    total, lock, counter = jobs.qsize(), threading.Lock(), [0]
    print(f'[infer] {total} CPU inference jobs, {args.parallel} in parallel', flush=True)

    def worker():
        while True:
            try:
                config, db = jobs.get_nowait()
            except queue.Empty:
                return
            ckpt_dir, ckpt_name = checkpoint(args.card_type, config, db, args.seed)
            cmd = ['nice', '-n', '15', sys.executable, os.path.join(REPO, 'train.py'),
                   '--wl_base_path', LOOP, '--out_base_path', os.path.join(out_dir, 'models', config, db),
                   '--device', 'cpu', '--model_config', CONFIGS[config], '--data_keyword', 'complex_dd',
                   '--database', 'duckdb', '--test_against', db, '--include_pushdown_data',
                   '--card_type', args.card_type, '--min_runtime_ms', '50', '--max_runtime', '30',
                   '--num_workers', '2', '--skip_train', '--pretrained_model_artifact_dir', ckpt_dir,
                   '--pretrained_model_filename', ckpt_name, '--test_wl', ','.join(pair_workloads(db))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
                       LAMP_STATISTICS_FILE=os.path.join(stats_dir, f'{db}.json'))
            t0 = time.time()
            with open(os.path.join(out_dir, 'logs', f'{config}_{db}.log'), 'w') as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
            ok = rc == 0 and done(out_dir, config, db, args.card_type)
            with lock:
                counter[0] += 1
                print(f'[{counter[0]}/{total}] {config}/{db} {"ok" if ok else f"FAILED rc={rc}"} '
                      f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.parallel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'[done] {out_dir}', flush=True)


if __name__ == '__main__':
    main()
