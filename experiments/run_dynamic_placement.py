#!/usr/bin/env python3
"""
Placement on the dynamic-loop benchmark: the models of a fold (trained on static loops only) predict
both candidates of every push-down / pull-up pair of the held-out database
(data_preparation/gen_dynamic_placement.py; push-down side in data/dyn_runs, pull-up side in
data/dyn_place). Inference only, on CPU, at low priority.

    python experiments/run_dynamic_placement.py --models full|loops
    python experiments/analyze_dynamic_placement.py --models full|loops
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
sys.path.insert(0, os.path.join(REPO, 'experiments'))
from run_dynamic_zeroshot import CONFIGS, checkpoint  # noqa: E402
from run_loo_dynamic import DYN_DBS, PATTERNS, WL_BASE  # noqa: E402
from run_placement_loops import CONFIGS as MODEL_CONFIGS  # noqa: E402

PUSH = os.path.join(REPO, 'data', 'dyn_runs')
PULL = os.path.join(REPO, 'data', 'dyn_place')


def pair_workloads(db):
    out = []
    for t in PATTERNS:
        push = os.path.join(PUSH, t, 'duckdb_pushdown', 'parsed_plans', db, 'workload_pairs.json')
        pull = os.path.join(PULL, t, 'duckdb_pullup', 'parsed_plans', db, 'workload_pairs.json')
        if os.path.exists(push) and os.path.exists(pull):
            out += [push, pull]
    return out


def done(out_dir, config, db, card):
    return len(glob.glob(os.path.join(out_dir, 'models', config, db, '**', f'*workload_pairs_*_T?_{card}_per_query.csv'),
                         recursive=True)) >= len(pair_workloads(db))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='act', choices=['act', 'est'])
    ap.add_argument('--models', default='full', choices=['full', 'loops'])
    ap.add_argument('--configs', nargs='+', default=['graceful', 'lamp_w', 'lamp_w_noab'])
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--root', default=None, help='benchmark root holding dyn_runs/ and dyn_place/ (e.g. data/dyn_placement)')
    args = ap.parse_args()
    global PUSH, PULL
    tag = ''
    if args.root:
        PUSH, PULL = os.path.join(REPO, args.root, 'dyn_runs'), os.path.join(REPO, args.root, 'dyn_place')
        tag = '_' + os.path.basename(os.path.normpath(args.root))
    out_dir = os.path.join(REPO, 'data', 'results', f'dyn_place{tag}_{args.card_type}_{args.models}')
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    stats_dir = os.path.join(REPO, 'data', 'fold_stats', f'{args.models}_{args.card_type}')
    models_dir = os.path.join(REPO, 'data', 'results', f'loo_loops_{args.card_type}_seed0' if args.models == 'loops'
                              else f'full_{args.card_type}_seed0')
    dbs = [db for db in DYN_DBS if pair_workloads(db) and os.path.isdir(os.path.join(models_dir, 'models', 'lamp_w', db))]
    jobs = queue.Queue()
    for db in dbs:
        for config in args.configs:
            if not done(out_dir, config, db, args.card_type):
                jobs.put((config, db))
    total, lock, counter = jobs.qsize(), threading.Lock(), [0]
    print(f'[infer] {total} jobs over {dbs}', flush=True)

    def worker():
        while True:
            try:
                config, db = jobs.get_nowait()
            except queue.Empty:
                return
            ckpt_config, mode, extra_env = CONFIGS[config]
            ckpt_dir, ckpt_name = checkpoint(models_dir, ckpt_config, db)
            cmd = ['nice', '-n', '15', sys.executable, os.path.join(REPO, 'train.py'),
                   '--wl_base_path', WL_BASE, '--out_base_path', os.path.join(out_dir, 'models', config, db),
                   '--device', 'cpu', '--model_config', MODEL_CONFIGS[ckpt_config], '--data_keyword', 'complex_dd',
                   '--database', 'duckdb', '--test_against', db, '--include_pushdown_data',
                   '--card_type', args.card_type, '--min_runtime_ms', '50', '--max_runtime', '30',
                   '--num_workers', '2', '--skip_train', '--pretrained_model_artifact_dir', ckpt_dir,
                   '--pretrained_model_filename', ckpt_name, '--test_wl', ','.join(pair_workloads(db))]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', LAMP_AUTOBOUND=mode,
                       LAMP_STATISTICS_FILE=os.path.join(stats_dir, f'{db}.json'), **extra_env)
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
