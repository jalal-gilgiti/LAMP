#!/usr/bin/env python3
"""
Dynamic loops with the loop-only LOO models (no retraining): fold <db>'s GRACEFUL and LAMP-W
checkpoints from loo_loops_<card>_seed0 (trained on the static-loop corpus of the other databases)
predict the measured dynamic-loop workload of <db> (data/dyn_runs, T1-T8).

This is the deployment question behind Auto-Bound: a model that learned loop costs from static
loops meets loops whose bound is only known at run time. GRACEFUL sees no_iter = -1; LAMP-W gets
Lambda from Auto-Bound (on), the fixed fallback bound (off) or the true mean trip count (oracle).

The dynamic labels were measured on a different machine than the training corpus, so the analysis
reports both raw Q-errors and Q-errors after one global scale factor per model (fitted on T1, the
static control pattern; see analyze_loo_dynamic.py --calibrate_t1).

Inference only, on CPU (no GPU is visible to the jobs), at low priority.

    python experiments/run_dynamic_zeroshot.py --card_type act [--models full]
    python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act --calibrate_t1
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
from run_loo_dynamic import DYN_DBS, WL_BASE, dyn_workloads, done  # noqa: E402
from run_placement_loops import CONFIGS as MODEL_CONFIGS  # noqa: E402
from run_loo_loops import CONFIGS as LOO_CONFIGS  # noqa: E402

CONFIGS = {  # name -> (checkpoint config, LAMP_AUTOBOUND mode, extra environment)
    'graceful': ('graceful', 'on', {}),
    'lamp_w': ('lamp_w', 'on', {}),
    'lamp_w_noab': ('lamp_w', 'off', {}),
    'lamp_w_oracle': ('lamp_w', 'oracle', {}),
    # LAMP-BR: checkpoints trained with branch-reach weighting (run_loo_loops.py lamp_w_bw, full-workload models)
    'lamp_w_bwtrain': ('lamp_w_bw', 'on', {'LAMP_BRANCH_WEIGHT': 'on'}),
    # GRACEFUL + Auto-Bound (local): the GRACEFUL checkpoint, with the Auto-Bound trip count written into its
    # own local no_iter feature of each dynamic loop (dd_plan_batching.ab_local_bounds); no LAMP features
    'graceful_ablocal': ('graceful', 'on', {'LAMP_AB_LOCAL': 'on'}),
    'graceful_oraclelocal': ('graceful', 'on', {'LAMP_AB_LOCAL': 'oracle'}),
}
STATS_TAG = {'lamp_w_bwtrain': 'full_bw'}  # feature statistics the retrained checkpoints were normalised with
DEFAULT_CONFIGS = ['graceful', 'lamp_w', 'lamp_w_noab', 'lamp_w_oracle']


def checkpoint(models_dir, config, db):
    hits = glob.glob(os.path.join(models_dir, 'models', config, db, '**', '*.pt'), recursive=True)
    assert len(hits) == 1, f'{models_dir}/{config}/{db}: {hits}'
    return os.path.dirname(hits[0]), os.path.basename(hits[0])[:-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='act', choices=['act', 'est'])
    ap.add_argument('--dbs', nargs='+', default=DYN_DBS)
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS, choices=list(CONFIGS))
    ap.add_argument('--models', default='loops', choices=['loops', 'full'],
                    help='loops: loo_loops_<card>_seed0 (19 folds); full: full_<card>_seed0 (full-corpus training)')
    ap.add_argument('--seed', type=int, default=0, help='training seed of the checkpoints (loop-only models)')
    args = ap.parse_args()
    suffix = ('' if args.models == 'loops' else '_full') + (f'_seed{args.seed}' if args.seed else '')
    out_dir = os.path.join(REPO, 'data', 'results', f'dyn_zs_{args.card_type}{suffix}')
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    stats_dir = os.path.join(REPO, 'data', 'fold_stats', f'{args.models}_{args.card_type}')
    models_dir = os.path.join(REPO, 'data', 'results',
                              f'loo_loops_{args.card_type}_seed{args.seed}' if args.models == 'loops'
                              else f'full_{args.card_type}_seed{args.seed}')
    args.dbs = [db for db in args.dbs if os.path.isdir(os.path.join(models_dir, 'models', 'lamp_w', db))]

    jobs = queue.Queue()
    for db in args.dbs:
        for config in args.configs:
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
            ckpt_config, mode, extra_env = CONFIGS[config]
            ckpt_dir, ckpt_name = checkpoint(models_dir, ckpt_config, db)
            cmd = ['nice', '-n', '15', sys.executable, os.path.join(REPO, 'train.py'),
                   '--wl_base_path', WL_BASE, '--out_base_path', os.path.join(out_dir, 'models', config, db),
                   '--device', 'cpu', '--model_config', MODEL_CONFIGS.get(ckpt_config, LOO_CONFIGS.get(ckpt_config)), '--data_keyword', 'complex_dd',
                   '--database', 'duckdb', '--test_against', db, '--include_pushdown_data',
                   '--card_type', args.card_type, '--min_runtime_ms', '50', '--max_runtime', '30',
                   '--num_workers', '2', '--skip_train', '--pretrained_model_artifact_dir', ckpt_dir,
                   '--pretrained_model_filename', ckpt_name, '--test_wl', ','.join(dyn_workloads([db]))]
            cfg_stats = (os.path.join(REPO, 'data', 'fold_stats', f'{STATS_TAG[config]}_{args.card_type}')
                         if config in STATS_TAG else stats_dir)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled', LAMP_AUTOBOUND=mode,
                       LAMP_STATISTICS_FILE=os.path.join(cfg_stats, f'{db}.json'), **extra_env)
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
