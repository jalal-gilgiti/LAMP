#!/usr/bin/env python3
"""
Leave-one-database-out on loop-bearing plans only (full GRACEFUL corpus).

Each fold trains on the loop plans of every other database and tests on all loop plans
of the held-out database. Per fold:
  * feature statistics are fitted on the training databases only
    (experiments/build_fold_stats.py -> LAMP_STATISTICS_FILE),
  * no runtime re-balancing, so every loop plan of the held-out DB is evaluated,
  * results are read from the per-card per-query CSVs written by train.py
    (..._pushdown_<card>_per_query.csv), never from positional log parsing.

Data: data/loop_runs/duckdb_pushdown (data_preparation/extract_loop_subset.py).
basketball (14 loop plans) is used for training only, not as a test fold.

Usage:
    python experiments/run_loo_loops.py --card_type est --gpus 0 1 2 3 4 5 6 7
    python experiments/analyze_loo_loops.py --card_type est
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

TEST_DBS = [
    'accidents', 'airline', 'baseball', 'carcinogenesis', 'consumer', 'credit', 'employee', 'fhnk',
    'financial', 'geneea', 'genome', 'hepatitis', 'imdb', 'movielens', 'seznam', 'ssb', 'tournament',
    'tpc_h', 'walmart',
]

CONFIGS = {
    'graceful': 'ddactfonudfnolv_liboh_gradnorm_mldupl_loopend_loopedge',
    'lamp_w': 'ddactfonudfgraphlvw_liboh_gradnorm_mldupl_loopend_loopedge',
    'lamp_w_bw': 'ddactfonudfgraphlvw_liboh_gradnorm_mldupl_loopend_loopedge',
    # local per-node encoding: log Lambda(v) attached to every UDF node (LV-PN)
    'lv_pn': 'ddactfonudf_liboh_gradnorm_mldupl_loopend_loopedge',
    # ablations of LAMP-W, one component removed each
    'abl_noedge': 'ddactfonudfgraphlvwnoedge_liboh_gradnorm_mldupl_loopend_loopedge',
    'abl_noops': 'ddactfonudfgraphlvwnoops_liboh_gradnorm_mldupl_loopend_loopedge',
    'abl_nooplam': 'ddactfonudfgraphlvwnooplam_liboh_gradnorm_mldupl_loopend_loopedge',
    'abl_noscope': 'ddactfonudfgraphlvwnoscope_liboh_gradnorm_mldupl_loopend_loopedge',
}
# extra environment per config, applied to its fold statistics and its training job
# (udf_graph/annotate_graph_info.py: LAMP_BRANCH_WEIGHT weights loop work by the rows reaching each node)
CONFIG_ENV = {'lamp_w_bw': {'LAMP_BRANCH_WEIGHT': 'on'}}


def per_query_csv(out_dir, config, db, card):
    hits = glob.glob(os.path.join(out_dir, 'models', config, db, '**', f'*_pushdown_{card}_per_query.csv'),
                     recursive=True)
    return hits[0] if hits else None


def build_stats(db, card_type, stats_dir, wl_base=WL_BASE, extra_env=None):
    out = os.path.join(stats_dir, f'{db}.json')
    if os.path.exists(out):
        return out
    log = open(os.path.join(stats_dir, f'{db}.log'), 'w')
    subprocess.run([sys.executable, os.path.join(REPO, 'experiments', 'build_fold_stats.py'),
                    '--exclude', db, '--card_type', card_type, '--wl_base_path', wl_base, '--out', out],
                   stdout=log, stderr=subprocess.STDOUT, check=True, cwd=REPO, env=dict(os.environ, **(extra_env or {})))
    return out


def pretrained_checkpoint(args, config, db):
    """Checkpoint of a finished run under --pretrained_from, as (artifact dir, file name without .pt)."""
    hits = glob.glob(os.path.join(args.pretrained_from, 'models', config, db, '**', '*.pt'), recursive=True)
    assert len(hits) == 1, f'expected one checkpoint for {config}/{db}, found {hits}'
    return os.path.dirname(hits[0]), os.path.basename(hits[0])[:-len('.pt')]


def train_cmd(args, config, db, gpu_out):
    extra = []
    if args.pretrained_from:
        ckpt_dir, ckpt_name = pretrained_checkpoint(args, config, db)
        extra = ['--skip_train', '--pretrained_model_artifact_dir', ckpt_dir, '--pretrained_model_filename', ckpt_name]
    return [sys.executable, os.path.join(REPO, 'train.py')] + extra + [
            '--wl_base_path', args.wl_base,
            '--out_base_path', gpu_out,
            '--device', 'cuda:0',
            '--model_config', CONFIGS[config],
            '--data_keyword', 'complex_dd',
            '--database', 'duckdb',
            '--test_against', db,
            '--include_pushdown_data',
            '--card_type', args.card_type,
            '--min_runtime_ms', '50',
            '--max_runtime', '30',
            '--epochs', str(args.epochs),
            '--seed', str(args.seed),
            '--num_workers', str(args.num_workers)]


def gpu_uuid(index):
    """UUID of the GPU at nvidia-smi index `index` (device indices are not stable across machine restarts)."""
    out = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=uuid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out.startswith('GPU-'), out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card_type', default='est', choices=['est', 'act', 'dd', 'wj'])
    ap.add_argument('--configs', nargs='+', default=list(CONFIGS))
    ap.add_argument('--dbs', nargs='+', default=TEST_DBS)
    ap.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    ap.add_argument('--jobs_per_gpu', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--wl_base', default=WL_BASE,
                    help='workload root holding duckdb_pushdown/ (default: loop-only subset; data/full_runs = full corpus)')
    ap.add_argument('--stats_tag', default='loops', help='fold statistics go to data/fold_stats/<tag>_<card>')
    ap.add_argument('--pretrained_from', default=None,
                    help='results dir of a finished run: evaluate its checkpoints (no training), e.g. to score '
                         'act-trained models under DeepDB / WanderJoin estimates')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(REPO, 'data', 'results', f'loo_loops_{args.card_type}_seed{args.seed}')
    stats_dir = os.path.join(REPO, 'data', 'fold_stats', f'{args.stats_tag}_{args.card_type}')
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)

    # fold statistics are shared by all configs of a run, so they must agree on the feature environment
    envs = {tuple(sorted(CONFIG_ENV.get(c, {}).items())) for c in args.configs}
    assert len(envs) == 1, f'configs with different feature environments need separate runs: {args.configs}'
    extra_env = dict(envs.pop())
    print(f'[stats] building fold statistics for {len(args.dbs)} folds -> {stats_dir} {extra_env}', flush=True)
    with ThreadPoolExecutor(4) as ex:
        stats = dict(zip(args.dbs, ex.map(lambda db: build_stats(db, args.card_type, stats_dir, args.wl_base, extra_env),
                                          args.dbs)))

    jobs = queue.Queue()
    for config in args.configs:
        for db in args.dbs:
            if per_query_csv(out_dir, config, db, args.card_type):
                print(f'[skip] {config}/{db} already done', flush=True)
            else:
                jobs.put((config, db))
    total = jobs.qsize()
    print(f'[train] {total} jobs on GPUs {args.gpus} x {args.jobs_per_gpu}', flush=True)

    lock = threading.Lock()
    done = [0]

    claim_dir = os.path.join(out_dir, 'claims')
    os.makedirs(claim_dir, exist_ok=True)

    def claim(config, db):
        """Atomically claim a job so that several runners can share one output directory; a claim whose
        runner is no longer alive is taken over (train.py resumes from its checkpoint)."""
        path = os.path.join(claim_dir, f'{config}_{db}')
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return path
            except FileExistsError:
                boot = time.time() - float(open('/proc/uptime').read().split()[0])
                try:
                    if os.path.getmtime(path) > boot:  # claims from before a reboot are always stale
                        os.kill(int(open(path).read().strip() or 0), 0)
                        return None  # held by a live runner
                except (ProcessLookupError, ValueError):
                    pass
                with open(path, 'w') as f:  # take over the stale claim
                    f.write(str(os.getpid()))
                return path
        return None

    def worker(gpu):
        while True:
            try:
                config, db = jobs.get_nowait()
            except queue.Empty:
                return
            if per_query_csv(out_dir, config, db, args.card_type):
                continue  # finished by another runner
            claim_path = claim(config, db)
            if claim_path is None:
                print(f'[skip] {config}/{db} claimed by another runner', flush=True)
                continue
            model_out = os.path.join(out_dir, 'models', config, db)
            log_path = os.path.join(out_dir, 'logs', f'{config}_{db}.log')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid(gpu), LAMP_STATISTICS_FILE=stats[db],
                       WANDB_MODE='disabled', **CONFIG_ENV.get(config, {}))
            t0 = time.time()
            with open(log_path, 'w') as log:
                rc = subprocess.run(train_cmd(args, config, db, model_out), stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=REPO).returncode
            ok = rc == 0 and per_query_csv(out_dir, config, db, args.card_type)
            os.remove(claim_path)
            with lock:
                done[0] += 1
                print(f'[{done[0]}/{total}] gpu{gpu} {config}/{db} {"ok" if ok else f"FAILED rc={rc}"} '
                      f'({(time.time() - t0) / 60:.1f} min) log={log_path}', flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus for _ in range(args.jobs_per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f'[done] results in {out_dir}; run experiments/analyze_loo_loops.py --card_type {args.card_type}',
          flush=True)


if __name__ == '__main__':
    main()
