# LAMP: Representing Loop Amplification for Learned UDF Cost Estimation

This repository contains the code, data pointers, and scripts to reproduce the results of the paper
*LAMP* (PVLDB submission). LAMP is a planning-time representation layer for learned cost estimation of
SQL queries with Python UDFs. It composes loop trip counts over procedural scopes, summarizes the
resulting work (including the operation mix) at the `INVOCATION` node of the joint SQL-UDF graph, and
resolves input-dependent trip counts at planning time with **Auto-Bound**. LAMP is implemented on top of
the GRACEFUL estimator, whose architecture and training procedure are left unchanged.

## Contents

```
lamp_dynamic/auto_bound/   Auto-Bound: trip-count resolution from the SQL call, catalog statistics, UDF source
lamp_dynamic/col_stats.py  column statistics (median, median value length) used by Auto-Bound
udf_graph/                 UDF graph construction and annotation (scope composition, work summaries, loop-scope edges)
models/                    GRACEFUL estimator (unchanged) and feature configurations
experiments/               training runners and analysis scripts for every result in the paper
data_preparation/          extraction of the loop workload, dynamic-loop benchmark, placement pairs
figures/                   scripts for the figures of the paper (output in figures/out/)
scripts/                   data download and end-to-end reproduction scripts
cross_db_benchmark/, deepdb/, card_est/   query execution, DeepDB and WanderJoin cardinality estimates (from GRACEFUL)
```

Configuration names used in code and result folders:

| Paper            | Code config            |
|------------------|------------------------|
| GRACEFUL         | `graceful`             |
| LAMP             | `lamp_w`               |
| LAMP-BR          | `lamp_w_bw` / `lamp_w_bwtrain` (environment `LAMP_BRANCH_WEIGHT=on`) |
| LV-PN            | `lv_pn`                |
| Ablations        | `abl_noscope`, `abl_noops`, `abl_nooplam`, `abl_noedge` |
| LAMP, fixed bound / measured bound | `lamp_w_noab` / `lamp_w_oracle` (environment `LAMP_AUTOBOUND=off` / `oracle`) |
| GRACEFUL+AB      | `graceful_ablocal` / `graceful_oraclelocal` |

Cardinality sources: `act` (actual), `est` (DuckDB), `dd` (DeepDB), `wj` (WanderJoin).

## 1. Environment

Python 3.9, CUDA 11.6 (for training). Tested on Ubuntu with NVIDIA RTX 3090 GPUs.

```bash
python3.9 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

All analysis scripts run on CPU. Training uses one GPU per job (`--gpus` selects device indices).

## 2. Data

**Released with this paper** (attached to the [v1.0 release](https://github.com/jalal-gilgiti/LAMP/releases/tag/v1.0) of this repository):

```bash
bash scripts/download_data.sh          # downloads and verifies the archives, extracts them into data/
```

| Archive | Size | Contents |
|---|---|---|
| `lamp_dynamic_benchmark.tar.zst` | 0.28 GB | `data/dyn_runs/`: the measured dynamic-loop benchmark (patterns T1-T8, 1,857 queries in [50 ms, 30 s]: rewritten UDFs, queries, measured runtimes, true trip counts, Auto-Bound estimates); `data/dyn_placement/`: the dynamic placement pairs; `data/placement/pairs.json`: the 453 static placement pairs |
| `lamp_fold_stats.tar.zst` | 0.11 GB | `data/fold_stats/`: feature normalization statistics of every leave-one-database-out fold |
| `lamp_predictions.tar.zst` | 0.06 GB | `data/results/`: per-query predictions of every model and run reported in the paper, and the loop class of every benchmark plan |

These archives suffice to reproduce every table and figure of the paper (Section 3).

**GRACEFUL benchmark** (needed only for retraining, re-running the dynamic benchmark, and the overhead
measurement): download the GRACEFUL data release from
[github.com/DataManagementLab/graceful](https://github.com/DataManagementLab/graceful), then

```bash
# loop-bearing subset used for training (data/loop_runs)
python data_preparation/extract_loop_subset.py --src /path/to/graceful_data/extracted/workload_runs --out data/loop_runs
# complete push-down workload for the full-workload models (data/full_runs)
python data_preparation/link_full_workload.py --workload_runs /path/to/graceful_data/extracted/workload_runs
# DuckDB databases of the eight dynamic-benchmark databases, as data/dyn_dbs/<db>_10_1.db
mkdir -p data/dyn_dbs && cp /path/to/graceful_data/<duckdb databases>/{baseball,credit,financial,geneea,movielens,ssb,tpc_h,walmart}_10_1.db data/dyn_dbs/
```

## 3. Reproducing the paper from the released predictions (CPU, minutes)

```bash
bash scripts/reproduce_paper.sh       # writes every table value to reproduce/*.txt and figures to figures/out/
```

The script runs the following commands; each line names the paper result it reproduces.

| Paper result | Command |
|---|---|
| Fig. 3, Table 4 (a), Table 5, Fig. 4 | `python experiments/analyze_loo_loops.py --card_type {act,est,dd,wj}` |
| Fig. 3, Table 4 (b): trained on actual, deployed with estimates | `python experiments/analyze_loo_loops.py --card_type act --eval_card est`; `--card_type act --eval_card {dd,wj} --results_dir data/results/loo_loops_act_seed0_ddwj` |
| Initialization robustness (three seeds) | `python experiments/analyze_seeds.py` |
| Training on the full workload | `python experiments/analyze_loo_loops.py --card_type act --results_dir data/results/full_act_seed0 --wl_base data/full_runs` |
| Fig. 5, Table 7 (LV-PN, ablations) | `python experiments/analyze_loo_loops.py --card_type act` |
| Table 6 (flat models) | `python experiments/flat_baselines_loops.py` |
| Fig. 6, Table 8, RQ3 text (full-workload models) | `python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act_full --calibrate_t1 [--eval_card est]` |
| RQ3, loop-only models (seeds 0, 1, 2) | `python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act{,_seed1,_seed2} --calibrate_t1` |
| RQ3, GRACEFUL+AB | `python experiments/analyze_local_bound.py --results_dir data/results/dyn_zs_act_full --card act` |
| Table 9, trained on actual (three seeds, seed average) | `python experiments/analyze_placement_ensemble.py --card_type act --eval_card {act,dd,wj,est}` |
| Table 9, trained on DeepDB / WanderJoin (seed 0) | `python experiments/analyze_placement.py --card_type {dd,wj}` |
| Table 9, trained on DuckDB (three seeds) | `python experiments/analyze_placement_ensemble.py --card_type est` |
| RQ4, placement with run-time loop bounds | `python experiments/analyze_dynamic_placement.py --models {loops,full} --root data/dyn_placement` |
| Planning overhead (needs `data/loop_runs`) | `python experiments/measure_overhead.py` |
| Figures 3-6 | `python figures/make_fig4_card_sources.py`, `make_fig_perdb_lines.py`, `make_fig_lvpn.py`, `make_fig_dynamic_patterns.py` |

## 4. Retraining from scratch (GPU)

Every model is trained per leave-one-database-out fold (19 held-out databases, 50 epochs, seed 0 unless noted).
The runners skip folds whose predictions already exist; use a new `--out_dir` to retrain.

```bash
# RQ1/RQ2: loop-only models, each cardinality source; ablations and LV-PN with actual cardinalities
python experiments/run_loo_loops.py --card_type act --configs graceful lamp_w lv_pn abl_noscope abl_noops abl_nooplam abl_noedge --gpus 0 1 2 3
python experiments/run_loo_loops.py --card_type {est,dd,wj} --configs graceful lamp_w --gpus 0 1 2 3
# seeds 1 and 2
python experiments/run_loo_loops.py --card_type {act,est} --configs graceful lamp_w --seed {1,2} --gpus 0 1 2 3
# trained on actual, evaluated with DeepDB / WanderJoin estimates
python experiments/run_loo_loops.py --card_type act --configs graceful lamp_w --pretrained_from data/results/loo_loops_act_seed0 \
       --out_dir data/results/loo_loops_act_seed0_ddwj --gpus 0 1
# full-workload models (five held-out databases), and LAMP-BR
python experiments/run_loo_loops.py --card_type act --configs graceful lamp_w --wl_base data/full_runs --stats_tag full \
       --out_dir data/results/full_act_seed0 --dbs baseball financial geneea movielens ssb --gpus 0 1
python experiments/run_loo_loops.py --card_type act --configs lamp_w_bw --wl_base data/full_runs --stats_tag full_bw \
       --out_dir data/results/full_act_seed0 --dbs baseball financial geneea movielens ssb --gpus 0
# RQ3: zero-shot prediction of the dynamic-loop benchmark with the trained models
python experiments/run_dynamic_zeroshot.py --card_type act --models full
python experiments/run_dynamic_zeroshot.py --card_type act --models loops --seed {0,1,2}
# RQ4: placement
python experiments/run_placement_loops.py --card_type {act,est} --seed {0,1,2}
python experiments/run_placement_loops.py --card_type {dd,wj}
python experiments/run_dynamic_placement.py --models {loops,full} --root data/dyn_placement
```

Rebuilding the data from the GRACEFUL benchmark and its databases (`data/dyn_dbs/`, see Section 2); Auto-Bound
reads column statistics directly from the databases:

```bash
python data_preparation/build_placement_pairs.py                         # static placement pairs
python data_preparation/gen_dynamic_loops.py --phase {generate,execute,parse,graphs}
python data_preparation/annotate_dynamic_loops.py                        # Auto-Bound annotation, true trip counts
python data_preparation/gen_placement_dynamic.py --phase {generate,execute,parse,graphs,annotate,pairs}
```

Execution times depend on the hardware; the released runtimes were measured with DuckDB 0.10.1 on two Intel Xeon
Gold 6226R CPUs.

## License

Apache License 2.0 (see `LICENSE`). This repository builds on the GRACEFUL code base (Apache 2.0); see `NOTICE`.
