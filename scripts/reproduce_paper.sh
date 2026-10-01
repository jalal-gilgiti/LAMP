#!/bin/bash
# Recompute every reported result from the released per-query predictions (CPU only).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
OUT=reproduce
mkdir -p $OUT

# RQ1: cost estimation across cardinality sources (Fig. 3, Tables 4 and 5, Fig. 4)
for c in act est dd wj; do
    python experiments/analyze_loo_loops.py --card_type $c > $OUT/rq1_same_source_$c.txt
done
python experiments/analyze_loo_loops.py --card_type act --eval_card est > $OUT/rq1_trained_act_eval_est.txt
for c in dd wj; do
    python experiments/analyze_loo_loops.py --card_type act --eval_card $c \
        --results_dir data/results/loo_loops_act_seed0_ddwj > $OUT/rq1_trained_act_eval_$c.txt
done
python experiments/analyze_seeds.py > $OUT/rq1_seeds.txt
python experiments/analyze_loo_loops.py --card_type act --results_dir data/results/full_act_seed0 \
    --wl_base data/full_runs > $OUT/rq1_full_workload.txt

# RQ2: representation design (Fig. 5, Tables 6 and 7; ablations are part of rq1_same_source_act.txt)
python experiments/flat_baselines_loops.py > $OUT/rq2_flat_models.txt

# RQ3: input-dependent loop work (Fig. 6, Table 8)
python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act_full --calibrate_t1 > $OUT/rq3_full_models_act.txt
python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act_full --calibrate_t1 --eval_card est > $OUT/rq3_full_models_est.txt
for s in "" _seed1 _seed2; do
    python experiments/analyze_loo_dynamic.py --results_dir data/results/dyn_zs_act$s --calibrate_t1 > $OUT/rq3_loop_models$s.txt
done
python experiments/analyze_local_bound.py --results_dir data/results/dyn_zs_act_full --card act > $OUT/rq3_local_bound.txt

# RQ4: placement (Table 9)
for c in act dd wj est; do
    python experiments/analyze_placement_ensemble.py --card_type act --eval_card $c > $OUT/rq4_trained_act_eval_$c.txt
done
for c in dd wj; do
    python experiments/analyze_placement.py --card_type $c > $OUT/rq4_trained_$c.txt
done
python experiments/analyze_placement_ensemble.py --card_type est > $OUT/rq4_trained_est.txt
for m in loops full; do
    python experiments/analyze_dynamic_placement.py --models $m --root data/dyn_placement > $OUT/rq4_dynamic_$m.txt
done

# planning overhead (needs the GRACEFUL loop workload in data/loop_runs, see README)
if [ -d data/loop_runs/duckdb_pushdown/dbs ]; then
    python experiments/measure_overhead.py > $OUT/overhead.txt
fi

# figures
python figures/make_fig4_card_sources.py
python figures/make_fig_perdb_lines.py
python figures/make_fig_lvpn.py
python figures/make_fig_dynamic_patterns.py
echo "results in $OUT/, figures in figures/out/"
