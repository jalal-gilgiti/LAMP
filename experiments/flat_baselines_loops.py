#!/usr/bin/env python3
"""
Flat-model controls on the loop-bearing workload (RQ2: is aggregation alone sufficient?).

Same queries, labels and leave-one-database-out folds as the GNN results
(data/results/loo_loops_act_seed0/per_query_act.csv, 19 held-out databases): for fold <db>
a flat model is fitted on the queries of the other 18 databases and tested on <db>.
Target: log runtime. Features per query, from its UDF graph (actual cardinalities):

  log-linear  rows only              log in_rows
              rows + max log Lambda       + max_log_lambda_s
              rows + scope shape     + max / frac / mean log Lambda, log_loop_work
              total work only        log_total_work
  XGBoost     structural             log in_rows, #loops, #branches, operation counts per type
              structural + LAMP-W    + the 14 LAMP-W invocation features

Runs on CPU with a fixed thread budget (--threads, default 4) at low priority.

    nice -n 19 python experiments/flat_baselines_loops.py
"""
import argparse
import math
import os
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMP_W = ['max_log_lambda_s', 'frac_amplified_s', 'mean_log_lambda_s', 'log_loop_work', 'log_total_work',
          'has_dynamic_loop', 'n_numpy', 'n_math', 'n_string', 'n_arith', 'w_numpy', 'w_math', 'w_string', 'w_arith']
LOG_LINEAR = {
    'rows only': ['log_rows'],
    'rows + max log Lambda': ['log_rows', 'max_log_lambda_s'],
    'rows + scope shape': ['log_rows', 'max_log_lambda_s', 'frac_amplified_s', 'mean_log_lambda_s', 'log_loop_work'],
    'total work only': ['log_total_work'],
}


def graph_features(db, udf):
    from udf_graph.annotate_graph_info import compute_lambda_v
    g = nx.read_gpickle(os.path.join(REPO, 'data', 'loop_runs', 'duckdb_pushdown', 'dbs', db, 'created_graphs',
                                     f'{udf}.loopend.gpickle'))
    compute_lambda_v(g, 'act')
    inv = next(a for _, a in g.nodes(data=True) if a['type'] == 'INVOCATION')
    f = {'log_rows': math.log(max(1.0, float(inv.get('in_rows_act') or 1)))}
    f.update({k: float(inv.get(k, 0.0) or 0.0) for k in LAMP_W})
    ops = Counter()
    for _, a in g.nodes(data=True):
        if a['type'] == 'LOOP_HEAD':
            ops['#loops'] += 1
        elif a['type'] == 'BRANCH':
            ops['#branches'] += 1
        elif a['type'] == 'COMP':
            for op in a.get('ops') or []:
                ops[f'op_{op}'] += 1
            if a.get('lib_onehot') not in (None, 'null'):
                ops[f'lib_{a["lib_onehot"]}'] += 1
    f.update(ops)
    return f


def qerr(label, pred):
    return np.maximum(pred / label, label / pred)


def gm(values):
    return float(np.exp(np.mean(np.log(values))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    import xgboost
    from sklearn.linear_model import LinearRegression

    res_dir = os.path.join(REPO, 'data', 'results', 'loo_loops_act_seed0')
    pq = pd.read_csv(os.path.join(res_dir, 'per_query_act.csv'))
    queries = pq[pq.config == 'graceful'][['db', 'udf', 'label']].reset_index(drop=True)

    cache = os.path.join(REPO, 'data', 'results', 'flat_baselines_loops_features.csv')
    if os.path.exists(cache):
        feats = pd.read_csv(cache)
    else:
        rows = [graph_features(db, udf) for db, udf in zip(queries.db, queries.udf)]
        feats = pd.concat([queries, pd.DataFrame(rows).fillna(0.0)], axis=1)
        feats.to_csv(cache, index=False)
    structural = ['log_rows'] + sorted(c for c in feats.columns if c.startswith(('op_', 'lib_', '#')))
    models = {f'log-linear: {k}': ('lin', v) for k, v in LOG_LINEAR.items()}
    models['XGBoost: structural'] = ('xgb', structural)
    models['XGBoost: structural + LAMP-W'] = ('xgb', structural + LAMP_W)

    y = np.log(feats['label'].values)
    per_fold = {m: {} for m in models}
    for db in sorted(feats.db.unique()):
        tr, te = feats.db != db, feats.db == db
        for name, (kind, cols) in models.items():
            X = feats[cols].values
            if kind == 'lin':
                model = LinearRegression().fit(X[tr], y[tr])
            else:
                model = xgboost.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=5,
                                             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                             n_jobs=args.threads, random_state=0).fit(X[tr], y[tr])
            pred = np.clip(np.exp(model.predict(X[te])), 0.01, None)
            q = qerr(feats['label'].values[te], pred)
            per_fold[name][db] = (np.quantile(q, 0.5), np.quantile(q, 0.95))
        print(f'fold {db} done', flush=True)

    gnn = pq.groupby(['config', 'db'])['qerr'].agg(q50=lambda s: s.quantile(0.5), q95=lambda s: s.quantile(0.95))
    g50 = gm(gnn.loc['graceful']['q50'])
    lines = [f'{"model":<34}{"Q50":>8}{"Q95":>8}{"gap vs GRACEFUL (Q50)":>24}']
    for name in models:
        q50 = gm([v[0] for v in per_fold[name].values()])
        q95 = gm([v[1] for v in per_fold[name].values()])
        lines.append(f'{name:<34}{q50:>8.3f}{q95:>8.2f}{100 * (q50 / g50 - 1):>+23.1f}%')
    for c, label in [('graceful', 'GNN: GRACEFUL'), ('lamp_w', 'GNN: LAMP-W')]:
        q50, q95 = gm(gnn.loc[c]['q50']), gm(gnn.loc[c]['q95'])
        lines.append(f'{label:<34}{q50:>8.3f}{q95:>8.2f}{100 * (q50 / g50 - 1):>+23.1f}%')
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(REPO, 'data', 'results', 'flat_baselines_loops.txt'), 'w') as f:
        f.write(f'19 LOO folds, loop-bearing queries, actual cardinalities, {len(feats)} queries\n{report}\n')


if __name__ == '__main__':
    main()
