# different features used for plan nodes, filter columns etc. used for postgres plans

PostgresUDFTest = dict(
    PLAN_FEATURES=['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['reltuples', 'relpages'],
    INVOC_FEATURES=['in_rows', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows', 'lib', 'ops', 'loop_part'],
    RETURN_FEATURES=['out_dts', 'in_rows'],
    BRANCH_FEATURES=['in_rows', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

DuckDBActUDF = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_act', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

DuckDBActUDFFilterUDF = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_act', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

DuckDBEstUDF = dict(
    PLAN_FEATURES=['est_card', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_est', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_est', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_est', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_est', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

DuckDBEstUDFFilterUDF = dict(
    PLAN_FEATURES=['est_card', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_est', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_est', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_est', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_est', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_est', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

DuckDBDeepUDFFilterUDF = dict(
    PLAN_FEATURES=['dd_est_card', 'op_name', 'dd_est_children_card', 'above_udf_filter'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_deepdb', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_deepdb', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_deepdb', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_deepdb', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

DuckDBWJUDFFilterUDF = dict(
    PLAN_FEATURES=['wj_est_card', 'op_name', 'wj_est_children_card', 'above_udf_filter'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_deepdb', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_deepdb', 'ops', 'loop_part', 'lambda_v'],
    RET_FEATURES=['out_dts', 'in_rows_deepdb', 'lambda_v'],
    BRANCH_FEATURES=['in_rows_deepdb', 'cmops', 'loop_part', 'lambda_v'],
    LOOP_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
    LOOPEND_FEATURES=['in_rows_deepdb', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part', 'lambda_v'],
)

PostgresTrueCardDetail = dict(
    PLAN_FEATURES=['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

PostgresEstSystemCardDetail = dict(
    PLAN_FEATURES=['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

PostgresDeepDBEstSystemCardDetail = dict(
    PLAN_FEATURES=['dd_est_card', 'est_width', 'workers_planned', 'op_name', 'dd_est_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature'],
    COLUMN_FEATURES=['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac'],
    OUTPUT_COLUMN_FEATURES=['aggregation'],
    TABLE_FEATURES=['reltuples', 'relpages'],
)

DuckDBActUDFFilterUDF_NoLV = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_act'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

# Scope-composed amplification summaries on the INVOCATION node (annotate_loop_scope_features).
# Lambda(v) is composed over the body of each loop, so nested loops multiply and sequential
# loops stay separate regions; log_loop_work / log_total_work sum the amplified node executions.
# in_rows_* is routed to the configured card_type by create_udf_feat_list.
DuckDBActUDFFilterUDF_GraphLV_Scope = dict(
    PLAN_FEATURES=['act_card', 'op_name', 'act_children_card'],
    FILTER_FEATURES=['operator', 'literal_feature', 'on_udf'],
    COLUMN_FEATURES=['data_type'],
    OUTPUT_COLUMN_FEATURES=["aggregation", "udf_output"],
    TABLE_FEATURES=['estimated_size'],
    INV_FEATURES=['in_rows_act', 'in_dts', 'no_params',
                  'max_log_lambda_s', 'frac_amplified_s', 'mean_log_lambda_s',
                  'log_loop_work', 'log_total_work', 'has_dynamic_loop'],
    COMP_FEATURES=['in_rows_act', 'ops', 'loop_part'],
    RET_FEATURES=['out_dts', 'in_rows_act'],
    BRANCH_FEATURES=['in_rows_act', 'cmops', 'loop_part'],
    LOOP_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
    LOOPEND_FEATURES=['in_rows_act', 'loop_type', 'fixed_iter', 'no_iter', 'loop_part'],
)

# LAMP: scope summaries plus the operation mix and loop-scope edges.
# n_<op> / w_<op>: static count and loop-amplified executions of numpy / math / string /
# arithmetic operations (annotate_loop_scope_features). LOOP_SCOPE_EDGES connects every
# LOOP_HEAD to all statements of its body (add_loop_scope_edges).
DuckDBActUDFFilterUDF_GraphLV_Work = dict(
    DuckDBActUDFFilterUDF_GraphLV_Scope,
    INV_FEATURES=DuckDBActUDFFilterUDF_GraphLV_Scope['INV_FEATURES'] + [
        'n_numpy', 'n_math', 'n_string', 'n_arith', 'w_numpy', 'w_math', 'w_string', 'w_arith'],
    LOOP_SCOPE_EDGES=True,
)

# Component ablations: each variant removes exactly one component.
_OP_COUNTS = ['n_numpy', 'n_math', 'n_string', 'n_arith']
_OP_WORK = ['w_numpy', 'w_math', 'w_string', 'w_arith']
_BASE_INV = ['in_rows_act', 'in_dts', 'no_params']
# (1) without loop-scope edges
DuckDBActUDFFilterUDF_GraphLV_Work_NoEdges = dict(DuckDBActUDFFilterUDF_GraphLV_Work, LOOP_SCOPE_EDGES=False)
# (2) without the operation mix (scope summaries plus loop-scope edges)
DuckDBActUDFFilterUDF_GraphLV_Work_NoOps = dict(DuckDBActUDFFilterUDF_GraphLV_Scope, LOOP_SCOPE_EDGES=True)
# (3) without loop amplification of the operation mix (static counts n_<op> only)
DuckDBActUDFFilterUDF_GraphLV_Work_NoOpLambda = dict(
    DuckDBActUDFFilterUDF_GraphLV_Scope,
    INV_FEATURES=DuckDBActUDFFilterUDF_GraphLV_Scope['INV_FEATURES'] + _OP_COUNTS,
    LOOP_SCOPE_EDGES=True,
)
# (4) without the scope-composed amplification summaries (max / frac / mean log Lambda, loop and total work)
DuckDBActUDFFilterUDF_GraphLV_Work_NoScope = dict(
    DuckDBActUDFFilterUDF_GraphLV_Work,
    INV_FEATURES=_BASE_INV + ['has_dynamic_loop'] + _OP_COUNTS + _OP_WORK,
)

featurization_dict = {
    'ddact': DuckDBActUDF,
    'ddactfonudf': DuckDBActUDFFilterUDF,
    'ddactfonudfnolv': DuckDBActUDFFilterUDF_NoLV,
    'ddactfonudfgraphlvs': DuckDBActUDFFilterUDF_GraphLV_Scope,
    'ddactfonudfgraphlvw': DuckDBActUDFFilterUDF_GraphLV_Work,
    'ddactfonudfgraphlvwnoedge': DuckDBActUDFFilterUDF_GraphLV_Work_NoEdges,
    'ddactfonudfgraphlvwnoops': DuckDBActUDFFilterUDF_GraphLV_Work_NoOps,
    'ddactfonudfgraphlvwnooplam': DuckDBActUDFFilterUDF_GraphLV_Work_NoOpLambda,
    'ddactfonudfgraphlvwnoscope': DuckDBActUDFFilterUDF_GraphLV_Work_NoScope,
    'ddest': DuckDBEstUDF,
    'ddestfonudf': DuckDBEstUDFFilterUDF,
    'dddeepfonudf': DuckDBDeepUDFFilterUDF,
    'ddwjfonudf': DuckDBWJUDFFilterUDF,
    'pgzsdd': PostgresDeepDBEstSystemCardDetail,
    'pgzsact': PostgresTrueCardDetail,
}
