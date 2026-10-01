import json
import math
import os
import re
import time
from typing import List, Tuple, Dict, Union

import duckdb
import networkx as nx
from deepdb.inference import FilterCond, DeepDBEstimator

from card_est.run_wanderjoin import get_wj_card_est
from cross_db_benchmark.benchmark_tools.generate_workload import negate_op
from cross_db_benchmark.benchmark_tools.dbms.parse_filter import parse_filter
from udf_graph.dbms_wrapper import DBMSWrapper
from udf_graph.helper import extract_filter_join_conds_below_udf_from_plan, JoinCond, join_cond_to_sql
from udf_graph.utils import extract_where_conditions, q_err


def enhanceUDFgraph_query(udfgraph, func_name, func_header, dbms: str):
    """
    Analyzes the header of the UDF definition in DuckDB/Postgres style to extract the input and output datatypes
    These features are relevant for the invocation and return nodes
    TODO: Adjust the SQL_DATA_TYPES list once we have more datatypes to support; currently, FLOAT is enough
    func_header is a string that looks like:
    postgres: CREATE FUNCTION func_0(col2 FLOAT,col5 FLOAT) RETURNS FLOAT
    duckdb: DEF FUNC_0(PS_SUPPKEY:INT,PS_PARTKEY:INT) -> INT:
    """
    func_header = func_header.upper()  # capitalize everything
    # extract the function arguments
    if dbms == 'postgres':
        search_str = f'{func_name.upper()}((.*)) RETURNS'
    elif dbms == 'duckdb':
        search_str = f'{func_name.upper()}((.*)) ->'
    else:
        raise Exception(f'Unknown DBMS: {dbms}')
    try:
        args = re.search(search_str, func_header).group(1)
    except Exception as e:
        print(search_str)
        print(func_header)
        print(func_name)
        raise e
    args = args[1:-1]  # remove the brackets
    args_lst = args.split(",")

    # determine the input datatypes
    dts = []
    for arg in args_lst:
        if dbms == 'postgres':
            dt = re.search('( .*)', arg).group(1)
        elif dbms == 'duckdb':
            dt = arg.split(":")[1]
        else:
            raise Exception(f'Unknown DBMS: {dbms}')

        dt = dt.lstrip().rstrip()  # remove white spaces at beginning and end of string
        dts.append(dt)
    # determine the output datatype
    if dbms == 'postgres':
        ret_dt = func_header.split("RETURNS ")[1].rstrip().lstrip()
    elif dbms == 'duckdb':
        ret_dt = func_header.split("-> ")[1].rstrip().lstrip()
    else:
        raise Exception(f'Unknown DBMS: {dbms}')

    # iterate over all nodes in the graph; append feature datatype to INVOCATION and RETURN node
    for node in udfgraph.nodes:
        if udfgraph.nodes[node]["type"] == "INVOCATION":
            nx.set_node_attributes(udfgraph, {node: dts}, name="in_dts")
            nx.set_node_attributes(udfgraph, {node: len(dts)}, name="no_params")
        elif udfgraph.nodes[node]["type"] == "RETURN":
            nx.set_node_attributes(udfgraph, {node: ret_dt}, name="out_dts")
    return udfgraph


def enhanceUDFgraph_card(graph, table_name, dbms_wrapper: DBMSWrapper, pullup_udf: bool, udf_intermed_pos:bool, query_plan: Dict,
                         database_statistics: Dict, sql_query: str, est_stats: List[Dict], duckdb_kwargs: Dict,
                         schema_relationships: Dict,
                         deepdb_estimator: DeepDBEstimator = None, card_est_assume_lazy_eval: bool = False,
                         verbose: bool = False, skip_wj: bool = False, skip_deepdb:bool=False):
    # create materialized view based on the raw query
    # create mv if udf call is input to an aggregate function
    mat_view_name = mat_view_handling(raw_query=sql_query, dbms_wrapper=dbms_wrapper, table_name=table_name,
                                      delete=False, pullup_udf=pullup_udf or udf_intermed_pos)

    assert not card_est_assume_lazy_eval
    where_conds, join_conds, tables = extract_filter_join_conds_below_udf_from_plan(query_plan_node=query_plan,
                                                                                    database_statistics=database_statistics,
                                                                                    udf_found=False,
                                                                                    assume_lazy_dbms=card_est_assume_lazy_eval,
                                                                                    schema_relationships=schema_relationships, )

    if verbose:
        print(f"Where conds: {where_conds}", flush=True)
        print(f"Join conds: {join_conds}", flush=True)

    if pullup_udf:
        # get where conditions from sql instead of plan - duckdb is introducing additonal conditions which confuse deepdb
        where_conds = extract_where_conditions(raw_query=sql_query, target_table=None)

    if verbose:
        print(f'Table name: {table_name}', flush=True)
        print(f'sql query: {sql_query}', flush=True)
        print(f"Where conds: {where_conds}", flush=True)

    deepdb_tables = tables[:]

    if mat_view_name is not None:
        tables = [mat_view_name]
    else:
        # udf aggr is only query which is on top of joins, so mult. tables involved
        assert len(tables) == 1, f"Multiple tables found in UDF graph: {tables} - {query_plan}"

    # compute all paths from INV to RET
    paths = get_all_paths(graph)

    assert len(paths) > 0, "No paths found in UDF graph"

    col_col_contained = False

    # iterate over all paths that exist from INV to RET
    for path in paths:
        # retrieve all conditions that are present in the path
        if mat_view_name is not None:
            dbms_conditions = []
        else:
            assert len(tables) == 1, f"Multiple tables found in UDF graph: {tables} - {query_plan}"
            dbms_conditions = where_conds[:] + join_conds[:]

        deepdb_conditions = where_conds[:]

        # iterate over all nodes in the path and extract the conditions
        num_col_col_comps = 0

        for node in path:
            if "branch_cond" in graph.nodes[node].keys():
                branch_cond = graph.nodes[node]["branch_cond"]
                if branch_cond.startswith('NOT '):
                    negate = True
                    branch_cond = branch_cond[4:]
                else:
                    negate = False
                filter = parse_filter(f'({branch_cond})', duckdb=True, sql=sql_query)
                assert len(filter.children) == 0, f"Branch condition has children: {filter}"

                literal = filter.literal.strip().strip('"')

                found_col_col = False
                for entry in database_statistics['column_stats']:
                    if entry['table_name'] == table_name and entry['column_name'] == literal:
                        # we found a col = col condition
                        found_col_col = True
                        break
                if found_col_col:
                    num_col_col_comps += 1

                if len(filter.column) == 1:
                    filter_column = filter.column[0]
                else:
                    filter_column = filter.column[1]

                if filter_column.startswith('"') and filter_column.endswith('"'):
                    filter_column = filter_column[1:-1]

                if negate:
                    op = str(negate_op(filter.operator))
                else:
                    op = str(filter.operator)

                if len(filter.column) == 2:
                    assert table_name == filter.column[0].strip(
                        '"'), f"Table name in filter condition does not match table name: {table_name} / {filter.column[0]}"

                dbms_f = FilterCond(table_name=table_name if mat_view_name is None else mat_view_name,
                                    column_name=filter_column, # if not udf_intermed_pos else f'{table_name}_{filter_column}',
                                    operator=op, value=filter.literal)

                dbms_conditions.append(dbms_f)

                if not found_col_col:
                    f = FilterCond(table_name=table_name, column_name=filter_column, operator=op,
                                   value=filter.literal)
                    deepdb_conditions.append(f)

        # create the query based on the conditions along the path
        count_query = create_query(tables, dbms_conditions,
                                   with_count=True)
        query = create_query(tables, dbms_conditions,
                             with_count=False)

        # use helper function that runs the query to determine the cardinality
        try:
            path_act_card = dbms_wrapper.exec(count_query)[0][0]
            path_est_card = dbms_wrapper.get_est_card(query)
        except Exception as e:
            print(f'count query: {count_query}', flush=True)
            print(json.dumps(query_plan), flush=True)
            raise e

        if verbose:
            print(f'DeepDb conds: {deepdb_conditions}, deepdb tables: {deepdb_tables}', flush=True)

        t0 = time.time()
        msg = None
        if len(deepdb_conditions) == 0 and len(deepdb_tables) == 1:
            path_dd_card = path_est_card
        elif deepdb_estimator is not None and not skip_deepdb:
            try:
                join_conds_str = [join_cond_to_sql(j) for j in join_conds]
                path_dd_card = deepdb_estimator.estimate_card(
                    tuple([d for d in deepdb_conditions if isinstance(d, FilterCond)]), frozenset(deepdb_tables),
                    join_conditions=tuple(join_conds_str))
            except Exception as e:
                print(f"Error in deepdb estimation: {e}")
                path_dd_card = path_est_card
                msg = e

            path_dd_card = round(path_dd_card)
        else:
            path_dd_card = path_est_card
        t1 = time.time()

        if skip_wj:
            path_wj_card = 0
            wj_is_dd = False
        else:
            if len(join_conds) > 0:
                path_wj_card = get_wj_card_est(
                    filter_conds=deepdb_conditions,
                    join_conds=join_conds,
                    duckdb_kwargs=duckdb_kwargs
                )
                wj_is_dd = False
            else:
                wj_is_dd = True
                path_wj_card = path_dd_card

        t2 = time.time()

        if verbose:
            print(f'Path cardinalities: {path_act_card} / {path_est_card} / {path_dd_card} / {path_wj_card}',
                  flush=True)
            print(f'DeepDB conds: {deepdb_conditions} / {deepdb_tables}', flush=True)
            print(f'Query: {query}', flush=True)

        # adjust the cardinalities to the number of column-column comparisons
        if num_col_col_comps > 0:
            col_col_contained = True
            path_est_card = path_est_card // (num_col_col_comps + 1)
            path_dd_card = path_dd_card // (num_col_col_comps + 1)
            path_wj_card = path_wj_card // (num_col_col_comps + 1)

        # compute the q-error and latency
        q_err_deepdb = q_err(path_dd_card, path_act_card)
        q_err_dbms = q_err(path_est_card, path_act_card)
        q_err_wj = q_err(path_wj_card, path_act_card)

        # this was probably a bug, anyway rarely happens
        if q_err_deepdb > 100 * q_err_dbms:
            path_dd_card = path_est_card
            msg = "DeepDB error too high"

            q_err_deepdb = q_err(path_dd_card, path_act_card)

            if wj_is_dd:
                # wanderjoin has not been executed, instead the deepdb results have been used
                path_wj_card = path_est_card
                q_err_wj = q_err(path_wj_card, path_act_card)

        # this was probably a bug, anyway rarely happens
        if q_err_wj > 100 * q_err_dbms:
            path_wj_card = path_est_card
            msg = "WanderJoin error too high"

            q_err_wj = q_err(path_wj_card, path_act_card)

        est_stats.append({
            'q_errors_dbms': q_err_dbms,
            'q_errors_deepdb': q_err_deepdb,
            'q_errors_wj': q_err_wj,
            'latencies_ms_dd': (t1 - t0) * 1000,
            'latencies_ms_wj': (t2 - t1) * 1000,
            'msg': msg
        })

        assert isinstance(path_act_card,
                          int), f"Cardinality is not an integer: {path_act_card} / {type(path_act_card)}"
        assert isinstance(path_est_card,
                          int), f"Cardinality is not an integer: {path_est_card} / {type(path_est_card)}"

        # add the cardinality to the list of cardinality for all nodes in this path
        for node in path:
            if "card_list_act" not in graph.nodes[node].keys():
                nx.set_node_attributes(graph, {node: [path_act_card]}, name="card_list_act")
                nx.set_node_attributes(graph, {node: [path_est_card]}, name="card_list_est")
                nx.set_node_attributes(graph, {node: [path_dd_card]}, name="card_list_deepdb")
                nx.set_node_attributes(graph, {node: [path_wj_card]}, name="card_list_wj")
            else:
                graph.nodes[node]["card_list_act"].append(path_act_card)
                graph.nodes[node]["card_list_est"].append(path_est_card)
                graph.nodes[node]["card_list_deepdb"].append(path_dd_card)
                graph.nodes[node]["card_list_wj"].append(path_wj_card)

    # delete the materialized view
    if sql_query is not None:
        _ = mat_view_handling(raw_query=sql_query, dbms_wrapper=dbms_wrapper, table_name=table_name, delete=True,
                              pullup_udf=pullup_udf or udf_intermed_pos)
    graph = det_final_card(graph)
    return graph, col_col_contained


def get_all_paths(udf_graph: nx.DiGraph) -> List[Tuple[int]]:
    """
    Helper function that first determines the node id of the invocation node and of all return nodes
    Next, the function computes all paths from the invocation node to the return nodes
    """

    invocation_node = None
    return_nodes = []

    # iterate over all nodes in the graph and determine the invocation node and all return nodes
    for node in udf_graph.nodes._nodes.keys():
        if udf_graph.nodes[node]["type"] == "INVOCATION":
            assert invocation_node is None, "Multiple invocation nodes found in UDF graph"
            invocation_node = node
        elif udf_graph.nodes[node]["type"] == "RETURN":
            return_nodes.append(node)

    assert len(return_nodes) == 1, "Multiple return nodes found in UDF graph"
    assert invocation_node is not None
    return_node = return_nodes[0]

    # retrieve all edges connecting LOOP_HEAD -> LOOP_END
    # we need to be careful with them, since they are shortcuts that can lead to duplicated cardinality annotations
    loop_edges = []
    for edge in udf_graph.edges:
        if udf_graph.nodes[edge[0]]["type"] == "LOOP_HEAD" and udf_graph.nodes[edge[1]]["type"] == "LOOP_END":
            loop_edges.append(edge)

    assert len(loop_edges) == len(set(loop_edges)), f"Loop edges are not unique: {loop_edges}"

    # compute all paths from the invocation node to the return nodes
    paths = nx.all_simple_paths(udf_graph, source=invocation_node, target=return_node)
    paths = list(paths)

    # extract all loop nodes which have a not-empty body
    loop_start_nodes_with_body = []
    for path in paths:
        for edge in zip(path[:-1], path[1:]):
            if udf_graph.nodes[edge[0]]["type"] == "LOOP_HEAD" and udf_graph.nodes[edge[1]]["type"] != "LOOP_END":
                loop_start_nodes_with_body.append(edge[0])

    out_paths = []
    for path in paths:
        # discard all paths containing LOOP_HEAD -> LOOP_END edges that have a not-empty body
        if any([edge in loop_edges and edge[0] in loop_start_nodes_with_body for edge in zip(path[:-1], path[1:])]):
            continue

        out_paths.append(path)

    if len(out_paths) == 0:
        for path in paths:
            print([udf_graph.nodes[node]["type"] for node in path])
    assert len(out_paths) > 0, f"No paths found in UDF graph {paths}"
    return out_paths


def create_query(tables: List[str], conditions: List[Union[FilterCond, JoinCond]], with_count: bool):
    """
    Helper function that gets a list of conditions as input
    From this list of conditions, the functions creates a query that can be used to determine the cardinality of the different branches
    """
    for c in conditions:
        assert isinstance(c, FilterCond) or isinstance(c, JoinCond), f"Unknown condition type: {c}"

    tables_str_list = [f'"{table}"' for table in tables]
    tables_str = ', '.join(tables_str_list)

    join_conds = [cond for cond in conditions if isinstance(cond, JoinCond)]
    assert len(join_conds) == len(
        tables) - 1, f"Number of join conditions does not match number of tables: {join_conds} / {tables}"

    if len(conditions) > 0:
        query = f'SELECT {"COUNT(*)" if with_count else "*"} FROM {tables_str} WHERE '

        cond_str_list = []
        for cond in conditions:
            if isinstance(cond, FilterCond):
                if cond.table_name == 'helper_view':
                    cond_str_list.append(
                        f'{cond.column_name} {cond.operator} {cond.value}')
                else:
                    cond_str_list.append(
                        f'"{cond.table_name}"."{cond.column_name}" {cond.operator} {cond.value}')
            elif isinstance(cond, JoinCond):
                cond_str_list.append(
                    f'"{cond.table_name1}"."{cond.column_name1}" {cond.operator} "{cond.table_name2}"."{cond.column_name2}"')

        cond_query = ' AND '.join(
            cond_str_list)

        assert cond_query.strip() != "", f"No conditions found in query: {cond_query} \n {tables} \n {conditions}"
        return query + cond_query + ";"
    else:
        query = f'SELECT {"COUNT(*)" if with_count else "*"} FROM {tables_str};'
        return query


def det_final_card(graph):
    # def num_pred(pred_iter, graph):
    #     counter = 0
    #     for elem in pred_iter:
    #         if graph.nodes[elem]["type"] != "VAR":
    #             counter += 1
    #     return counter

    for node in graph.nodes:
        if "card_list_act" in graph.nodes._nodes[node].keys():
            nx.set_node_attributes(graph, {node: sum(graph.nodes[node]["card_list_act"])}, name="in_rows_act")
            nx.set_node_attributes(graph, {node: sum(graph.nodes[node]["card_list_est"])}, name="in_rows_est")
            nx.set_node_attributes(graph, {node: sum(graph.nodes[node]["card_list_deepdb"])}, name="in_rows_deepdb")
            nx.set_node_attributes(graph, {node: sum(graph.nodes[node]["card_list_wj"])}, name="in_rows_wj")

    return graph


# cardinality source -> UDF node attribute; every source must map to its own attribute
CARD_ATTR = {'act': 'in_rows_act', 'est': 'in_rows_est', 'dd': 'in_rows_deepdb', 'wj': 'in_rows_wj'}

# fixed trip count of the Auto-Bound fallback rule
AUTOBOUND_FALLBACK_ITERS = 10

_DEFAULT_LOOP_ITERS = 100  # fallback iteration estimate for while-loops or range loops with unknown bounds

# Optional variants (environment switches; unset = the LAMP features of the paper):
#   LAMP_BRANCH_WEIGHT=on   weight each node's loop work by the fraction of rows that reach it,
#                           in_rows(v) / in_rows(INVOCATION) under the same cardinality source
#                           (branch-guarded regions execute only for the rows passing the guard)
#   LAMP_FALLBACK=abstain   Auto-Bound's fallback tier contributes no amplification (lambda = 1)
#                           instead of the fixed AUTOBOUND_FALLBACK_ITERS guess


def branch_weight_enabled() -> bool:
    return os.environ.get('LAMP_BRANCH_WEIGHT', 'off') == 'on'


def loop_body(graph: nx.DiGraph, head) -> set:
    """
    Nodes executed once per iteration of the loop starting at `head`: every node on a
    path from the LOOP_HEAD to its own LOOP_END (head included, LOOP_END excluded).
    Code after the loop, including a following sequential loop, is not an ancestor of
    this LOOP_END and is therefore excluded.
    Graphs built without LOOP_END nodes fall back to the loop_part-marked descendants,
    which cannot separate sequential loops.
    """
    ends = [s for s in graph.successors(head) if graph.nodes[s].get('type') == 'LOOP_END']
    if len(ends) == 1:
        return (nx.descendants(graph, head) & nx.ancestors(graph, ends[0])) | {head}
    return {v for v in nx.descendants(graph, head)
            if str(graph.nodes[v].get('loop_part', 'False')).strip().lower() == 'true'} | {head}


OP_CATEGORIES = ['numpy', 'math', 'string', 'arith']


def op_category(attrs: Dict):
    """Operation category of a COMP node: library family of its call, else arithmetic."""
    lib = str(attrs.get('lib_onehot', 'null'))
    if lib.startswith('numpy.'):
        return 'numpy'
    if lib.startswith('math.'):
        return 'math'
    if lib not in ('null', 'None', ''):
        return 'string'  # str methods (upper, find, isdigit, ...)
    if attrs.get('ops'):
        return 'arith'
    return None


def annotate_loop_scope_features(graph: nx.DiGraph, loop_lambda: Dict, card_attr: str) -> None:
    """
    Scope-composed loop amplification and work summaries on the INVOCATION node.

    Lambda(v) = prod lambda(h) over the loops h whose body contains v, so nested loops multiply and
    sequential loops stay separate regions. Stored per node as `lambda_scope` = log Lambda(v).

    INVOCATION receives, over the UDF-side nodes V_U (all nodes except VAR):
      max_log_lambda_s   max_v log Lambda(v)                         strongest nested product
      frac_amplified_s   |{v : Lambda(v) > 1}| / |V_U|               share of the UDF inside loops
      mean_log_lambda_s  mean_v log Lambda(v)                        average exposure
      log_loop_work      log sum_v Lambda(v)                           node executions per invocation:
                                                                sequential regions add, nested
                                                                regions multiply, body size counts
      log_total_work     log max(1, rows) + log_loop_work       total node executions for the query
      n_<op>, w_<op>     log(1 + #COMP nodes of category op), log(1 + sum Lambda(v) over them), for
                         op in OP_CATEGORIES: how many numpy / math / string / arithmetic
                         operations the UDF contains and how often they execute per invocation
    """
    log_lam = {v: 0.0 for v in graph.nodes}
    for head, lam in loop_lambda.items():
        log_l = math.log(max(1.0, lam))
        for v in loop_body(graph, head):
            log_lam[v] += log_l
    nx.set_node_attributes(graph, log_lam, name='lambda_scope')

    ell = [log_lam[v] for v, a in graph.nodes(data=True) if a.get('type') != 'VAR']
    if not ell:
        return
    m = max(ell)

    # executions per invocation are counted per node; with LAMP_BRANCH_WEIGHT only the share of rows
    # that reaches a node (it may sit behind a branch guard) is counted
    reach = {v: 1.0 for v in graph.nodes}
    if branch_weight_enabled():
        inv_rows = [float(a.get(card_attr) or 0) for _, a in graph.nodes(data=True) if a.get('type') == 'INVOCATION']
        if inv_rows and inv_rows[0] > 0:
            for v, a in graph.nodes(data=True):
                if a.get('type') != 'VAR' and a.get(card_attr) is not None:
                    reach[v] = min(1.0, max(0.0, float(a[card_attr]) / inv_rows[0]))
    work = [reach[v] * math.exp(log_lam[v] - m) for v, a in graph.nodes(data=True) if a.get('type') != 'VAR']
    log_loop_work = m + math.log(max(sum(work), 1e-12))

    # operation mix: static counts and loop-amplified executions per operation category
    op_count = {c: 0.0 for c in OP_CATEGORIES}
    op_work = {c: 0.0 for c in OP_CATEGORIES}
    for v, attrs in graph.nodes(data=True):
        if attrs.get('type') != 'COMP':
            continue
        category = op_category(attrs)
        if category is not None:
            op_count[category] += 1
            op_work[category] += reach[v] * math.exp(log_lam[v])

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') == 'INVOCATION':
            rows = float(attrs.get(card_attr, 1) or 1)
            nx.set_node_attributes(graph, {node_id: {
                'max_log_lambda_s': m,
                'frac_amplified_s': sum(1 for l in ell if l > 0.0) / len(ell),
                'mean_log_lambda_s': sum(ell) / len(ell),
                'log_loop_work': log_loop_work,
                'log_total_work': math.log(max(1.0, rows)) + log_loop_work,
                **{f'n_{c}': math.log1p(op_count[c]) for c in OP_CATEGORIES},
                **{f'w_{c}': math.log1p(op_work[c]) for c in OP_CATEGORIES},
            }})


def compute_lambda_v(graph: nx.DiGraph, card_type: str = 'act') -> nx.DiGraph:
    """
    Annotate a UDF graph with loop amplification.

    Trip count lambda(h) of every LOOP_HEAD h:
      the constant bound if the loop has a fixed bound,
      otherwise the Auto-Bound estimate (lambda_est, log scale) if present,
      otherwise the input cardinality of the selected source, or _DEFAULT_LOOP_ITERS.

    Annotations:
      INVOCATION node   scope summaries, work and operation mix (annotate_loop_scope_features)
                        and has_dynamic_loop (some loop has no fixed bound)
      every node        lambda_v = log of the product of lambda(h) over the enclosing loops,
                        used by the per-node encoding LV-PN

    card_type selects the cardinality attribute (CARD_ATTR): act, est, dd (DeepDB) or wj (WanderJoin).

    Environment switches used by the Auto-Bound experiments:
      LAMP_AUTOBOUND=off      dynamic loops use AUTOBOUND_FALLBACK_ITERS instead of the estimate
      LAMP_AUTOBOUND=oracle   dynamic loops use the measured mean trip count (lambda_true)
    """
    _card_attr = CARD_ATTR[card_type]

    loop_lambda: dict = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') != 'LOOP_HEAD':
            continue

        fixed = str(attrs.get('fixed_iter', 'False')).strip().lower() == 'true'
        try:
            no_iter = int(attrs.get('no_iter', -1))
        except (ValueError, TypeError):
            no_iter = -1

        if fixed and no_iter > 0:
            lam = float(no_iter)
        else:
            lambda_est_attr = attrs.get('lambda_est')
            mode = os.environ.get('LAMP_AUTOBOUND', 'on')
            if mode == 'off' and lambda_est_attr is not None:
                lambda_est_attr = math.log(AUTOBOUND_FALLBACK_ITERS)
            elif mode == 'oracle' and attrs.get('lambda_true') is not None:
                lambda_est_attr = attrs['lambda_true']
            if (mode == 'on' and attrs.get('bound_strategy') == 'fallback'
                    and os.environ.get('LAMP_FALLBACK', 'const') == 'abstain'):
                lambda_est_attr = 0.0
            if lambda_est_attr is not None:
                try:
                    lam = max(1.0, math.exp(float(lambda_est_attr)))
                except (ValueError, TypeError):
                    lambda_est_attr = None
            if lambda_est_attr is None:
                try:
                    in_rows = float(attrs.get(_card_attr, -1) or -1)
                except (ValueError, TypeError):
                    in_rows = -1.0
                lam = in_rows if in_rows > 0 else float(_DEFAULT_LOOP_ITERS)

        loop_lambda[node_id] = lam

    annotate_loop_scope_features(graph, loop_lambda, _card_attr)

    # per-node amplification for LV-PN; only nodes marked loop_part are inside a loop body
    for node_id, attrs in graph.nodes(data=True):
        inside_loop = str(attrs.get('loop_part', 'False')).strip().lower() == 'true'
        lambda_v = 1.0
        if node_id in loop_lambda and attrs.get('type') == 'LOOP_HEAD':
            lambda_v = loop_lambda[node_id]
        elif loop_lambda and inside_loop:
            for a in nx.ancestors(graph, node_id):
                if a in loop_lambda:
                    lambda_v *= loop_lambda[a]
        nx.set_node_attributes(graph, {node_id: math.log(lambda_v)}, name='lambda_v')

    has_dynamic_loop = float(any(
        str(attrs.get('fixed_iter', 'False')).strip().lower() != 'true'
        for _, attrs in graph.nodes(data=True) if attrs.get('type') == 'LOOP_HEAD'))
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') == 'INVOCATION':
            nx.set_node_attributes(graph, {node_id: has_dynamic_loop}, name='has_dynamic_loop')

    return graph


def mat_view_handling(raw_query, delete, dbms_wrapper: DBMSWrapper, table_name, pullup_udf: bool):
    """
    Helper function that creates a materialized view based on the raw query - only necessary if the udf call is input to an aggregate function (i.e. consumes data produced by joins/...)
    :param raw_query: original sql query
    :param delete: create / delete the materialized view
    :param dbms_wrapper:
    :param table_name:
    :return:
    """

    def transform_raw_query(raw_query, pullup: bool):
        transformed_query = ""
        # CASE 1: UDF call is input to an aggregate function
        # CASE 2: UDF call is input to a GROUP BY clause
        if ("SUM(func_" in raw_query or "AVG(func_" in raw_query or "COUNT(func_" in raw_query) or \
                ("GROUP BY func_" in raw_query):
            # remove everything beteween SELECT and FROM and replace with *
            transformed_query = re.sub("(?<=SELECT )(.*)(?= FROM)", "\"" + table_name + "\"" + ".*", raw_query)
            # remove group by clause if there is one
            transformed_query = re.sub("(GROUP BY)(.*)", ";", transformed_query)
            return transformed_query
        elif pullup:
            # UDF has been pulled up
            # raw query should be nested query
            assert '(SELECT' in raw_query, f"Raw query does not contain nested query: {raw_query}"
            parts = raw_query.split('(SELECT')
            assert len(parts) == 2, f"Multiple nested queries found: {raw_query}"
            assert ' as nested_query' in parts[1], f"Nested query does not have alias: {raw_query}"
            parts = parts[1].split(') as nested_query')
            assert len(parts) == 2, f"Multiple nested queries found: {raw_query}"

            if not parts[0].startswith('*'):
                tmp = parts[0]
                splits = tmp.split(' FROM ')
                assert len(splits)==2, f"Multiple FROM clauses found: {tmp}"
                transformed_query = f'SELECT * FROM {splits[1]}'
            else:
                transformed_query = f'SELECT {parts[0]}'
            return transformed_query
        else:
            return None

    # create the materialized view based on the raw query
    if not delete:
        transformed_query = transform_raw_query(raw_query, pullup=pullup_udf)
        if transformed_query is not None:
            if dbms_wrapper.dbms == 'postgres':
                mvq = "CREATE MATERIALIZED VIEW helper_view AS " + transformed_query + ";"
            elif dbms_wrapper.dbms == 'duckdb':
                mvq = "CREATE VIEW helper_view AS " + transformed_query + ";"
            else:
                raise Exception(f'Unknown DBMS: {dbms_wrapper.dbms}')
            try:
                dbms_wrapper.exec(mvq)
            except duckdb.CatalogException as e:
                dbms_wrapper.exec("DROP VIEW IF EXISTS helper_view;")
                dbms_wrapper.exec(mvq)
            except Exception as e:
                print(f"Failed to create materialized view: {e}", flush=True)
                dbms_wrapper.exec("DROP VIEW IF EXISTS helper_view;")
                dbms_wrapper.exec(mvq)
            return "helper_view"
        else:
            return None
    # delete the materialized view after it was used if it exists
    else:
        transformed_query = transform_raw_query(raw_query, pullup=pullup_udf)
        if transformed_query is not None:
            if dbms_wrapper.dbms == 'postgres':
                mvq = "DROP MATERIALIZED VIEW helper_view;"
            elif dbms_wrapper.dbms == 'duckdb':
                mvq = "DROP VIEW helper_view;"
            else:
                raise Exception(f'Unknown DBMS: {dbms_wrapper.dbms}')
            dbms_wrapper.exec(mvq)
        return None
