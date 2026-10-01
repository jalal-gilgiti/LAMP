"""
Top-level automatic annotation pipeline.

annotate_graph(graph, sql_query, udf_name, col_stats, db_name, ...)
    Takes a UDF graph (already processed by create_graph.py - LOOP_HEAD nodes
    must have influencing_params set by Algorithm 2) and annotates every
    dynamic LOOP_HEAD with:
        lambda_est   (float, log-scale)  - best estimate of log(no_iter)
        lambda_unc   (float, >= 0)        - uncertainty (0 = exact, 1 = high)
        bound_strategy (str)             - which estimation tier fired

    Also propagates to the INVOCATION node:
        has_uncertain_loop   (float)  1.0 if any loop has lambda_unc > 0
        max_lambda_unc       (float)  max uncertainty across loop heads
        log_total_ops_est    (float)  log(in_rows_est) + max(lambda_est)

annotate_graph_batch(graphs_and_queries, col_stats, db_name, ...)
    Convenience wrapper for annotating a list of (graph, sql, udf_name) tuples.

This module is the single entry point - callers do not need to import
sql_arg_parser or bound_estimator directly.
"""

import math
import networkx as nx
from typing import List, Optional, Tuple

from .sql_arg_parser import parse_call_args
from .bound_estimator import estimate_loop_bound


def annotate_graph(
    graph: nx.DiGraph,
    sql_query: str,
    udf_name: str,
    col_stats: dict,
    db_name: str,
    known_tables: Optional[List[str]] = None,
    udf_source: Optional[str] = None,
    value_length=None,
) -> nx.DiGraph:
    """
    Annotate all LOOP_HEAD nodes with automatic lambda estimates.

    Args:
        graph:         UDF graph (LOOP_HEAD nodes must have influencing_params).
        sql_query:     SQL query string that calls udf_name.
        udf_name:      UDF function name as called in the SQL (e.g. 'dyn_func_285').
        col_stats:     Dict from col_stats.load_col_stats().
        db_name:       Target database name.
        known_tables:  Table names in the DB schema (helps resolve column refs).
                       If None, inferred from col_stats keys.
        udf_source:    UDF source code (planner-visible); enables the source-based rules
                       (bound clipped by min(..., C), collection loops over a parameter).
        value_length:  callable (table, col) -> median value length of the column
                       (a catalog statistic, cf. avg_width in PostgreSQL's pg_stats).

    Returns:
        The graph, modified in-place.
    """
    # Infer known tables from col_stats if not provided
    if known_tables is None:
        known_tables = _infer_tables(col_stats, db_name)

    # Parse SQL call arguments once for this query
    parsed_args = parse_call_args(sql_query, udf_name, known_tables)

    loop_head_data = []  # collect (lambda_est, lambda_unc) for INVOCATION propagation

    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') != 'LOOP_HEAD':
            continue

        lam_est, lam_unc, strategy = estimate_loop_bound(
            node_attrs=attrs,
            parsed_args=parsed_args,
            col_stats=col_stats,
            db_name=db_name,
            udf_name=udf_name,
            udf_source=udf_source,
            value_length=value_length,
        )

        nx.set_node_attributes(graph, {node_id: {
            'lambda_est':      lam_est,
            'lambda_unc':      lam_unc,
            'bound_strategy':  strategy,
        }})
        loop_head_data.append((lam_est, lam_unc))

    # Propagate aggregate signals to INVOCATION node
    if loop_head_data:
        unc_vals  = [u for _, u in loop_head_data]
        est_vals  = [e for e, _ in loop_head_data]
        max_unc   = max(unc_vals)
        max_lam   = max(est_vals)
        has_unc   = float(any(u > 0.0 for u in unc_vals))

        for node_id, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'INVOCATION':
                in_rows_est = float(attrs.get('in_rows_est', 1) or 1)
                log_total_ops_est = math.log(max(1.0, in_rows_est)) + max_lam
                nx.set_node_attributes(graph, {node_id: {
                    'has_uncertain_loop':  has_unc,
                    'max_lambda_unc':      max_unc,
                    'log_total_ops_est':   log_total_ops_est,
                }})
                break

    return graph


def annotate_graph_batch(
    items: List[Tuple[nx.DiGraph, str, str]],
    col_stats: dict,
    db_name: str,
    known_tables: Optional[List[str]] = None,
) -> List[nx.DiGraph]:
    """
    Annotate a batch of graphs.

    Args:
        items:  List of (graph, sql_query, udf_name) tuples.
        col_stats, db_name, known_tables: same as annotate_graph.

    Returns:
        List of annotated graphs (same objects, modified in-place).
    """
    results = []
    for graph, sql_query, udf_name in items:
        annotate_graph(
            graph=graph,
            sql_query=sql_query,
            udf_name=udf_name,
            col_stats=col_stats,
            db_name=db_name,
            known_tables=known_tables,
        )
        results.append(graph)
    return results


def get_annotation_summary(graph: nx.DiGraph) -> dict:
    """
    Return a human-readable summary of the lambda annotations on this graph.
    Useful for debugging and evaluation.
    """
    loops = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('type') == 'LOOP_HEAD':
            loops.append({
                'node_id':       node_id,
                'loop_type':     attrs.get('loop_type'),
                'fixed_iter':    attrs.get('fixed_iter'),
                'no_iter':       attrs.get('no_iter'),
                'lambda_est':    attrs.get('lambda_est'),
                'lambda_unc':    attrs.get('lambda_unc'),
                'bound_strategy': attrs.get('bound_strategy'),
                'est_no_iter':   round(math.exp(attrs['lambda_est']), 2)
                                 if attrs.get('lambda_est') is not None else None,
            })
    inv = {}
    for _, attrs in graph.nodes(data=True):
        if attrs.get('type') == 'INVOCATION':
            inv = {
                'has_uncertain_loop':  attrs.get('has_uncertain_loop'),
                'max_lambda_unc':      attrs.get('max_lambda_unc'),
                'log_total_ops_est':   attrs.get('log_total_ops_est'),
            }
            break
    return {'loop_heads': loops, 'invocation': inv}


# Internal helper
def _infer_tables(col_stats: dict, db_name: str) -> List[str]:
    """Extract table names from col_stats keys (format: 'table.col')."""
    db_stats = col_stats.get(db_name, {})
    tables = set()
    for key in db_stats:
        table, _, _ = key.partition('.')
        if table:
            tables.add(table)
    return list(tables)
