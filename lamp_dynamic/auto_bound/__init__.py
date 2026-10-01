"""
Auto-Bound: planning-time trip-count resolution for UDF loops.

For every LOOP_HEAD of a UDF graph, the loop-controlling UDF parameter is mapped
to its argument at the SQL call site and the trip count is taken from the first
applicable rule:
  1. static bound in the UDF code
  2. collection loop over a column-bound parameter: median value length
  3. numeric literal argument: the literal
  4. column argument: column median (p50), clipped by min(x, C) if present
  5. fallback: 10

Usage:
    from lamp_dynamic.auto_bound import annotate_graph
    annotate_graph(graph, sql_query, udf_name, col_stats, db_name)
"""

from .pipeline import annotate_graph, annotate_graph_batch

__all__ = ['annotate_graph', 'annotate_graph_batch']
