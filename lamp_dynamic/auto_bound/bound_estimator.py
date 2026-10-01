"""
Per-LOOP_HEAD automatic bound estimator.

Given a LOOP_HEAD node's attributes, the parsed SQL arguments, and column
statistics, determines the best available estimate of log(iteration_count)
and associated uncertainty.

Priority chain
--------------
1. fixed_iter=True       -> exact, lambda_unc = 0
2. Numeric literal arg   -> exact value from SQL call, lambda_unc = 0
3. Column reference arg  -> p50 from DB catalog, lambda_unc = std/p50,
                           clipped to C when the UDF source bounds the trip-count
                           variable with min(..., C)
   Collection loop        -> `for x in str(p)` / `for x in p` over a parameter p that maps
                           to a column: median value length of that column
4. Conservative fallback -> log(DEFAULT_ITER), lambda_unc = 1.0
   (convergence while-loops have no derivable bound and always reach this tier)

All rules read only planner-visible information: the SQL call, catalog statistics and
the UDF source code (udf_source; without it the source-based rules are skipped).

For each LOOP_HEAD the caller passes:
  - node_attrs       : node attribute dict from the graph
  - parsed_args      : output of sql_arg_parser.parse_call_args()
  - col_stats        : {db_name: {'table.col': {mean, std, p50, ...}}}
  - db_name          : which DB's stats to use

Returns
-------
(lambda_est: float, lambda_unc: float, strategy: str)
  lambda_est  - log(estimated_no_iter)
  lambda_unc  - normalized uncertainty (0 = exact, >0 = uncertain)
  strategy    - human-readable label for which tier fired
"""

import math
import re
from typing import Callable, Optional

from .sql_arg_parser import get_arg

# Defaults when nothing better is available
_DEFAULT_ITER    = 10.0   # conservative: log(10) ~ 2.3
_DEFAULT_UNC     = 1.0    # 100% coefficient of variation -> high uncertainty


def _source_lines(udf_source: Optional[str]) -> list:
    return udf_source.split('\n') if udf_source else []


def _param_names(lines: list) -> list:
    m = re.match(r'\s*def\s+\w+\s*\((.*)\)', lines[0]) if lines else None
    return [p.split(':')[0].split('=')[0].strip() for p in m.group(1).split(',') if p.strip()] if m else []


def _bound_variable(header: str) -> Optional[str]:
    """Trip-count variable of a loop header: range(n) / range(a, n) / while i < n."""
    m = re.search(r'range\(\s*(?:[^,()]+,\s*)?(\w+)\s*\)', header)
    if m:
        return m.group(1)
    m = re.search(r'while\s+\w+\s*<=?\s*(\w+)\s*:', header)
    return m.group(1) if m else None


def _source_cap(lines: list, var: Optional[str]) -> Optional[float]:
    """C if the UDF assigns the trip-count variable with min(..., C), else None."""
    if not var or var.isdigit():
        return None
    for ln in lines:
        m = re.match(rf'\s*{re.escape(var)}\s*=\s*(.+)$', ln)
        if m and 'min(' in m.group(1):
            caps = re.findall(r',\s*(\d+)\s*\)', m.group(1)[m.group(1).index('min('):])
            if caps:
                return float(caps[0])
    return None


def _collection_param(header: str, params: list) -> Optional[int]:
    """Position of the parameter a for-each loop iterates over (`for x in str(p)` / `for x in p`)."""
    m = re.search(r'for\s+\w+\s+in\s+(?:str\(\s*)?(\w+)\s*\)?\s*:', header)
    return params.index(m.group(1)) if m and m.group(1) in params else None


def estimate_loop_bound(
    node_attrs: dict,
    parsed_args: list,
    col_stats: dict,
    db_name: str,
    udf_name: str = '',
    udf_source: Optional[str] = None,
    value_length: Optional[Callable[[str, str], Optional[float]]] = None,
) -> tuple:
    """
    Estimate (lambda_est, lambda_unc, strategy) for a single LOOP_HEAD node.

    Args:
        node_attrs:   Attribute dict of the LOOP_HEAD node.
        parsed_args:  Result of sql_arg_parser.parse_call_args() for this query.
        col_stats:    Column statistics dict (from col_stats.load_col_stats()).
        db_name:      Target database name.
        udf_name:     UDF function name.

    Returns:
        (lambda_est, lambda_unc, strategy)
    """
    # Tier 1: Static loop - exact
    fixed = str(node_attrs.get('fixed_iter', 'True')).strip().lower()
    if fixed == 'true':
        no_iter = max(1.0, float(node_attrs.get('no_iter', 1)))
        return math.log(no_iter), 0.0, 'static_exact'

    lt = str(node_attrs.get('loop_type', 'for')).lower()
    influencing = node_attrs.get('influencing_params', [])

    db_col_stats = col_stats.get(db_name, {}) if db_name else {}
    lines = _source_lines(udf_source)
    lineno = node_attrs.get('lineno')
    header = lines[lineno - 1] if lines and isinstance(lineno, int) and 0 < lineno <= len(lines) else ''

    # Collection loop over a parameter's value: median value length of its column
    if lt == 'cursor' and header and value_length is not None:
        pos = _collection_param(header, _param_names(lines))
        arg = get_arg(parsed_args, pos) if pos is not None else None
        if arg is not None and arg['type'] == 'col_ref' and arg.get('table'):
            length = value_length(arg['table'], arg['col'])
            if length:
                return math.log(max(1.0, length)), _DEFAULT_UNC * 0.5, 'col_len_p50'

    for param_idx in influencing:
        arg = get_arg(parsed_args, param_idx)
        if arg is None:
            continue

        # Tier 2: Literal argument in SQL call
        if arg['type'] == 'literal':
            val = max(1.0, arg['value'])
            return math.log(val), 0.0, 'sql_literal'

        # Tier 3: Column reference -> fetch catalog stats
        if arg['type'] == 'col_ref':
            table = arg.get('table')
            col   = arg.get('col')

            stat = _lookup_stat(db_col_stats, table, col)
            if stat:
                p50 = max(1.0, float(stat.get('p50', _DEFAULT_ITER)))
                std = float(stat.get('std', 0.0))
                cap = _source_cap(lines, _bound_variable(header)) if header else None
                if cap is not None:
                    return math.log(min(p50, max(1.0, cap))), std / p50, 'col_stat_p50_capped'
                return math.log(p50), std / p50, 'col_stat_p50'

            # Column found but no stats: treat as unknown
            return math.log(_DEFAULT_ITER), _DEFAULT_UNC, 'col_ref_no_stat'

    # Tier 4: conservative fallback
    return math.log(_DEFAULT_ITER), _DEFAULT_UNC, 'fallback'


# Internal helper
def _lookup_stat(db_col_stats: dict, table: Optional[str], col: str) -> Optional[dict]:
    """
    Look up column statistics. Tries:
      1. '{table}.{col}'  (direct key)
      2. Any key ending in '.{col}' (when table is None or unresolved)
    """
    if table:
        stat = db_col_stats.get(f"{table}.{col}")
        if stat:
            return stat

    # Fuzzy fallback: match by column name alone
    col_lower = col.lower()
    for key, stat in db_col_stats.items():
        _, _, key_col = key.partition('.')
        if key_col.lower() == col_lower:
            return stat

    return None
