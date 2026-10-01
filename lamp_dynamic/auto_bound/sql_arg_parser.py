"""
Parse SQL call arguments to determine whether each argument is:
  - a numeric literal  (e.g. my_func(50, col_a))
  - a column reference (e.g. my_func(col_a, district.A7))

Handles the two column reference formats found in analytical SQL:
  Format A: table.col          -> "district.A7"
  Format B: alias.table_col    -> "nested_query.district_A7"  (benchmark subquery alias)

Public API
----------
parse_call_args(sql, func_name, known_tables=None)
    Returns list[dict], one per argument:
        {'type': 'literal',  'value': float, 'pos': int}
        {'type': 'col_ref',  'table': str, 'col': str, 'pos': int, 'raw': str}
        {'type': 'expr',     'raw': str, 'pos': int}

get_arg(parsed_args, pos)
    Returns the parsed argument dict at position pos, or None.
"""

import re
from typing import List, Optional


# Internal helpers
def _extract_call_args_str(sql: str, func_name: str) -> Optional[str]:
    """
    Return the raw argument string inside the first call to func_name in sql.
    E.g. "SELECT f(a, b, 7) FROM t" -> "a, b, 7"
    Returns None if func_name not found.
    """
    pattern = re.compile(
        r'\b' + re.escape(func_name) + r'\s*\(([^)]+)\)',
        re.IGNORECASE,
    )
    m = pattern.search(sql)
    return m.group(1) if m else None


def _try_literal(token: str) -> Optional[float]:
    """Return float value if token is a bare positive numeric literal, else None."""
    t = token.strip()
    try:
        val = float(t)
        return val if val > 0 else None
    except ValueError:
        return None


def _try_col_ref(token: str, known_tables: Optional[List[str]]) -> Optional[dict]:
    """
    Try to interpret token as a column reference.

    Handles:
      Format A: "district.A7"           -> table=district, col=A7
      Format B: "nested_query.district_A7" -> table=district, col=A7
                (alias.{table}_{col} where table is in known_tables)
      Format C: "A7"                    -> table=None, col=A7 (if no dot)

    Returns {'table': str|None, 'col': str} or None.
    """
    t = token.strip()
    # Format D: DuckDB-quoted identifiers -> "table"."col"  or  "col"
    # Strip double-quote wrappers before processing
    t = re.sub(r'"([A-Za-z_][A-Za-z0-9_]*)"', r'\1', t)
    # Must look like an identifier (letters/digits/underscore, optionally with one dot)
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_.]*$', t):
        return None

    parts = t.split('.')
    if len(parts) == 1:
        # No dot - bare column name
        return {'table': None, 'col': parts[0]}

    if len(parts) == 2:
        qualifier, name = parts[0], parts[1]

        # Format A: qualifier IS a table name
        if known_tables and qualifier.lower() in [tb.lower() for tb in known_tables]:
            return {'table': qualifier, 'col': name}

        # Format B: qualifier is an alias, name is {table}_{col}
        if known_tables:
            for tb in sorted(known_tables, key=len, reverse=True):
                prefix = tb + '_'
                if name.lower().startswith(prefix.lower()):
                    col = name[len(prefix):]
                    return {'table': tb, 'col': col}

        # Format A fallback: treat qualifier as table even if not in known_tables
        # (e.g. when schema is not provided)
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', qualifier):
            return {'table': qualifier, 'col': name}

    return None


# Public API
def parse_call_args(
    sql: str,
    func_name: str,
    known_tables: Optional[List[str]] = None,
) -> List[dict]:
    """
    Parse arguments of a SQL UDF call.

    Args:
        sql:          Full SQL query string.
        func_name:    Name of the UDF function to find (e.g. 'dyn_func_285').
        known_tables: List of table names in the DB schema. Helps resolve
                      benchmark-style aliases like 'nested_query.district_A7'.

    Returns:
        List of dicts, one per argument, in call order:
          {'type': 'literal',  'value': float, 'pos': int}
          {'type': 'col_ref',  'table': str|None, 'col': str, 'pos': int, 'raw': str}
          {'type': 'expr',     'raw': str, 'pos': int}
    """
    args_str = _extract_call_args_str(sql, func_name)
    if args_str is None:
        return []

    raw_args = [a.strip() for a in args_str.split(',')]
    result = []
    for i, raw in enumerate(raw_args):
        lit = _try_literal(raw)
        if lit is not None:
            result.append({'type': 'literal', 'value': lit, 'pos': i})
            continue

        col = _try_col_ref(raw, known_tables)
        if col is not None:
            result.append({'type': 'col_ref', 'table': col['table'],
                           'col': col['col'], 'pos': i, 'raw': raw})
            continue

        result.append({'type': 'expr', 'raw': raw, 'pos': i})

    return result


def get_arg(parsed_args: List[dict], pos: int) -> Optional[dict]:
    """Return the parsed argument at position pos, or None."""
    for a in parsed_args:
        if a['pos'] == pos:
            return a
    return None
