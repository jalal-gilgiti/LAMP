"""
Rewrite static loops of GRACEFUL benchmark UDFs into the loop patterns of the paper's
taxonomy (Table 1), so that trip counts become input dependent.

Benchmark loops have the forms
    for iK in range(N):                  or      iK = 0
                                                 while iK < N:
                                                     iK += 1
Every rewrite keeps the loop body unchanged; only the trip count changes:

  T1  static            unchanged (control)
  T2  query-level       bound = new parameter n_dyn, passed as an SQL literal
  T3  nested            outer loop of a nested pair gets the T4 bound
  T4  per-tuple         bound = max(1, min(int(abs(x)), CAP)), x a numeric argument
  T5  collection        loop over the characters of str(x), x any argument
  T6  convergence       Newton iteration for sqrt(|x| + 2) until |y - y_prev| <= 1e-6
  T7  sequential        first of two sequential loops gets the T4 bound
  T8  branch-guarded    single loop wrapped in `if x > THRESHOLD:`

Dynamic bounds are computed into a variable before the loop (`n_dyn = ...`), so the loop
header carries no numeric literal: create_graph.py marks any header containing a number as a
static loop with that number as its trip count.
"""
import re
from typing import Dict, List, Optional, Tuple

DEF_RE = re.compile(r'^def (\w+)\((.*)\)\s*->\s*(\w+)\s*:\s*$')
FOR_RE = re.compile(r'^(\t+)for (\w+) in range\((\d+)\):\s*$')
WHILE_RE = re.compile(r'^(\t+)while (\w+) < (\d+):\s*$')
NUMERIC_TYPES = {'int', 'float'}


def parse_def(line: str) -> Tuple[str, List[Tuple[str, str]], str]:
    m = DEF_RE.match(line.strip())
    assert m, f'unexpected def line: {line!r}'
    params = []
    for p in [p for p in m.group(2).split(',') if p.strip()]:
        name, typ = p.split(':')
        params.append((name.strip(), typ.strip()))
    return m.group(1), params, m.group(3)


def find_loops(lines: List[str]) -> List[Dict]:
    """Loop headers with their line index, indentation depth, kind and literal bound."""
    loops = []
    for i, ln in enumerate(lines):
        m = FOR_RE.match(ln) or WHILE_RE.match(ln)
        if m:
            loops.append(dict(idx=i, depth=len(m.group(1)), kind='for' if ln.lstrip().startswith('for') else 'while',
                              var=m.group(2), bound=int(m.group(3))))
    return loops


def block_end(lines: List[str], start: int) -> int:
    """Index one past the last line of the block opened at lines[start]."""
    depth = len(lines[start]) - len(lines[start].lstrip('\t'))
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and len(ln) - len(ln.lstrip('\t')) <= depth:
            break
        end += 1
    return end


def loop_structure(lines: List[str]) -> Optional[str]:
    """'single', 'nested' or 'sequential' (same classification as the evaluation), None if no loop."""
    loops = find_loops(lines)
    if not loops:
        return None
    # nested: a loop header inside another loop's block (a loop inside an `if` is not nested)
    for outer in loops:
        end = block_end(lines, outer['idx'])
        if any(outer['idx'] < inner['idx'] < end for inner in loops):
            return 'nested'
    return 'single' if len(loops) == 1 else 'sequential'


def _header(loop: Dict, bound_expr: str) -> str:
    tabs = '\t' * loop['depth']
    if loop['kind'] == 'for':
        return f'{tabs}for {loop["var"]} in range({bound_expr}):'
    return f'{tabs}while {loop["var"]} < {bound_expr}:'


def _numeric_param(params) -> Optional[str]:
    for name, typ in params:
        if typ in NUMERIC_TYPES:
            return name
    return None


def rewrite(lines: List[str], pattern: str, *, literal: Optional[int] = None,
            threshold: Optional[float] = None) -> Optional[Dict]:
    """
    Return dict(lines, param, extra_arg, bound) for the rewritten UDF, or None if the UDF
    does not fit the pattern. `param` is the UDF parameter that drives the trip count
    (None for T1/T2), `extra_arg` the SQL literal to append to the call (T2 only).
    """
    lines = [ln.rstrip('\n') for ln in lines]
    name, params, ret = parse_def(lines[0])
    loops = find_loops(lines)
    structure = loop_structure(lines)
    out = list(lines)

    if pattern == 'T1':
        return dict(lines=out, param=None, extra_arg=None) if structure == 'single' else None

    if pattern == 'T2':
        if structure != 'single' or literal is None:
            return None
        loop = loops[0]
        out[0] = f'def {name}({",".join(f"{p}:{t}" for p, t in params)},n_dyn:int) -> {ret}:'
        out[loop['idx']] = _header(loop, 'n_dyn')
        return dict(lines=out, param='n_dyn', extra_arg=int(literal))

    if pattern in ('T3', 'T4', 'T7'):
        wanted = {'T3': 'nested', 'T4': 'single', 'T7': 'sequential'}[pattern]
        x = _numeric_param(params)
        if structure != wanted or x is None:
            return None
        loop = loops[0]  # T3: outermost loop, T7: first of the sequential loops
        cap = 3 * loop['bound']
        tabs = '\t' * loop['depth']
        out[loop['idx']] = _header(loop, 'n_dyn')
        out.insert(loop['idx'], f'{tabs}n_dyn = max(1, min(int(abs({x})), {cap}))')
        return dict(lines=out, param=x, extra_arg=None, cap=cap)

    if pattern == 'T5':
        if structure != 'single' or not params:
            return None
        loop = loops[0]
        x = params[0][0]
        tabs = '\t' * loop['depth']
        out[loop['idx']] = f'{tabs}for ch_dyn in str({x}):'
        return dict(lines=out, param=x, extra_arg=None)

    if pattern == 'T6':
        x = _numeric_param(params)
        if structure != 'single' or x is None:
            return None
        loop = loops[0]
        tabs = '\t' * loop['depth']
        body_tabs = tabs + '\t'
        header = f'{tabs}while abs(y_dyn - y_prev) > 1e-6:'
        prelude = [f'{tabs}t_dyn = abs(float({x})) + 2.0', f'{tabs}y_dyn = t_dyn', f'{tabs}y_prev = 0.0']
        step = [f'{body_tabs}y_prev = y_dyn', f'{body_tabs}y_dyn = 0.5 * (y_dyn + t_dyn / y_dyn)']
        out = out[:loop['idx']] + prelude + [header] + step + out[loop['idx'] + 1:]
        return dict(lines=out, param=x, extra_arg=None)

    if pattern == 'T8':
        x = _numeric_param(params)
        if structure != 'single' or x is None or threshold is None:
            return None
        loop = loops[0]
        start = loop['idx']
        if loop['kind'] == 'while':  # keep the counter initialisation with the guarded loop
            init_re = re.compile(rf'^\t+{re.escape(loop["var"])} = 0\s*$')
            if start > 0 and init_re.match(out[start - 1]):
                start -= 1
        end = block_end(out, loop['idx'])
        tabs = '\t' * loop['depth']
        guarded = ['\t' + ln if ln.strip() else ln for ln in out[start:end]]
        thr = repr(float(threshold))
        out = out[:start] + [f'{tabs}if {x} > {thr}:'] + guarded + out[end:]
        return dict(lines=out, param=x, extra_arg=None, threshold=float(threshold))

    raise ValueError(pattern)


def add_sql_literal(sql: str, udf_name: str, literal: int) -> str:
    """Append a literal argument to the (single) call of udf_name in sql."""
    pat = re.compile(r'\b' + re.escape(udf_name) + r'\(([^()]*)\)')
    m = pat.search(sql)
    assert m, f'{udf_name} call not found'
    return sql[:m.start()] + f'{udf_name}({m.group(1)}, {literal})' + sql[m.end():]
