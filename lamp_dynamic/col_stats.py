"""
Extract per-column statistics from DuckDB databases for dynamic lambda estimation.

Usage (one-time, per database):
    python -m lamp_dynamic.col_stats \
        --db_path /path/to/financial.db \
        --db_name financial \
        --out_dir  $GRACEFUL_DATA/lamp_dynamic/col_stats

Output JSON structure:
    {
      "financial": {
        "district.A7": {"mean": 7.3, "std": 2.9, "p25": 5.0, "p50": 7.0, "p75": 10.0, "max": 13.0},
        "district.A9": { ... },
        ...
      }
    }
"""

import json
import os
import argparse
import duckdb


_NUMERIC_TYPES = {
    'INTEGER', 'BIGINT', 'SMALLINT', 'TINYINT', 'HUGEINT',
    'FLOAT', 'DOUBLE', 'DECIMAL', 'REAL',
    'INT', 'INT4', 'INT8', 'INT2', 'INT1',
}


def _is_numeric(col_type: str) -> bool:
    return col_type.upper().split('(')[0].strip() in _NUMERIC_TYPES


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def extract_col_stats(db_path: str, tables: list = None) -> dict:
    """
    Extract numeric column statistics from every table in db_path.

    Args:
        db_path: Path to the DuckDB .db file.
        tables:  Optional list of table names. If None, all tables are scanned.

    Returns:
        dict  {"table.column": {"mean", "std", "p25", "p50", "p75", "max"}}
    """
    conn = duckdb.connect(db_path, read_only=True)

    if tables is None:
        tables = [
            r[0] for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        ]

    result = {}
    for table in tables:
        try:
            # Use fetchdf() to access columns by name - robust across DuckDB versions
            df = conn.execute(f"SUMMARIZE {table}").fetchdf()
        except Exception as exc:
            print(f"  [WARN] SUMMARIZE {table} failed: {exc}")
            continue

        for _, row in df.iterrows():
            col_name = str(row['column_name'])
            col_type = str(row['column_type'])
            if not _is_numeric(col_type):
                continue

            key = f"{table}.{col_name}"
            result[key] = {
                'mean': _safe_float(row.get('avg')),
                'std':  _safe_float(row.get('std')),
                'p25':  _safe_float(row.get('q25')),
                'p50':  _safe_float(row.get('q50')),
                'p75':  _safe_float(row.get('q75')),
                'max':  _safe_float(row.get('max')),
                'min':  _safe_float(row.get('min')),
            }

    conn.close()
    return result


def save_col_stats(stats: dict, db_name: str, out_dir: str) -> str:
    """
    Wrap stats under db_name key and save to {out_dir}/{db_name}_col_stats.json.
    Returns the saved file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {db_name: stats}
    out_path = os.path.join(out_dir, f"{db_name}_col_stats.json")
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_col_stats(col_stats_dir: str) -> dict:
    """
    Load all *_col_stats.json files from col_stats_dir into one merged dict.
    Returns  {"db_name": {"table.col": {...}, ...}, ...}
    """
    merged = {}
    if not os.path.isdir(col_stats_dir):
        return merged
    for fname in os.listdir(col_stats_dir):
        if not fname.endswith('_col_stats.json'):
            continue
        path = os.path.join(col_stats_dir, fname)
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
    return merged


def get_col_stat(col_stats: dict, db_name: str, table: str, column: str) -> dict:
    """
    Convenience accessor. Returns the stat dict, or empty dict if not found.
    """
    return col_stats.get(db_name, {}).get(f"{table}.{column}", {})


# CLI
def _cli():
    parser = argparse.ArgumentParser(
        description='Extract column statistics from a DuckDB database.'
    )
    parser.add_argument('--db_path',  required=True, help='Path to .db file')
    parser.add_argument('--db_name',  required=True, help='Logical database name (e.g. financial)')
    parser.add_argument('--out_dir',  required=True, help='Output directory for JSON file')
    parser.add_argument('--tables',   nargs='*',     help='Table names to scan (default: all)')
    args = parser.parse_args()

    print(f"Extracting stats from {args.db_path} ...")
    stats = extract_col_stats(args.db_path, tables=args.tables)
    out_path = save_col_stats(stats, args.db_name, args.out_dir)
    print(f"Saved {len(stats)} column entries -> {out_path}")

    # Print a summary of high-cardinality numeric columns
    top = sorted(stats.items(), key=lambda kv: kv[1].get('max', 0), reverse=True)[:10]
    print("\nTop-10 columns by max value:")
    for key, s in top:
        print(f"  {key:<40}  mean={s['mean']:8.1f}  p50={s['p50']:8.1f}  max={s['max']:8.1f}  std={s['std']:6.1f}")


if __name__ == '__main__':
    _cli()
