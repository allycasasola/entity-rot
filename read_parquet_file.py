import pandas as pd
import duckdb

PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"

# Use DuckDB to efficiently read just the first few rows without loading entire file
conn = duckdb.connect()

print("=" * 80)
print("SCHEMA:")
print("=" * 80)
schema_df = conn.execute(f"DESCRIBE SELECT * FROM '{PATH_TO_PARQUET_FILE}'").df()
print(schema_df)
print()

print("=" * 80)
print("TOTAL ROW COUNT:")
print("=" * 80)
count = conn.execute(
    f"SELECT COUNT(*) as total FROM '{PATH_TO_PARQUET_FILE}'"
).fetchone()[0]
print(f"Total rows in dataset: {count:,}")
print()

print("=" * 80)
print("FIRST 5 ROWS WITH FULL CONTENT:")
print("=" * 80)

# Query just the first 5 rows efficiently with DuckDB
df = conn.execute(f"SELECT * FROM '{PATH_TO_PARQUET_FILE}' LIMIT 5").df()

# Display each row with full content
for idx, row in df.iterrows():
    print(f"\n{'=' * 80}")
    print(f"ROW {idx}")
    print("=" * 80)
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  {row[col]}")
    print()

conn.close()
