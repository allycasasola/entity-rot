import pandas as pd
import duckdb
import csv

PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/allyc/city_ordinances.parquet"
OUTPUT_CSV = "cities_and_jurisdictions.csv"

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
print("TOTAL WORD TOKENS:")
print("=" * 80)
total_tokens = conn.execute(
    f"SELECT SUM(word_tokens_approx) as total FROM '{PATH_TO_PARQUET_FILE}'"
).fetchone()[0]
print(f"Total word_tokens_approx: {total_tokens:,}")
print("=" * 80)

print("=" * 80)
print("EXTRACTING CITIES AND JURISDICTIONS:")
print("=" * 80)

# Get all unique combinations of city_slug and jurisdiction_name
query = f"""
    SELECT DISTINCT city_slug, jurisdiction_name, state, state_code
    FROM '{PATH_TO_PARQUET_FILE}' 
    ORDER BY city_slug, jurisdiction_name, state, state_code
"""
results_df = conn.execute(query).df()

# Save to CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved {len(results_df)} entries to {OUTPUT_CSV}")
print()

# Also print summary
print("Sample entries:")
print(results_df.head(10).to_string(index=False))
print()
if len(results_df) > 10:
    print(f"... and {len(results_df) - 10} more entries")
    print()


# print("FIRST 5 ROWS WITH FULL CONTENT:")
# print("=" * 80)


# # Query just the first 5 rows efficiently with DuckDB
# df = conn.execute(f"SELECT * FROM '{PATH_TO_PARQUET_FILE}' LIMIT 5").df()

# # Display each row with full content
# for idx, row in df.iterrows():
#     print(f"\n{'=' * 80}")
#     print(f"ROW {idx}")
#     print("=" * 80)
#     for col in df.columns:
#         print(f"\n{col}:")
#         print(f"  {row[col]}")
#     print()

conn.close()
