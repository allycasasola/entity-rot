import pandas as pd
import duckdb

PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"

# Set pandas display options to show full content
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Read first several rows into pandas for better display control
df = pd.read_parquet(PATH_TO_PARQUET_FILE)

print("=" * 80)
print("SCHEMA:")
print("=" * 80)
print(df.dtypes)
print()

print("=" * 80)
print("FIRST 5 ROWS WITH FULL CONTENT:")
print("=" * 80)

# Display first 5 rows with full content
for idx, row in df.head(5).iterrows():
    print(f"\n{'=' * 80}")
    print(f"ROW {idx}")
    print("=" * 80)
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  {row[col]}")
    print()
