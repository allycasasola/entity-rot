import pandas as pd
import duckdb
PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"

# Query directly from Parquet
duckdb.query(f"SELECT * FROM '{PATH_TO_PARQUET_FILE}' LIMIT 10").show()

# Check schema
duckdb.query(f"DESCRIBE SELECT * FROM '{PATH_TO_PARQUET_FILE}'").show()