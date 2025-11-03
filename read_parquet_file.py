import pandas as pd

PATH_TO_PARQUET_FILE = "/oak/stanford/groups/deho/dbateyko/municipal_codes/data/output/municode_sections.parquet"

def main():
    df = pd.read_parquet(PATH_TO_PARQUET_FILE)
    print(df.head())
    # Explore the data
    print(df.columns)
    print(df.head())
    print(df.tail())

    print(df.shape)
    print(df.dtypes)

if __name__ == "__main__":
    main()