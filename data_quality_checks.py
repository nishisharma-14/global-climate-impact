import pandas as pd


def run_data_quality_checks(csv_path: str = "global_warming_dataset.csv") -> None:
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    if "Year" in df.columns:
        print(f"Year range: {df['Year'].min()} to {df['Year'].max()}")

    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")

    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    print("\nColumns with missing values:")
    if missing_values.empty:
        print("None")
    else:
        for col, count in missing_values.items():
            pct = (count / len(df)) * 100
            print(f"- {col}: {count} ({pct:.2f}%)")

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print("\nNumeric summary (min / max / mean):")
        summary = df[numeric_cols].agg(["min", "max", "mean"]).round(3)
        print(summary)


if __name__ == "__main__":
    run_data_quality_checks()
