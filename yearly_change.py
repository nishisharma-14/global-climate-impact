import pandas as pd


def yearly_temperature_change(csv_path: str = 'global_warming_dataset.csv'):
    df = pd.read_csv(csv_path)
    yearly = df.groupby('Year')['Temperature_Anomaly'].mean().sort_index()
    yoy = yearly.diff()
    result = pd.DataFrame({'Year': yearly.index, 'Temp_Anomaly': yearly.values, 'YoY_Change': yoy.values})
    return result


if __name__ == '__main__':
    df = yearly_temperature_change()
    print(df.tail(10))
