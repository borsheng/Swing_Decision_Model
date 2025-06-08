import pandas as pd
from pybaseball import statcast, cache


def load_statcast_data(seasons):
    cache.enable()
    dfs = []
    for start_dt, end_dt in seasons:
        df_season = statcast(start_dt=start_dt, end_dt=end_dt)
        dfs.append(df_season)
    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    seasons = [
        ('2022-03-31', '2022-10-15'),
        ('2023-03-30', '2023-10-01'),
        ('2024-03-28', '2024-10-01')
    ]
    df = load_statcast_data(seasons)
    df.to_csv("data/raw/statcast_2022_2024.csv", index=False)