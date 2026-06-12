import pandas as pd

df = pd.read_csv("cleaned_quikr.csv")

df['car_age'] = 2026 - df['year']

print(df[['year','car_age']].head())

df.to_csv(
    "processed_quikr.csv",
    index=False
)
