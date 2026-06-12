import pandas as pd

df = pd.read_csv("cleaned_quikr.csv")

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nStatistics:")
print(df.describe())
print(df['company'].value_counts())
print(df['fuel_type'].value_counts())
import matplotlib.pyplot as plt

plt.hist(df['Price'], bins=30)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")

plt.show()
