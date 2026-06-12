import pandas as pd

# Load dataset
df = pd.read_csv("quikr_car.csv")

# Remove rows with invalid year values
df = df[df['year'].astype(str).str.isnumeric()]
df['year'] = df['year'].astype(int)

# Remove rows where Price is "Ask For Price"
df = df[df['Price'] != 'Ask For Price']

# Clean Price column
df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Clean kms_driven column
df['kms_driven'] = df['kms_driven'].astype(str)
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '', regex=False)
df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=False)

# Convert to numeric, invalid values become NaN
df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')

# Remove rows with missing values
df = df.dropna()

# Convert columns to integer
df['Price'] = df['Price'].astype(int)
df['kms_driven'] = df['kms_driven'].astype(int)

# Reset index
df = df.reset_index(drop=True)

# Check final dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# Save cleaned dataset
df.to_csv("cleaned_quikr.csv", index=False)

print("\n✅ Data cleaned successfully!")
print("✅ Saved as: cleaned_quikr.csv")
