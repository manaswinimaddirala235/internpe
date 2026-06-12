import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv("cleaned_quikr.csv")

print("Dataset Shape:", df.shape)

# =====================================
# FEATURE ENGINEERING
# =====================================

# Create car age feature
CURRENT_YEAR = 2026
df["car_age"] = CURRENT_YEAR - df["year"]

# Shorten car name
df["name"] = df["name"].apply(
    lambda x: " ".join(str(x).split()[:3])
)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop(columns=["Price"])
y = df["Price"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# PREPROCESSING
# =====================================

categorical_columns = [
    "name",
    "company",
    "fuel_type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# =====================================
# LINEAR REGRESSION
# =====================================

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)

print("\n===== LINEAR REGRESSION =====")
print("R2 Score :", round(lr_r2, 4))
print("MAE      :", round(lr_mae, 2))

# =====================================
# RANDOM FOREST
# =====================================

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

print("\n===== RANDOM FOREST =====")
print("R2 Score :", round(rf_r2, 4))
print("MAE      :", round(rf_mae, 2))

# =====================================
# SELECT BEST MODEL
# =====================================

if rf_r2 > lr_r2:
    best_model = rf_model
    best_name = "Random Forest"
    best_score = rf_r2
else:
    best_model = lr_model
    best_name = "Linear Regression"
    best_score = lr_r2

print("\n==========================")
print("BEST MODEL :", best_name)
print("BEST R2    :", round(best_score, 4))
print("==========================")

# =====================================
# SAVE MODEL
# =====================================

pickle.dump(
    best_model,
    open("car_price_model.pkl", "wb")
)

print("\nModel saved as car_price_model.pkl")

# =====================================
# SAMPLE PREDICTION
# =====================================

sample_car = pd.DataFrame({
    "name": ["Hyundai Santro Xing"],
    "company": ["Hyundai"],
    "year": [2015],
    "kms_driven": [45000],
    "fuel_type": ["Petrol"],
    "car_age": [CURRENT_YEAR - 2015]
})

predicted_price = best_model.predict(sample_car)

print(
    "\nPredicted Price: ₹{:,.0f}".format(
        predicted_price[0]
    )
)
