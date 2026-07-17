"""
Data loading and preparation for the Walmart weekly-sales pipeline.

This mirrors the cleaning / feature-engineering steps from the original
modeling notebook (merge -> impute -> date features -> one-hot encode ->
chronological split) so that training here reproduces the same feature
set the notebook's Random Forest / XGBoost models were built on.
"""

import numpy as np
import pandas as pd

# Final feature column order used for training AND for the serving API's
# request schema. Keep this in sync with src/serve.py.
FEATURE_COLUMNS = [
    "Store",
    "Dept",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Size",
    "Year",
    "Month",
    "Week",
    "Quarter",
    "Day",
    "Type_B",
    "Type_C",
]

TARGET_COLUMN = "Weekly_Sales"


def build_dataset(cfg: dict):
    """Load the three raw CSVs and return (X_train, X_test, y_train, y_test)."""
    train = pd.read_csv(cfg["data"]["train_csv"])
    features = pd.read_csv(cfg["data"]["features_csv"])
    stores = pd.read_csv(cfg["data"]["stores_csv"])

    data = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    data = data.merge(stores, on="Store", how="left")

    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)
    data["Quarter"] = data["Date"].dt.quarter
    data["Day"] = data["Date"].dt.day

    data = pd.get_dummies(data, columns=["Type"], drop_first=True)

    # drop_first=True on "A"/"B"/"C" always yields Type_B / Type_C, but
    # guard against a data sample that happens not to contain every
    # category (e.g. a small smoke-test CSV).
    for col in ("Type_B", "Type_C"):
        if col not in data.columns:
            data[col] = False

    data = data.sort_values("Date").reset_index(drop=True)

    split_idx = int(len(data) * (1 - cfg["data"]["test_size"]))
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test
