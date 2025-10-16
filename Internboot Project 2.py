import pandas as pd
import numpy as np # Importing numpy is good practice for data manipulation

# --- STEP 1: Data Preparation ---
try:
    df_raw = pd.read_csv('train.csv')
except FileNotFoundError:
    print("❌ ERROR: 'train.csv' not found. Ensure the file is in the same folder.")
    exit()

# Re-aggregate total sales across all stores for each day
df_raw['date'] = pd.to_datetime(df_raw['date'])
sales_df = df_raw.groupby('date')['sales'].sum().reset_index()

print("Initial Sales Data Snapshot (pre-Feature Engineering):")
print(sales_df.head())
print("-" * 50)


# --- STEP 2: Extract Time Features ---
print("Extracting Time-Based Features...")

sales_df['year'] = sales_df['date'].dt.year
sales_df['month'] = sales_df['date'].dt.month
sales_df['day'] = sales_df['date'].dt.day
sales_df['dayofweek'] = sales_df['date'].dt.dayofweek # Monday=0, Sunday=6

# Create a binary flag for weekends
# The apply(lambda) function checks if the dayofweek (x) is 5 (Saturday) or 6 (Sunday)
sales_df['is_weekend'] = sales_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)


# --- STEP 3: Implement Lag Features ---
print("Implementing Lag Features...")

# Sales from 7 days ago (sales_t-7)
sales_df['sales_lag_7'] = sales_df['sales'].shift(7)

# Sales from 30 days ago (sales_t-30)
sales_df['sales_lag_30'] = sales_df['sales'].shift(30)


# --- STEP 4: Handle Missing Values ---
# Lag features introduced NaN values in the first 30 rows
initial_rows = sales_df.shape[0]
sales_df.dropna(inplace=True)
final_rows = sales_df.shape[0]

print(f"Dropped {initial_rows - final_rows} rows with NaN values (the first 30 days).")
print("-" * 50)

# Final output
print("✅ Feature Engineering Complete! New Data Snapshot:")
print(sales_df[['date', 'sales', 'year', 'month', 'is_weekend', 'sales_lag_7']].head())
print(f"The new DataFrame has {sales_df.shape[0]} rows and {sales_df.shape[1]} columns.")