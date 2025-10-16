import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ===================================================================
# Step 1: Data Loading, Merging, and Cleaning (External Data)
# ===================================================================
print("--- Step 1: Loading, Merging, and Cleaning External Data ---")
try:
    # Load Core Data and aggregate to daily sales
    df_train = pd.read_csv('train.csv')
    df_train['date'] = pd.to_datetime(df_train['date'])
    sales_df = df_train.groupby('date')['sales'].sum().reset_index()

    # Load External Data
    df_holidays = pd.read_csv('holidays_events.csv')
    df_oil = pd.read_csv('oil.csv')
    
except FileNotFoundError:
    print("❌ ERROR: Ensure 'train.csv', 'holidays_events.csv', and 'oil.csv' are in the same folder.")
    exit()

# A. Prepare Oil Data (Daily Oil Price)
df_oil['date'] = pd.to_datetime(df_oil['date'])
# Use forward-fill (ffill) to use the last known price for missing days (weekends/holidays)
df_oil['dcoilwtico'] = df_oil['dcoilwtico'].ffill() 
df_oil.rename(columns={'dcoilwtico': 'oil_price'}, inplace=True)

# B. Prepare Holiday Data (Create a simple binary holiday flag)
df_holidays['date'] = pd.to_datetime(df_holidays['date'])
# Filter for non-transfer holidays (which tend to have the biggest impact)
holiday_dates = df_holidays[df_holidays['transferred'] == False]['date'].unique()
holiday_dates_df = pd.DataFrame(holiday_dates, columns=['date'])
holiday_dates_df['is_holiday'] = 1 

# Merge All DataFrames
sales_df = sales_df.merge(df_oil[['date', 'oil_price']], on='date', how='left')
sales_df = sales_df.merge(holiday_dates_df, on='date', how='left')

# Fill missing holiday values with 0 (meaning 'not a holiday')
sales_df['is_holiday'] = sales_df['is_holiday'].fillna(0)

# --- CRITICAL FIX: Ensure no NaNs remain in oil_price before training ---
# 1. Forward-fill: Fills NaNs using the LAST known price (covers gaps/weekends).
sales_df['oil_price'].fillna(method='ffill', inplace=True) 

# 2. Backward-fill: Fills any remaining NaNs at the beginning of the series
#    (which ffill fails to cover) using the FIRST valid known price.
sales_df['oil_price'].fillna(method='bfill', inplace=True) 
# --- END CRITICAL FIX ---


print("✅ Data Merging Complete. External features added.")
print("-" * 50)


# ===================================================================
# Step 2: Feature Engineering (Internal & External)
# ===================================================================
print("--- Step 2: Feature Engineering and Data Prep ---")

# Time Step Feature (for Polynomial Trend)
sales_df['time_step'] = np.arange(len(sales_df)) 

# Seasonality Features
sales_df['month'] = sales_df['date'].dt.month
sales_df['dayofweek'] = sales_df['date'].dt.dayofweek # 0=Mon, 6=Sun

# One-Hot Encode Seasonality
sales_df = pd.get_dummies(sales_df, columns=['month', 'dayofweek'], drop_first=True)

# Define Final Features (X) and Target (y)
EXTERNAL_FEATURES = ['oil_price', 'is_holiday']
SEASONAL_FEATURES = [col for col in sales_df.columns if col.startswith('month_') or col.startswith('dayofweek_')]

X_cols = ['time_step'] + EXTERNAL_FEATURES + SEASONAL_FEATURES
X = sales_df[X_cols]
y = sales_df['sales']

print(f"Total Features (X): {len(X_cols)}")
print(f"Example Features: {X_cols[:5] + X_cols[-2:]}")
print("-" * 50)


# ===================================================================
# Step 3: Model Training and Evaluation
# ===================================================================
print("--- Step 3: Model Training and Evaluation ---")

# Split Data: Use the first 80% for training, last 20% for testing
split_point = int(len(sales_df) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Implement Polynomial Transformation (Degree 3)
poly_transformer = PolynomialFeatures(degree=3, include_bias=False)

# 1. Transform ONLY the 'time_step' column
time_step_train = X_train[['time_step']]
time_step_test = X_test[['time_step']]

X_train_poly_ts = poly_transformer.fit_transform(time_step_train)
X_test_poly_ts = poly_transformer.transform(time_step_test)

# 2. Recombine: Create new DataFrames with polynomial features + all other features
def create_final_features(X_original, X_poly_ts):
    X_final = pd.DataFrame(X_poly_ts).set_index(X_original.index)
    X_final.columns = [f'ts_poly_{i}' for i in range(1, X_final.shape[1] + 1)]
    # Concatenate polynomial features with the remaining non-time-step features
    return pd.concat([X_final, X_original.drop(columns=['time_step'])], axis=1)

X_train_final = create_final_features(X_train, X_train_poly_ts)
X_test_final = create_final_features(X_test, X_test_poly_ts)
X_full_final = create_final_features(X, poly_transformer.transform(X[['time_step']])) # For plotting

# 3. Train the Regression Model
model = LinearRegression()
model.fit(X_train_final, y_train)

# 4. Make Predictions and Evaluate
y_pred_test = model.predict(X_test_final)

# Calculate Root Mean Squared Error (RMSE) on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Test Set RMSE (External Features): ${rmse:,.2f}")
print("-" * 50)

# 5. Visualization
y_pred_full = model.predict(X_full_final)

plt.figure(figsize=(14, 7))
plt.plot(sales_df['date'], sales_df['sales'], label='Actual Sales', alpha=0.5, linewidth=1, color='blue')
plt.plot(sales_df['date'], y_pred_full, label='Regression Forecast (External Data + Trend + Seasonality)', color='red', linewidth=2)
plt.axvline(x=sales_df['date'].iloc[split_point], color='black', linestyle='--', label='Train/Test Split')
plt.title('Advanced Sales Forecast using External Data', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("🚀 Project Complete: The model is trained and its fit is visualized!")
