import pandas as pd
import numpy as np
import joblib # Library for saving models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

print("Starting Model Training and Saving...")

# --- 1. Load and Prepare Data (Same as Intermediate Task 3) ---
df_raw = pd.read_csv('train.csv')
df_raw['date'] = pd.to_datetime(df_raw['date'])
sales_df = df_raw.groupby('date')['sales'].sum().reset_index()

# Create required features: Time Step (for trend) and Seasonality (Month/DayofWeek)
sales_df['time_step'] = np.arange(len(sales_df))
sales_df['month'] = sales_df['date'].dt.month
sales_df['dayofweek'] = sales_df['date'].dt.dayofweek
sales_df = pd.get_dummies(sales_df, columns=['month', 'dayofweek'], drop_first=True)

# Define Features and Target
X_cols = ['time_step'] + [col for col in sales_df.columns if col.startswith('month_') or col.startswith('dayofweek_')]
X = sales_df[X_cols]
y = sales_df['sales']

# --- 2. Train the Final Model ---
# Use a simple degree=2 polynomial for demonstration
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_transformer.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# --- 3. Save Model and Transformer ---
# Save the trained model and the polynomial transformer object
joblib.dump(model, 'model.pkl')
joblib.dump(poly_transformer, 'poly_transformer.pkl')

print("✅ Model (model.pkl) and Transformer (poly_transformer.pkl) successfully saved.")
