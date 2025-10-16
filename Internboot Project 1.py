import pandas as pd
import matplotlib.pyplot as plt

# --- STEP 1: Data Loading ---
# NOTE: Ensure 'train.csv' is in the same directory as this code file
try:
    df = pd.read_csv('train.csv')
    print("✅ Data loaded successfully.")
    print("Initial Data Info:")
    df.info()
except FileNotFoundError:
    print("❌ ERROR: 'train.csv' not found. Make sure the file is in the same folder.")
    exit()

# --- STEP 2: Data Preprocessing and Aggregation ---
# 1. Convert the 'date' column to a proper datetime object
df['date'] = pd.to_datetime(df['date'])

# 2. Aggregate total sales across all stores for each day
# We sum the 'sales' column, grouping by 'date'
daily_sales = df.groupby('date')['sales'].sum().reset_index()

# 3. Set the 'date' column as the DataFrame index for time series analysis
daily_sales.set_index('date', inplace=True)

print("\nPreprocessed Daily Sales Data (Head):")
print(daily_sales.head())

# --- STEP 3: Implement the Moving Average (Forecasting Logic) ---

# Calculate the 7-Day Moving Average (Weekly Trend)
daily_sales['Weekly_MA'] = daily_sales['sales'].rolling(window=7).mean()

# Calculate the 30-Day Moving Average (Monthly Trend)
daily_sales['Monthly_MA'] = daily_sales['sales'].rolling(window=30).mean()

print("\nData with Moving Averages (Tail):")
print(daily_sales.tail())


# --- STEP 4: Visualization and Interpretation ---

plt.figure(figsize=(14, 7))

# Plot the raw sales data (the reality)
plt.plot(daily_sales['sales'], label='Actual Daily Sales', alpha=0.5, linewidth=1)

# Plot the 7-Day Moving Average (short-term trend/forecast)
plt.plot(daily_sales['Weekly_MA'].dropna(), label='7-Day Moving Average (Trend)', color='red', linewidth=2)

# Plot the 30-Day Moving Average (long-term trend/forecast)
plt.plot(daily_sales['Monthly_MA'].dropna(), label='30-Day Moving Average (Trend)', color='green', linewidth=2)

# Add titles and labels
plt.title('Store Sales Trend and Forecast using Moving Averages', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout() # Adjusts plot to prevent labels from overlapping
plt.show()

print("\n✅ Project Complete: Your sales trend lines have been generated!")