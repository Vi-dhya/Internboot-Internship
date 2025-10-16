from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import sys # Import the sys module to allow exiting the application

# --- 1. Initialize Flask App and Load Model ---
app = Flask(__name__)
try:
    # Load the pre-trained model and the feature transformer
    model = joblib.load('model.pkl')
    poly_transformer = joblib.load('poly_transformer.pkl')
    print("Model and Transformer loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Model files not found. Run 'python model_prep.py' first.")
    # CRITICAL FIX: Exit the application if the model files are missing
    sys.exit(1)


# Get the last known time step from the training data (for calculating future time steps)
# This value is based on the size of the original training data.
LAST_TIME_STEP_IN_TRAIN = 1684 


# --- 2. Define Prediction Function ---
def predict_sales(input_date):
    """Takes a date string and returns a predicted sales value."""
    
    # 1. Feature Engineering for the input date
    date_obj = pd.to_datetime(input_date)
    
    # Calculate the new time step (days since start of training data)
    # We use 2013-01-01 as Day 0 for the entire dataset.
    reference_date = pd.to_datetime('2013-01-01')
    new_time_step = (date_obj - reference_date).days
    
    # Create required features for the model
    data = {
        'time_step': new_time_step,
        'month': date_obj.month,
        'dayofweek': date_obj.dayofweek
    }
    
    # Create a DataFrame for the new input
    input_df = pd.DataFrame([data])
    
    # Convert month and dayofweek to one-hot encoding (as done in training)
    input_df = pd.get_dummies(input_df, columns=['month', 'dayofweek'])
    
    # --- 2. Align Columns with Trained Model ---
    # Define the full set of columns the model expects (1 time step + 11 months + 6 days of week = 18 total)
    expected_cols = ['time_step'] + [f'month_{m}' for m in range(2, 13)] + [f'dayofweek_{d}' for d in range(1, 7)]
    
    # Reindex the input_df to ensure all 18 columns exist (filling missing with 0)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    X_predict = input_df[expected_cols]

    # 3. Transform and Predict
    X_predict_poly = poly_transformer.transform(X_predict)
    prediction = model.predict(X_predict_poly)[0]
    
    return prediction


# --- 3. Define Web Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        # Get the date from the form submission
        date_input = request.form['date_input']
        
        try:
            # Get the prediction
            predicted_sales = predict_sales(date_input)
            
            # Format the output text
            prediction_text = f"Predicted Total Sales on {date_input}: ${predicted_sales:,.2f}"
        
        except Exception as e:
            # This handles errors during feature creation or prediction logic
            prediction_text = f"An error occurred during prediction: {e}"

    # Render the index.html template, passing the prediction text
    return render_template('index.html', prediction_result=prediction_text)

# --- 4. Run the Application ---
if __name__ == '__main__':
    # Flask runs on http://127.0.0.1:5000/ by default
    app.run(debug=True)