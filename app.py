import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
import time
from flask import Flask, request, render_template
import base64
import io
import os
import webbrowser
import threading
import logging

app = Flask(__name__)

# Load models and preprocessing artifacts
model = None
survmodel = None
explainer = None
scaler = None
label_encoders = {}
feature_names = []

# Load models and preprocessing artifacts
if os.path.exists('model.pkl'):
    model = pickle.load(open('model.pkl', 'rb'))
if os.path.exists('survivemodel.pkl'):
    survmodel = pickle.load(open('survivemodel.pkl', 'rb'))
if os.path.exists('explainer.bz2'):
    explainer = joblib.load(filename="explainer.bz2")
if os.path.exists('preprocessing_artifacts.pkl'):
    artifacts = pickle.load(open('preprocessing_artifacts.pkl', 'rb'))
    scaler = artifacts.get('scaler')
    label_encoders = artifacts.get('label_encoders', {})
    feature_names = artifacts.get('feature_names', [])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             prediction_text='Models not loaded. Please train the models first.')
    
    try:
        # Map form input to model's expected format (label encoded)
        # Gender: Male=0, Female=1
        gender_val = 'Female' if request.form.get("gender") == "1" else 'Male'
        
        # Senior Citizen: 0 or 1
        senior_citizen = 1 if 'SeniorCitizen' in request.form else 0
        
        # Partner: Yes=1, No=0
        partner_val = 'Yes' if 'Partner' in request.form else 'No'
        
        # Dependents: Yes=1, No=0
        dependents_val = 'Yes' if 'Dependents' in request.form else 'No'
        
        # Phone Service
        phone_service_val = 'Yes' if 'PhoneService' in request.form else 'No'
        
        # Multiple Lines
        if phone_service_val == 'No':
            multiple_lines_val = 'No phone service'
        else:
            multiple_lines_val = 'Yes' if 'MultipleLines' in request.form else 'No'
        
        # Internet Service
        internet_service_map = {'0': 'No', '1': 'DSL', '2': 'Fiber optic'}
        internet_service_val = internet_service_map.get(request.form.get("InternetService", "0"), 'No')
        
        # Online services
        if internet_service_val == 'No':
            online_security_val = 'No internet service'
            online_backup_val = 'No internet service'
            device_protection_val = 'No internet service'
            tech_support_val = 'No internet service'
            streaming_tv_val = 'No internet service'
            streaming_movies_val = 'No internet service'
        else:
            online_security_val = 'Yes' if 'OnlineSecurity' in request.form else 'No'
            online_backup_val = 'Yes' if 'OnlineBackup' in request.form else 'No'
            device_protection_val = 'Yes' if 'DeviceProtection' in request.form else 'No'
            tech_support_val = 'Yes' if 'TechSupport' in request.form else 'No'
            streaming_tv_val = 'Yes' if 'StreamingTV' in request.form else 'No'
            streaming_movies_val = 'Yes' if 'StreamingMovies' in request.form else 'No'
        
        # Paperless Billing
        paperless_billing_val = 'Yes' if 'PaperlessBilling' in request.form else 'No'
        
        # Contract
        contract_map = {'0': 'Month-to-month', '1': 'One year', '2': 'Two year'}
        contract_val = contract_map.get(request.form.get("Contract", "0"), 'Month-to-month')
        
        # Payment Method
        payment_map = {
            '0': 'Bank transfer (automatic)',
            '1': 'Credit card (automatic)',
            '2': 'Electronic check',
            '3': 'Mailed check'
        }
        payment_method_val = payment_map.get(request.form.get("PaymentMethod", "0"), 'Electronic check')
        
        # Numerical values
        monthly_charges = float(request.form["MonthlyCharges"])
        tenure = int(request.form["Tenure"])
        total_charges = monthly_charges * tenure
        
        # Create dataframe with raw values
        input_data = {
            'gender': [gender_val],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner_val],
            'Dependents': [dependents_val],
            'tenure': [tenure],
            'PhoneService': [phone_service_val],
            'MultipleLines': [multiple_lines_val],
            'InternetService': [internet_service_val],
            'OnlineSecurity': [online_security_val],
            'OnlineBackup': [online_backup_val],
            'DeviceProtection': [device_protection_val],
            'TechSupport': [tech_support_val],
            'StreamingTV': [streaming_tv_val],
            'StreamingMovies': [streaming_movies_val],
            'Contract': [contract_val],
            'PaperlessBilling': [paperless_billing_val],
            'PaymentMethod': [payment_method_val],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Apply label encoding
        df_encoded = df_input.copy()
        for col in df_input.columns:
            if col in label_encoders:
                df_encoded[col] = label_encoders[col].transform(df_input[col])
        
        # Apply scaling to numerical features
        if scaler is not None:
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
        
        # Ensure column order matches training
        final_features = df_encoded[feature_names]
        
        prediction = model.predict_proba(final_features)
        output = prediction[0, 1]
        
        # Generate gauge chart
        gauge_url = generate_gauge(output)
        
        # Generate SHAP plot if explainer exists
        shap_url = ""
        if explainer is not None:
            shap_url = generate_shap_plot(final_features, df_input.iloc[0])
        
        # Generate survival plots if survival model exists
        hazard_url = ""
        surv_url = ""
        CLTV = 0
        if survmodel is not None:
            hazard_url, surv_url, CLTV = generate_survival_plots(
                df_input, tenure, monthly_charges
            )
        
        return render_template('index.html', 
                             prediction_text=f'Churn Probability: {round(output*100, 1)}% | Expected Lifetime Value: ${round(CLTV, 2)}',
                             url_1=gauge_url, url_2=shap_url, 
                             url_3=hazard_url, url_4=surv_url)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

def generate_gauge(probability):
    """Generate gauge chart for churn probability"""
    try:
        gauge_img = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Simple gauge visualization
        colors = ['#007A00', '#0063BF', '#FFCC00', '#ED1C24']
        labels = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
        
        # Create semicircle
        for i, (color, label) in enumerate(zip(colors, labels)):
            start_angle = 180 - (i * 45)
            end_angle = 180 - ((i + 1) * 45)
            wedge = Wedge((0, 0), 0.4, end_angle, start_angle, 
                         width=0.1, facecolor=color, alpha=0.5, lw=2)
            ax.add_patch(wedge)
        
        # Add arrow
        pos = (1 - probability) * 180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 
                0.225 * np.sin(np.radians(pos)),
                width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
        
        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.1, 0.5)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(gauge_img, format='png', bbox_inches='tight')
        plt.close()
        gauge_img.seek(0)
        return base64.b64encode(gauge_img.getvalue()).decode()
    except:
        return ""

def generate_shap_plot(final_features, feature_display):
    """Generate SHAP values plot"""
    try:
        if explainer is None:
            return ""
        shap_img = io.BytesIO()
        shap_values = explainer.shap_values(final_features)
        
        # Use the first row of data
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], 
                       feature_display, matplotlib=True, show=False).savefig(
                       shap_img, bbox_inches="tight", format='png', dpi=150)
        shap_img.seek(0)
        return base64.b64encode(shap_img.getvalue()).decode()
    except Exception as e:
        print(f"SHAP Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def generate_survival_plots(df_input, tenure, monthly_charges):
    """Generate hazard and survival curves"""
    try:
        if survmodel is None:
            return "", "", 0
        
        # Prepare data for Cox model (uses one-hot encoding, not label encoding)
        surv_df = df_input.copy()
        
        # Convert to binary for certain features
        binary_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for var in binary_vars:
            if var in surv_df.columns:
                surv_df[var] = (surv_df[var] == surv_df[var].unique()[0]).astype(int)
        
        # Convert SeniorCitizen to string for dummy encoding
        surv_df['SeniorCitizen'] = surv_df['SeniorCitizen'].astype(str)
        
        # One-hot encoding for categorical features (drop_first=True to match training)
        categorical_features = ['SeniorCitizen', 'MultipleLines', 'InternetService', 
                               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                               'TechSupport', 'StreamingTV', 'StreamingMovies', 
                               'Contract', 'PaymentMethod']
        surv_df = pd.get_dummies(surv_df, columns=categorical_features, drop_first=True)
        
        # Get all feature columns that Cox model expects
        expected_features = survmodel.params_.index.tolist()
        
        # Add missing columns with 0 values
        for col in expected_features:
            if col not in surv_df.columns:
                surv_df[col] = 0
        
        # Select only the features Cox model expects in the correct order
        surv_df = surv_df[expected_features]
        
        # Hazard plot
        hazard_img = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 4))
        survmodel.predict_cumulative_hazard(surv_df).plot(ax=ax, color='red')
        plt.axvline(x=tenure, color='blue', linestyle='--')
        plt.legend(labels=['Hazard', 'Current Position'])
        ax.set_xlabel('Tenure', size=10)
        ax.set_ylabel('Cumulative Hazard', size=10)
        ax.set_title('Cumulative Hazard Over Time')
        plt.tight_layout()
        plt.savefig(hazard_img, format='png')
        plt.close()
        hazard_img.seek(0)
        hazard_url = base64.b64encode(hazard_img.getvalue()).decode()
        
        # Survival plot
        surv_img = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 4))
        survmodel.predict_survival_function(surv_df).plot(ax=ax, color='red')
        plt.axvline(x=tenure, color='blue', linestyle='--')
        plt.legend(labels=['Survival Function', 'Current Position'])
        ax.set_xlabel('Tenure', size=10)
        ax.set_ylabel('Survival Probability', size=10)
        ax.set_title('Survival Probability Over Time')
        plt.tight_layout()
        plt.savefig(surv_img, format='png')
        plt.close()
        surv_img.seek(0)
        surv_url = base64.b64encode(surv_img.getvalue()).decode()
        
        # Calculate CLTV
        life = survmodel.predict_survival_function(surv_df).reset_index()
        life.columns = ['Tenure', 'Probability']
        max_life = life.Tenure[life.Probability > 0.1].max()
        CLTV = max_life * monthly_charges
        
        return hazard_url, surv_url, CLTV
    except Exception as e:
        print(f"Survival plot error: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", "", 0

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5001')

if __name__ == "__main__":
    # Suppress werkzeug development server warning
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n" + "="*60)
    print("🚀 Customer Churn Prediction App")
    print("="*60)
    print("Opening browser at http://127.0.0.1:5001")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
