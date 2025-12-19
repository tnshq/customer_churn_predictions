#!/bin/bash

# Run script for Customer Churn Prediction Flask App

echo "🚀 Starting Customer Churn Prediction App..."
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "churn_prediction"; then
    echo "❌ Conda environment 'churn_prediction' not found."
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "🔄 Activating conda environment 'churn_prediction'..."
conda activate churn_prediction

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment."
    echo "Try manually: conda activate churn_prediction"
    exit 1
fi

# Check if models exist
if [ ! -f "model.pkl" ]; then
    echo "⚠️  Warning: model.pkl not found."
    echo "The app will run but predictions won't work until models are trained."
    echo "See train_model.py for model training instructions."
    echo ""
fi

# Set Flask environment variables
export FLASK_APP=app.py
export FLASK_ENV=development

# Run the Flask app
echo "✅ Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
python app.py

# Deactivate after exit
echo ""
echo "Deactivating conda environment..."
conda deactivate
