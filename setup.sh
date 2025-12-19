#!/bin/bash

# Setup script for Customer Churn Prediction on MacBook Air M2
# This script sets up the Conda environment and installs dependencies

echo "================================================"
echo "Customer Churn Prediction - Setup for M2 Mac"
echo "Using Conda Environment"
echo "================================================"
echo ""

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed."
    echo ""
    echo "Please install Miniconda or Anaconda first:"
    echo "Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html"
    echo "Anaconda: https://www.anaconda.com/products/distribution"
    echo ""
    echo "For M2 Mac, download the ARM64 (Apple Silicon) version."
    echo "Quick install command:"
    echo "  brew install --cask miniconda"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "churn_prediction"; then
    echo "⚠️  Environment 'churn_prediction' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n churn_prediction -y
    else
        echo "❌ Setup cancelled. Using existing environment."
        exit 0
    fi
fi

# Create conda environment
echo "📦 Creating conda environment from environment.yml..."
echo "   This may take several minutes..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Environment creation failed. Trying alternative method..."
    echo "📦 Creating conda environment with pip..."
    conda create -n churn_prediction python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate churn_prediction
    pip install -r requirements.txt
fi

echo ""
echo "================================================"
echo "✅ Setup completed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Activate the conda environment: conda activate churn_prediction"
echo "2. Download the Telco Customer Churn dataset and place in data/ folder"
echo "3. Run the app: python app.py"
echo ""
echo "To deactivate: conda deactivate"
echo "To remove environment: conda env remove -n churn_prediction"
echo ""
echo "Note: The models need to be trained first before predictions can be made."
echo "See train_model.py for details on model training."
echo ""
