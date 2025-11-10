#!/bin/bash

# Loan Approval Prediction - Complete Pipeline Execution Script
# This script runs the entire ML pipeline from data generation to visualization

echo "============================================================"
echo "LOAN APPROVAL PREDICTION - COMPLETE PIPELINE"
echo "============================================================"
echo ""

# Set JAVA_HOME for Spark
export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto

# Check if Java is installed
if [ ! -d "$JAVA_HOME" ]; then
    echo "❌ Java not found. Installing Java 11..."
    sudo dnf install -y java-11-amazon-corretto-headless
fi

echo "✓ Java configured: $JAVA_HOME"
echo ""

# Step 1: Generate Data
echo "============================================================"
echo "STEP 1: GENERATING LOAN DATA"
echo "============================================================"
python3 src/data_generator.py
if [ $? -ne 0 ]; then
    echo "❌ Data generation failed!"
    exit 1
fi
echo ""

# Step 2: Train Model
echo "============================================================"
echo "STEP 2: TRAINING MODEL & MAKING PREDICTIONS"
echo "============================================================"
python3 src/loan_prediction.py
if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi
echo ""

# Step 3: Generate Visualizations
echo "============================================================"
echo "STEP 3: GENERATING VISUALIZATIONS"
echo "============================================================"
python3 src/visualize_results.py
if [ $? -ne 0 ]; then
    echo "❌ Visualization generation failed!"
    exit 1
fi
echo ""

# Summary
echo "============================================================"
echo "PIPELINE EXECUTION COMPLETED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "📁 Output Files:"
echo "  Data:"
echo "    - data/loan_data.csv"
echo ""
echo "  Model Results:"
echo "    - output/predictions.csv"
echo "    - output/model_metrics.csv"
echo ""
echo "  Visualizations:"
echo "    - output/predicted_vs_actual_bar.png"
echo "    - output/confusion_matrix.png"
echo "    - output/roc_curve.png"
echo "    - output/metrics_summary.png"
echo "    - output/approval_distribution.png"
echo ""
echo "✓ All files generated successfully!"
echo "============================================================"
