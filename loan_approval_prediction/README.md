# Loan Approval Prediction using Apache Spark

A complete machine learning pipeline built with Apache Spark MLlib to predict loan approval decisions based on applicant demographics, financial data, and credit history.

## 📋 Project Overview

This project implements an end-to-end ML solution that:
- Generates synthetic loan application data with realistic patterns
- Handles missing values and inconsistent data
- Encodes categorical variables for ML processing
- Trains a Logistic Regression model using Spark MLlib
- Evaluates model performance with multiple metrics
- Visualizes results with comprehensive charts

## 🏗️ Project Structure

```
loan_approval_prediction/
├── data/
│   └── loan_data.csv              # Generated loan application dataset
├── src/
│   ├── data_generator.py          # Synthetic data generation script
│   ├── loan_prediction.py         # Main Spark ML pipeline
│   └── visualize_results.py       # Visualization script
├── output/
│   ├── predictions.csv            # Model predictions
│   ├── model_metrics.csv          # Performance metrics
│   ├── predicted_vs_actual_bar.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_summary.png
│   └── approval_distribution.png
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Features

### Data Generation
- **1000 loan applications** with realistic patterns
- **13 features** including demographics, income, loan details, and credit history
- **Missing values** (2-10% across different columns) to simulate real-world data
- **Correlated features** (e.g., education level affects income)

### Data Preprocessing
- **Missing value imputation**: Mean for numeric, mode for categorical
- **Categorical encoding**: StringIndexer + OneHotEncoder
- **Feature engineering**: Vector assembly of all features
- **Data validation**: Ensures no missing values after preprocessing

### Model Training
- **Algorithm**: Logistic Regression (Spark MLlib)
- **Train/Test Split**: 80/20
- **Hyperparameters**: 
  - Max iterations: 100
  - Regularization: 0.01
  - Elastic Net: 0.8

### Model Evaluation
- **Accuracy**: 88.27%
- **Precision**: 0.8960
- **Recall**: 0.8827
- **F1-Score**: 0.8882
- **ROC-AUC**: 0.8968

### Visualizations
1. **Predicted vs Actual Bar Chart**: Compares approval/rejection counts
2. **Confusion Matrix**: Shows true/false positives and negatives
3. **ROC Curve**: Displays model discrimination ability
4. **Metrics Summary**: Bar chart of all performance metrics
5. **Approval Distribution**: Pie charts comparing actual vs predicted distributions

## 📊 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| Loan_ID | String | Unique loan application ID |
| Gender | Categorical | Male/Female |
| Married | Categorical | Married/Single |
| Dependents | Numeric | Number of dependents (0-3) |
| Education | Categorical | Graduate/Not Graduate |
| Self_Employed | Categorical | Yes/No |
| ApplicantIncome | Numeric | Primary applicant's income |
| CoapplicantIncome | Numeric | Co-applicant's income |
| LoanAmount | Numeric | Requested loan amount |
| Loan_Amount_Term | Numeric | Loan term in months |
| Credit_History | Binary | 1 = Good, 0 = Bad |
| Property_Area | Categorical | Urban/Semiurban/Rural |
| Loan_Status | Binary | Y = Approved, N = Rejected |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Java 11+ (required for Apache Spark)
- pip package manager

### Install Dependencies

```bash
# Install Java (Amazon Linux 2023)
sudo dnf install -y java-11-amazon-corretto-headless

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto

# Install Python packages
pip install -r requirements.txt
```

### Dependencies
- pyspark==3.5.0
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0

## 🎯 Usage

### 1. Generate Data
```bash
python3 src/data_generator.py
```
**Output**: `data/loan_data.csv` (1000 loan applications)

### 2. Train Model & Make Predictions
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto
python3 src/loan_prediction.py
```
**Output**: 
- `output/predictions.csv` (model predictions)
- `output/model_metrics.csv` (performance metrics)

### 3. Generate Visualizations
```bash
python3 src/visualize_results.py
```
**Output**: 5 visualization PNG files in `output/` directory

### Run Complete Pipeline
```bash
# Generate data, train model, and create visualizations
export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto
python3 src/data_generator.py && \
python3 src/loan_prediction.py && \
python3 src/visualize_results.py
```

## 📈 Results

### Model Performance
- **Accuracy**: 88.27% - The model correctly predicts loan approval 88% of the time
- **ROC-AUC**: 0.8968 - Excellent discrimination between approved and rejected loans
- **Precision**: 0.8960 - High confidence in positive predictions
- **Recall**: 0.8827 - Successfully identifies most approved loans

### Key Insights
1. **Credit History** is the most important predictor of loan approval
2. **Total Income** (applicant + co-applicant) strongly influences approval
3. **Education Level** correlates with higher approval rates
4. **Loan-to-Income Ratio** affects approval decisions
5. Model performs well with **minimal false positives** (7 out of 162 test cases)

## 🔍 Technical Details

### Spark ML Pipeline Stages
1. **StringIndexer** (5 stages): Convert categorical strings to numeric indices
2. **OneHotEncoder** (5 stages): Create binary vectors for categorical features
3. **VectorAssembler** (1 stage): Combine all features into single vector
4. **LogisticRegression** (1 stage): Train binary classification model

### Data Preprocessing Strategy
- **Numeric Missing Values**: Imputed with column mean
- **Categorical Missing Values**: Imputed with mode (most frequent value)
- **Dependents**: Converted "3+" to numeric 3
- **Target Variable**: Converted Y/N to 1/0

### Model Configuration
```python
LogisticRegression(
    featuresCol="features",
    labelCol="Loan_Status_Numeric",
    maxIter=100,           # Maximum iterations
    regParam=0.01,         # Regularization parameter
    elasticNetParam=0.8    # Elastic Net mixing (L1/L2)
)
```

## 📝 Code Structure

### data_generator.py
- Generates synthetic loan data with realistic patterns
- Introduces missing values strategically
- Creates correlations between features (e.g., education → income)

### loan_prediction.py
- `LoanApprovalPredictor` class with methods:
  - `load_data()`: Load CSV into Spark DataFrame
  - `analyze_missing_values()`: Identify missing data patterns
  - `preprocess_data()`: Handle missing values and type conversions
  - `build_pipeline()`: Create ML pipeline with encoding and model
  - `train_model()`: Split data and train Logistic Regression
  - `evaluate_model()`: Calculate performance metrics
  - `save_predictions()`: Export predictions to CSV

### visualize_results.py
- `ResultVisualizer` class with methods:
  - `plot_predicted_vs_actual_bar()`: Bar chart comparison
  - `plot_confusion_matrix()`: Heatmap of prediction accuracy
  - `plot_roc_curve()`: ROC curve with AUC score
  - `plot_metrics_summary()`: Horizontal bar chart of metrics
  - `plot_approval_distribution()`: Pie charts of distributions

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Apache Spark DataFrame operations
- ✅ Handling missing and inconsistent data
- ✅ Categorical variable encoding techniques
- ✅ ML pipeline construction with Spark MLlib
- ✅ Binary classification with Logistic Regression
- ✅ Model evaluation with multiple metrics
- ✅ Data visualization best practices
- ✅ End-to-end ML workflow implementation

## 🔧 Troubleshooting

### Java Not Found
```bash
# Install Java
sudo dnf install -y java-11-amazon-corretto-headless

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-amazon-corretto
```

### Spark Warnings
Warnings about native Hadoop libraries can be safely ignored. They don't affect functionality.

### Memory Issues
If you encounter memory errors, reduce the dataset size in `data_generator.py`:
```python
df = generate_loan_data(n_samples=500)  # Reduce from 1000
```

## 📊 Sample Output

### Console Output (Model Training)
```
============================================================
MODEL PERFORMANCE METRICS
============================================================
Accuracy:           0.8827 (88.27%)
Precision:          0.8960
Recall:             0.8827
F1-Score:           0.8882
ROC-AUC:            0.8968
============================================================
```

### Confusion Matrix
```
                Predicted
              Rejected  Approved
Actual
Rejected         13        7
Approved         12       130
```

## 🚀 Future Enhancements

- [ ] Add more ML algorithms (Random Forest, Gradient Boosting)
- [ ] Implement hyperparameter tuning with CrossValidator
- [ ] Add feature importance analysis
- [ ] Create interactive dashboard with Plotly
- [ ] Deploy model as REST API
- [ ] Add real-time prediction capability
- [ ] Implement model versioning and tracking

## 📄 License

This project is created for educational purposes.

## 👤 Author

Created as part of Project 31: Loan Approval Prediction

---

**Note**: This project uses synthetic data for demonstration purposes. In production, use real loan data with proper privacy and compliance measures.
