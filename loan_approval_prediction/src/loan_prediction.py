"""
Loan Approval Prediction using Apache Spark MLlib
Complete ML pipeline with data preprocessing, model training, and evaluation
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, isnan, count
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np

class LoanApprovalPredictor:
    """Loan Approval Prediction Pipeline using Spark ML"""
    
    def __init__(self, data_path):
        """
        Initialize Spark session and load data
        
        Parameters:
        -----------
        data_path : str
            Path to the loan data CSV file
        """
        # Create Spark session
        self.spark = SparkSession.builder \
            .appName("LoanApprovalPrediction") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "4") \
            .getOrCreate()
        
        # Suppress INFO logs
        self.spark.sparkContext.setLogLevel("WARN")
        
        print("✓ Spark session initialized")
        
        # Load data
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None
        
    def load_data(self):
        """Load data into Spark DataFrame"""
        print("\n" + "="*60)
        print("STEP 1: LOADING DATA")
        print("="*60)
        
        self.df = self.spark.read.csv(
            self.data_path,
            header=True,
            inferSchema=True
        )
        
        print(f"✓ Loaded {self.df.count()} records")
        print(f"✓ Features: {len(self.df.columns)}")
        
        # Show schema
        print("\nData Schema:")
        self.df.printSchema()
        
        # Show sample data
        print("\nSample Data:")
        self.df.show(5, truncate=False)
        
        return self
    
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        print("\n" + "="*60)
        print("STEP 2: ANALYZING MISSING VALUES")
        print("="*60)
        
        # Calculate missing values for each column
        missing_counts = self.df.select([
            count(when(col(c).isNull(), c)).alias(c) 
            for c in self.df.columns
        ])
        
        print("\nMissing Values Summary:")
        missing_df = missing_counts.toPandas().T
        missing_df.columns = ['Missing_Count']
        missing_df['Missing_Percentage'] = (missing_df['Missing_Count'] / self.df.count() * 100).round(2)
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("No missing values found!")
        
        return self
    
    def preprocess_data(self):
        """Preprocess data: handle missing values and encode categorical variables"""
        print("\n" + "="*60)
        print("STEP 3: DATA PREPROCESSING")
        print("="*60)
        
        # Convert target variable to numeric (Y=1, N=0)
        print("\n1. Converting target variable (Loan_Status)...")
        self.df = self.df.withColumn(
            'Loan_Status_Numeric',
            when(col('Loan_Status') == 'Y', 1).otherwise(0)
        )
        
        # Handle numeric columns with missing values
        print("2. Handling missing values in numeric columns...")
        numeric_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        
        # Impute numeric columns with mean
        imputer = Imputer(
            inputCols=numeric_cols,
            outputCols=[f"{c}_imputed" for c in numeric_cols],
            strategy="mean"
        )
        
        self.df = imputer.fit(self.df).transform(self.df)
        
        # Replace original columns with imputed ones
        for col_name in numeric_cols:
            self.df = self.df.withColumn(col_name, col(f"{col_name}_imputed"))
            self.df = self.df.drop(f"{col_name}_imputed")
        
        # Handle categorical columns with missing values
        print("3. Handling missing values in categorical columns...")
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
        
        for col_name in categorical_cols:
            # Fill with mode (most frequent value)
            mode_value = self.df.groupBy(col_name).count().orderBy(col('count').desc()).first()[0]
            if mode_value is not None:
                self.df = self.df.fillna({col_name: mode_value})
        
        # Convert Dependents to numeric
        self.df = self.df.withColumn(
            'Dependents',
            when(col('Dependents') == '3+', 3).otherwise(col('Dependents').cast(IntegerType()))
        )
        
        # Ensure numeric types
        self.df = self.df.withColumn('ApplicantIncome', col('ApplicantIncome').cast(DoubleType()))
        self.df = self.df.withColumn('CoapplicantIncome', col('CoapplicantIncome').cast(DoubleType()))
        self.df = self.df.withColumn('LoanAmount', col('LoanAmount').cast(DoubleType()))
        self.df = self.df.withColumn('Loan_Amount_Term', col('Loan_Amount_Term').cast(DoubleType()))
        self.df = self.df.withColumn('Credit_History', col('Credit_History').cast(DoubleType()))
        
        print("✓ Missing values handled")
        
        # Verify no missing values remain
        missing_after = self.df.select([
            count(when(col(c).isNull(), c)).alias(c) 
            for c in self.df.columns if c not in ['Loan_ID', 'Loan_Status']
        ]).toPandas()
        
        total_missing = missing_after.sum().sum()
        print(f"✓ Total missing values after preprocessing: {total_missing}")
        
        return self
    
    def build_pipeline(self):
        """Build ML pipeline with feature engineering and model training"""
        print("\n" + "="*60)
        print("STEP 4: BUILDING ML PIPELINE")
        print("="*60)
        
        # Define categorical and numeric features
        categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
        numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                          'Loan_Amount_Term', 'Credit_History', 'Dependents']
        
        print(f"\nCategorical features: {categorical_features}")
        print(f"Numeric features: {numeric_features}")
        
        # String Indexing for categorical variables
        print("\n1. Creating String Indexers...")
        indexers = [
            StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index", handleInvalid="keep")
            for col_name in categorical_features
        ]
        
        # One-Hot Encoding
        print("2. Creating One-Hot Encoders...")
        encoders = [
            OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_encoded")
            for col_name in categorical_features
        ]
        
        # Assemble all features into a single vector
        print("3. Creating Feature Vector Assembler...")
        feature_cols = [f"{col}_encoded" for col in categorical_features] + numeric_features
        
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        # Logistic Regression model
        print("4. Creating Logistic Regression model...")
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="Loan_Status_Numeric",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.8
        )
        
        # Create pipeline
        pipeline_stages = indexers + encoders + [assembler, lr]
        self.pipeline = Pipeline(stages=pipeline_stages)
        
        print(f"✓ Pipeline created with {len(pipeline_stages)} stages")
        
        return self
    
    def train_model(self, train_ratio=0.8):
        """Train the model"""
        print("\n" + "="*60)
        print("STEP 5: TRAINING MODEL")
        print("="*60)
        
        # Split data
        print(f"\nSplitting data (train: {train_ratio*100}%, test: {(1-train_ratio)*100}%)...")
        self.train_data, self.test_data = self.df.randomSplit([train_ratio, 1-train_ratio], seed=42)
        
        print(f"✓ Training set: {self.train_data.count()} records")
        print(f"✓ Test set: {self.test_data.count()} records")
        
        # Train model
        print("\nTraining Logistic Regression model...")
        self.model = self.pipeline.fit(self.train_data)
        print("✓ Model training completed")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("STEP 6: MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        print("\nMaking predictions on test set...")
        self.predictions = self.model.transform(self.test_data)
        
        # Calculate metrics
        print("\nCalculating evaluation metrics...")
        
        # Binary Classification Evaluator (ROC-AUC)
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="Loan_Status_Numeric",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        
        roc_auc = binary_evaluator.evaluate(self.predictions)
        
        # Multiclass Classification Evaluator (Accuracy, Precision, Recall, F1)
        multiclass_evaluator = MulticlassClassificationEvaluator(
            labelCol="Loan_Status_Numeric",
            predictionCol="prediction"
        )
        
        accuracy = multiclass_evaluator.evaluate(self.predictions, {multiclass_evaluator.metricName: "accuracy"})
        precision = multiclass_evaluator.evaluate(self.predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
        recall = multiclass_evaluator.evaluate(self.predictions, {multiclass_evaluator.metricName: "weightedRecall"})
        f1 = multiclass_evaluator.evaluate(self.predictions, {multiclass_evaluator.metricName: "f1"})
        
        # Print results
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:          {precision:.4f}")
        print(f"Recall:             {recall:.4f}")
        print(f"F1-Score:           {f1:.4f}")
        print(f"ROC-AUC:            {roc_auc:.4f}")
        print("="*60)
        
        # Save metrics to file
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('/vercel/sandbox/loan_approval_prediction/output/model_metrics.csv', index=False)
        print("\n✓ Metrics saved to output/model_metrics.csv")
        
        return self
    
    def save_predictions(self):
        """Save predictions for visualization"""
        print("\n" + "="*60)
        print("STEP 7: SAVING PREDICTIONS")
        print("="*60)
        
        # Select relevant columns
        results = self.predictions.select(
            'Loan_ID',
            'Loan_Status',
            'Loan_Status_Numeric',
            'prediction',
            'probability'
        )
        
        # Convert to Pandas and save
        results_pd = results.toPandas()
        results_pd['Predicted_Status'] = results_pd['prediction'].apply(lambda x: 'Y' if x == 1 else 'N')
        
        output_path = '/vercel/sandbox/loan_approval_prediction/output/predictions.csv'
        results_pd.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved to {output_path}")
        print(f"✓ Total predictions: {len(results_pd)}")
        
        # Show sample predictions
        print("\nSample Predictions:")
        print(results_pd[['Loan_ID', 'Loan_Status', 'Predicted_Status']].head(10))
        
        return self
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()
        print("\n✓ Spark session stopped")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("LOAN APPROVAL PREDICTION - SPARK ML PIPELINE")
    print("="*60)
    
    # Initialize predictor
    data_path = '/vercel/sandbox/loan_approval_prediction/data/loan_data.csv'
    predictor = LoanApprovalPredictor(data_path)
    
    # Execute pipeline
    try:
        predictor.load_data() \
                 .analyze_missing_values() \
                 .preprocess_data() \
                 .build_pipeline() \
                 .train_model(train_ratio=0.8) \
                 .evaluate_model() \
                 .save_predictions()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        predictor.stop()

if __name__ == "__main__":
    main()
