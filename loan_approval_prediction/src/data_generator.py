"""
Loan Application Data Generator
Generates synthetic loan application data with realistic patterns and missing values
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_loan_data(n_samples=1000):
    """
    Generate synthetic loan application data
    
    Parameters:
    -----------
    n_samples : int
        Number of loan applications to generate
    
    Returns:
    --------
    pd.DataFrame
        Generated loan application data
    """
    
    # Define categorical options
    genders = ['Male', 'Female']
    marital_statuses = ['Married', 'Single']
    education_levels = ['Graduate', 'Not Graduate']
    self_employed = ['Yes', 'No']
    property_areas = ['Urban', 'Semiurban', 'Rural']
    loan_statuses = ['Y', 'N']  # Y = Approved, N = Rejected
    
    data = []
    
    for i in range(n_samples):
        # Generate demographic data
        gender = random.choice(genders)
        married = random.choice(marital_statuses)
        dependents = random.choice([0, 1, 2, 3])
        education = random.choice(education_levels)
        self_emp = random.choice(self_employed)
        
        # Generate financial data with realistic patterns
        # Higher education and married tend to have higher income
        base_income = np.random.normal(5000, 2500)
        if education == 'Graduate':
            base_income *= 1.3
        if married == 'Married':
            base_income *= 1.2
        
        applicant_income = max(1000, int(base_income))
        
        # Co-applicant income (higher if married)
        if married == 'Married':
            coapplicant_income = max(0, int(np.random.normal(3000, 1500)))
        else:
            coapplicant_income = max(0, int(np.random.normal(500, 800)))
        
        # Loan amount (correlated with income)
        total_income = applicant_income + coapplicant_income
        loan_amount = max(50, int(np.random.normal(total_income * 2.5, total_income * 0.8)))
        
        # Loan term (in months)
        loan_term = random.choice([360, 180, 120, 240, 300])
        
        # Credit history (1 = good, 0 = bad)
        # Higher income and education increase probability of good credit
        credit_prob = 0.7
        if education == 'Graduate':
            credit_prob += 0.1
        if total_income > 8000:
            credit_prob += 0.1
        
        credit_history = 1 if random.random() < credit_prob else 0
        
        # Property area
        property_area = random.choice(property_areas)
        
        # Loan status (approval decision)
        # Factors: credit history (most important), income, education, loan amount
        approval_prob = 0.3
        
        if credit_history == 1:
            approval_prob += 0.5
        if education == 'Graduate':
            approval_prob += 0.1
        if total_income > 6000:
            approval_prob += 0.15
        if loan_amount < total_income * 3:
            approval_prob += 0.1
        if property_area == 'Urban':
            approval_prob += 0.05
        
        loan_status = 'Y' if random.random() < approval_prob else 'N'
        
        # Create record
        record = {
            'Loan_ID': f'LP{str(i+1).zfill(6)}',
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_emp,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area,
            'Loan_Status': loan_status
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values (realistic patterns)
    # Missing values more likely in certain fields
    missing_patterns = {
        'Gender': 0.02,
        'Married': 0.01,
        'Dependents': 0.02,
        'Self_Employed': 0.05,
        'LoanAmount': 0.08,
        'Loan_Amount_Term': 0.05,
        'Credit_History': 0.10
    }
    
    for column, missing_rate in missing_patterns.items():
        missing_indices = np.random.choice(
            df.index, 
            size=int(len(df) * missing_rate), 
            replace=False
        )
        df.loc[missing_indices, column] = np.nan
    
    return df

def main():
    """Generate and save loan application data"""
    print("Generating loan application data...")
    
    # Generate data
    df = generate_loan_data(n_samples=1000)
    
    # Save to CSV
    output_path = '/vercel/sandbox/loan_approval_prediction/data/loan_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} loan applications")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset Info:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Features: {len(df.columns)}")
    print(f"  - Approved loans: {(df['Loan_Status'] == 'Y').sum()} ({(df['Loan_Status'] == 'Y').sum()/len(df)*100:.1f}%)")
    print(f"  - Rejected loans: {(df['Loan_Status'] == 'N').sum()} ({(df['Loan_Status'] == 'N').sum()/len(df)*100:.1f}%)")
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"  - {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    
    print(f"\nSample Data:")
    print(df.head())

if __name__ == "__main__":
    main()
