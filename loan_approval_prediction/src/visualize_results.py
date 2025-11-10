"""
Visualization Script for Loan Approval Prediction Results
Creates bar charts, confusion matrix, and ROC curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class ResultVisualizer:
    """Visualize loan approval prediction results"""
    
    def __init__(self, predictions_path, metrics_path):
        """
        Initialize visualizer with prediction and metrics data
        
        Parameters:
        -----------
        predictions_path : str
            Path to predictions CSV file
        metrics_path : str
            Path to metrics CSV file
        """
        self.predictions = pd.read_csv(predictions_path)
        self.metrics = pd.read_csv(metrics_path)
        self.output_dir = '/vercel/sandbox/loan_approval_prediction/output'
        
        print("✓ Data loaded successfully")
        print(f"  - Predictions: {len(self.predictions)} records")
        print(f"  - Metrics: {len(self.metrics.columns)} metrics")
    
    def plot_predicted_vs_actual_bar(self):
        """Create bar chart comparing predicted vs actual loan approvals"""
        print("\n1. Creating Predicted vs Actual Bar Chart...")
        
        # Count actual and predicted approvals
        actual_approved = (self.predictions['Loan_Status'] == 'Y').sum()
        actual_rejected = (self.predictions['Loan_Status'] == 'N').sum()
        predicted_approved = (self.predictions['prediction'] == 1).sum()
        predicted_rejected = (self.predictions['prediction'] == 0).sum()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Approved', 'Rejected']
        actual_counts = [actual_approved, actual_rejected]
        predicted_counts = [predicted_approved, predicted_rejected]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual_counts, width, label='Actual', 
                       color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, predicted_counts, width, label='Predicted', 
                       color='#3498db', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Loan Status', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Applications', fontsize=12, fontweight='bold')
        ax.set_title('Predicted vs Actual Loan Approval Results', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/predicted_vs_actual_bar.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_path}")
        print(f"   - Actual: Approved={actual_approved}, Rejected={actual_rejected}")
        print(f"   - Predicted: Approved={predicted_approved}, Rejected={predicted_rejected}")
    
    def plot_confusion_matrix(self):
        """Create confusion matrix visualization"""
        print("\n2. Creating Confusion Matrix...")
        
        # Calculate confusion matrix
        y_true = self.predictions['Loan_Status_Numeric']
        y_pred = self.predictions['prediction']
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Count'},
                   square=True, linewidths=2, linecolor='black',
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Loan Approval Prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticklabels(['Rejected (0)', 'Approved (1)'])
        ax.set_yticklabels(['Rejected (0)', 'Approved (1)'])
        
        # Add accuracy text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
                ha='center', transform=ax.transAxes,
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_path}")
        print(f"   - True Negatives: {cm[0,0]}")
        print(f"   - False Positives: {cm[0,1]}")
        print(f"   - False Negatives: {cm[1,0]}")
        print(f"   - True Positives: {cm[1,1]}")
    
    def plot_roc_curve(self):
        """Create ROC curve visualization"""
        print("\n3. Creating ROC Curve...")
        
        # Extract probabilities for positive class
        y_true = self.predictions['Loan_Status_Numeric']
        
        # Parse probability column (it's a string representation of array)
        probabilities = []
        for prob_str in self.predictions['probability']:
            # Extract the second value (probability of class 1)
            prob_values = prob_str.strip('[]').split(',')
            prob_positive = float(prob_values[1])
            probabilities.append(prob_positive)
        
        y_scores = np.array(probabilities)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='#e74c3c', linewidth=3, 
               label=f'ROC Curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', linewidth=2, 
               linestyle='--', label='Random Classifier (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Loan Approval Prediction', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_path}")
        print(f"   - ROC-AUC Score: {roc_auc:.4f}")
    
    def plot_metrics_summary(self):
        """Create metrics summary visualization"""
        print("\n4. Creating Metrics Summary Chart...")
        
        # Extract metrics
        metrics_dict = self.metrics.iloc[0].to_dict()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_names = list(metrics_dict.keys())
        metrics_values = list(metrics_dict.values())
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.barh(metrics_names, metrics_values, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, metrics_values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}',
                   va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Summary', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, 1.1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/metrics_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_path}")
        for name, value in metrics_dict.items():
            print(f"   - {name}: {value:.4f}")
    
    def plot_approval_distribution(self):
        """Create distribution comparison chart"""
        print("\n5. Creating Approval Distribution Chart...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual distribution
        actual_counts = self.predictions['Loan_Status'].value_counts()
        colors_actual = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0.05)
        
        axes[0].pie(actual_counts.values, labels=['Approved', 'Rejected'], 
                   autopct='%1.1f%%', startangle=90, colors=colors_actual,
                   explode=explode, shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
        axes[0].set_title('Actual Loan Status Distribution', 
                         fontsize=13, fontweight='bold', pad=15)
        
        # Predicted distribution
        predicted_counts = self.predictions['prediction'].value_counts()
        
        axes[1].pie(predicted_counts.values, labels=['Approved', 'Rejected'], 
                   autopct='%1.1f%%', startangle=90, colors=colors_actual,
                   explode=explode, shadow=True, textprops={'fontsize': 12, 'weight': 'bold'})
        axes[1].set_title('Predicted Loan Status Distribution', 
                         fontsize=13, fontweight='bold', pad=15)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/approval_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ Saved to: {output_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.plot_predicted_vs_actual_bar()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_metrics_summary()
        self.plot_approval_distribution()
        
        print("\n" + "="*60)
        print("ALL VISUALIZATIONS COMPLETED!")
        print("="*60)

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("LOAN APPROVAL PREDICTION - VISUALIZATION")
    print("="*60)
    
    # Paths
    predictions_path = '/vercel/sandbox/loan_approval_prediction/output/predictions.csv'
    metrics_path = '/vercel/sandbox/loan_approval_prediction/output/model_metrics.csv'
    
    # Create visualizer
    visualizer = ResultVisualizer(predictions_path, metrics_path)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    print("\n✓ All visualization files saved to: output/")

if __name__ == "__main__":
    main()
