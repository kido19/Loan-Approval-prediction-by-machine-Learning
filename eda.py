import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
if not os.path.exists('eda_plots'):
    os.makedirs('eda_plots')

# -------------------------------------------------------
# 1. Load Real Dataset
# -------------------------------------------------------
DATASET_PATH = r"C:\Users\USER\Downloads\loan_approval_dataset.csv"

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

print(f"Dataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nloan_status distribution:\n{df['loan_status'].value_counts()}")

# Map target for numeric operations
df['loan_status_num'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# -------------------------------------------------------
# 2. EDA Plots
# -------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (9, 6)

# --- Plot 1: Loan Status Distribution ---
plt.figure()
ax = sns.countplot(x='loan_status', data=df, palette=['#2ecc71', '#e74c3c'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12)
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.savefig('eda_plots/loan_status_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: loan_status_distribution.png")

# --- Plot 2: Income Distribution ---
plt.figure()
sns.histplot(df['income_annum'], kde=True, color='steelblue', bins=40)
plt.title('Distribution of Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Frequency')
plt.savefig('eda_plots/income_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: income_distribution.png")

# --- Plot 3: CIBIL Score vs Loan Status ---
plt.figure()
sns.boxplot(x='loan_status', y='cibil_score', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('CIBIL Score vs Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('CIBIL Score')
plt.savefig('eda_plots/cibil_score_vs_approval.png', bbox_inches='tight')
plt.close()
print("Saved: cibil_score_vs_approval.png")

# --- Plot 4: Income vs Loan Status ---
plt.figure()
sns.boxplot(x='loan_status', y='income_annum', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Annual Income vs Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Annual Income')
plt.savefig('eda_plots/income_vs_approval.png', bbox_inches='tight')
plt.close()
print("Saved: income_vs_approval.png")

# --- Plot 5: Education vs Loan Status ---
plt.figure()
sns.countplot(x='education', hue='loan_status', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Education vs Loan Approval')
plt.xlabel('Education')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.savefig('eda_plots/education_vs_approval.png', bbox_inches='tight')
plt.close()
print("Saved: education_vs_approval.png")

# --- Plot 6: Self Employed vs Loan Status ---
plt.figure()
sns.countplot(x='self_employed', hue='loan_status', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Self Employed vs Loan Approval')
plt.xlabel('Self Employed')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.savefig('eda_plots/self_employed_vs_approval.png', bbox_inches='tight')
plt.close()
print("Saved: self_employed_vs_approval.png")

# --- Plot 7: Dependents vs Loan Status ---
plt.figure()
sns.countplot(x='no_of_dependents', hue='loan_status', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Number of Dependents vs Loan Approval')
plt.xlabel('No. of Dependents')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.savefig('eda_plots/dependents_vs_approval.png', bbox_inches='tight')
plt.close()
print("Saved: dependents_vs_approval.png")

# --- Plot 8: Loan Amount vs Income (scatter) ---
plt.figure()
colors = df['loan_status'].map({'Approved': '#2ecc71', 'Rejected': '#e74c3c'})
plt.scatter(df['income_annum'], df['loan_amount'], c=colors, alpha=0.4, edgecolors='none', s=20)
plt.title('Loan Amount vs Annual Income')
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=10, label='Approved'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Rejected')
]
plt.legend(handles=legend_elements)
plt.savefig('eda_plots/loan_amount_vs_income.png', bbox_inches='tight')
plt.close()
print("Saved: loan_amount_vs_income.png")

# --- Plot 9: Correlation Heatmap (numeric only) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(12, 9))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('eda_plots/correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

print("\n✅ All EDA plots saved in 'eda_plots/' directory.")
