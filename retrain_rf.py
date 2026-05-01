import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
df = pd.read_csv(r'C:\Users\USER\Downloads\loan_approval_dataset.csv')
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
df = df.drop(columns=['loan_id'], errors='ignore')

# Feature engineering
df['total_assets']   = (df['residential_assets_value'].fillna(0) +
                        df['commercial_assets_value'].fillna(0) +
                        df['luxury_assets_value'].fillna(0) +
                        df['bank_asset_value'].fillna(0))
df['loan_to_income'] = df['loan_amount'] / (df['income_annum'] + 1)
df['emi_estimate']   = df['loan_amount'] / (df['loan_term'] + 1)

X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

NUM_COLS = ['income_annum','loan_amount','loan_term','cibil_score',
            'residential_assets_value','commercial_assets_value',
            'luxury_assets_value','bank_asset_value',
            'total_assets','loan_to_income','emi_estimate','no_of_dependents']
CAT_COLS = ['education','self_employed']
ALL_COLS = NUM_COLS + CAT_COLS

num_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, NUM_COLS),
    ('cat', cat_pipeline, CAT_COLS)
])

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42))
])

print("Training Random Forest...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}  F1: {f1_score(y_test, y_pred):.4f}")

# Verify edge cases
def fe(row):
    d = pd.DataFrame([row])
    d['total_assets']   = (d['residential_assets_value'] + d['commercial_assets_value'] +
                           d['luxury_assets_value'] + d['bank_asset_value'])
    d['loan_to_income'] = d['loan_amount'] / (d['income_annum'] + 1)
    d['emi_estimate']   = d['loan_amount'] / (d['loan_term'] + 1)
    return d

bad = fe({'education':'Graduate','self_employed':'Yes','no_of_dependents':0.0,
          'loan_term':2.0,'income_annum':200000.0,'loan_amount':9999998.0,
          'cibil_score':300.0,'residential_assets_value':0.0,
          'commercial_assets_value':2000.0,'luxury_assets_value':300000.0,'bank_asset_value':56765.0})
good = fe({'education':'Graduate','self_employed':'No','no_of_dependents':1.0,
           'loan_term':12.0,'income_annum':9000000.0,'loan_amount':10000000.0,
           'cibil_score':800.0,'residential_assets_value':5000000.0,
           'commercial_assets_value':2000000.0,'luxury_assets_value':3000000.0,'bank_asset_value':1000000.0})

for label, df_t in [('BAD  (CIBIL=300, income=200k)', bad),
                    ('GOOD (CIBIL=800, income=9M  )', good)]:
    p  = pipeline.predict(df_t[ALL_COLS])[0]
    pr = pipeline.predict_proba(df_t[ALL_COLS])[0]
    status = 'Approved' if p == 1 else 'Rejected'
    print(f"{label}: {status} ({pr[1]*100:.1f}%)")

joblib.dump(pipeline, 'best_pipeline.pkl')
print("Pipeline saved as best_pipeline.pkl")
