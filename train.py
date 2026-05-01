import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Load Real Dataset
# -------------------------------------------------------
DATASET_PATH = r"C:\Users\USER\Downloads\loan_approval_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# Strip whitespace from column names and string values
df.columns = df.columns.str.strip()
df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nloan_status distribution:\n{df['loan_status'].value_counts()}")

# -------------------------------------------------------
# 2. Prepare Target Variable
# -------------------------------------------------------
# Map 'Approved' -> 1, 'Rejected' -> 0
df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop loan_id (not a feature)
df = df.drop(columns=['loan_id'], errors='ignore')

# -------------------------------------------------------
# 3. Feature Engineering
# -------------------------------------------------------
def feature_engineering(df_in):
    d = df_in.copy()
    # Total assets
    d['total_assets'] = (
        d['residential_assets_value'].fillna(0) +
        d['commercial_assets_value'].fillna(0) +
        d['luxury_assets_value'].fillna(0) +
        d['bank_asset_value'].fillna(0)
    )
    # Loan-to-income ratio
    d['loan_to_income'] = d['loan_amount'] / (d['income_annum'] + 1)
    # EMI estimate
    d['emi_estimate'] = d['loan_amount'] / (d['loan_term'] + 1)
    return d

df = feature_engineering(df)

# -------------------------------------------------------
# 4. Define Features & Split
# -------------------------------------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------
# 5. Preprocessing Pipelines
# -------------------------------------------------------
num_cols = [
    'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value',
    'bank_asset_value', 'total_assets', 'loan_to_income', 'emi_estimate',
    'no_of_dependents'
]

cat_ordinal_cols = ['education', 'self_employed']

num_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = SkPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_ordinal_cols),
])

# -------------------------------------------------------
# 6. Model Training & Evaluation
# -------------------------------------------------------
models = {
    'Logistic Regression': (LogisticRegression(max_iter=10000), {
        'classifier__C': [0.1, 1, 10]
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'classifier__max_depth': [None, 5, 10]
    }),
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    }),
    'XGBoost': (xgb.XGBClassifier(eval_metric='logloss', random_state=42), {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [5, 7, 10]
    })
}

best_models = {}
plt.figure(figsize=(10, 8))

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    grid = GridSearchCV(pipeline, params, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"[{name}] Best Params: {grid.best_params_}")
    print(f"[{name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"[{name}] Confusion Matrix:\n{cm}")

    # ROC Curve
    if hasattr(best_model.named_steps['classifier'], "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score = best_model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("\nROC curve saved as roc_curve.png")

# -------------------------------------------------------
# 7. Save Best Model (XGBoost)
# -------------------------------------------------------
# 8. Pick the best model by F1 score and save it
# -------------------------------------------------------
from sklearn.metrics import f1_score as _f1

best_name, best_score = None, -1
for name, m in best_models.items():
    score = _f1(y_test, m.predict(X_test))
    print(f"[{name}] F1 on test: {score:.4f}")
    if score > best_score:
        best_score = score
        best_name = name

print(f"\n>> Best model: {best_name} (F1={best_score:.4f})")
joblib.dump(best_models[best_name], 'best_pipeline.pkl')
print(f"Pipeline saved as best_pipeline.pkl")
