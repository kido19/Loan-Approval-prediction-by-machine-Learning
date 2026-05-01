from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'best_pipeline.pkl'

# -------------------------------------------------------
# Feature engineering — must match train.py exactly
# -------------------------------------------------------
def feature_engineering(df_in):
    d = df_in.copy()
    d['total_assets'] = (
        d['residential_assets_value'].fillna(0) +
        d['commercial_assets_value'].fillna(0) +
        d['luxury_assets_value'].fillna(0) +
        d['bank_asset_value'].fillna(0)
    )
    d['loan_to_income'] = d['loan_amount'] / (d['income_annum'] + 1)
    d['emi_estimate']   = d['loan_amount'] / (d['loan_term'] + 1)
    return d

# -------------------------------------------------------
# Business rule overrides (based on training data patterns)
# Training data loan/income ratios: 1x – 5x max
# Training data CIBIL < 500: almost always Rejected
# -------------------------------------------------------
def apply_business_rules(data, ml_pred, ml_prob):
    """
    Apply hard business rules for extreme / out-of-distribution inputs.
    Returns (prediction_label, probability_pct, override_reason or None).
    """
    cibil       = data.get('cibil_score', 900)
    income      = data.get('income_annum', 1)
    loan        = data.get('loan_amount', 0)
    total_assets = (data.get('residential_assets_value', 0) +
                    data.get('commercial_assets_value', 0) +
                    data.get('luxury_assets_value', 0) +
                    data.get('bank_asset_value', 0))

    loan_to_income = loan / (income + 1)
    asset_coverage = total_assets / (loan + 1)

    reasons = []

    # Rule 1: CIBIL below 500 is Very Poor — nearly always rejected in training data
    if cibil < 500:
        reasons.append(f"CIBIL score {cibil} is Very Poor (below 500)")

    # Rule 2: Loan-to-income ratio > 10x is financially unrealistic
    if loan_to_income > 10:
        reasons.append(f"Loan-to-income ratio is {loan_to_income:.1f}x (max realistic: 10x)")

    # Rule 3: Loan amount >> total assets with poor CIBIL
    if cibil < 600 and asset_coverage < 0.3:
        reasons.append(f"Total assets ({total_assets:,.0f}) cover only {asset_coverage*100:.0f}% of the loan")

    if reasons:
        override_reason = "Auto-rejected due to: " + "; ".join(reasons)
        return 0, round(min(ml_prob, 15.0), 1), override_reason

    return ml_pred, ml_prob, None

# -------------------------------------------------------
# Column order must match ColumnTransformer in train.py
# -------------------------------------------------------
NUM_COLS = [
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value',
    'total_assets', 'loan_to_income', 'emi_estimate',
    'no_of_dependents'
]
CAT_COLS = ['education', 'self_employed']
ALL_COLS = NUM_COLS + CAT_COLS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': f'Model not found at {MODEL_PATH}. Please run train.py first.'})

    try:
        pipeline = joblib.load(MODEL_PATH)
        data = request.json

        print("\n--- Incoming Request ---")
        print("Raw data: " + str(data))

        # Cast all numeric fields to float
        numeric_fields = [
            'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]
        for field in numeric_fields:
            if field in data:
                data[field] = float(data[field])

        # Build dataframe and apply feature engineering
        df = pd.DataFrame([data])
        df = feature_engineering(df)

        # Verify all expected columns are present
        missing = [c for c in ALL_COLS if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'})

        print(f"\nDataFrame columns: {list(df.columns)}")
        print(f"DataFrame values:\n{df[ALL_COLS].to_string()}")

        # Predict (ML model)
        pred_label    = pipeline.predict(df[ALL_COLS])[0]
        pred_proba    = pipeline.predict_proba(df[ALL_COLS])[0]
        prob_approval = float(round(pred_proba[1] * 100, 1))

        print(f"\nML raw proba [Rejected, Approved]: {pred_proba}")

        # Apply business rules for extreme / out-of-distribution inputs
        pred_label, prob_approval, override_reason = apply_business_rules(
            data, pred_label, prob_approval
        )

        status = 'Approved' if pred_label == 1 else 'Rejected'

        print("Final prediction: {} ({}%)".format(status, prob_approval))
        if override_reason:
            print("  [Override] " + override_reason)
        print("-" * 44 + "\n")

        response = {'prediction': status, 'probability': prob_approval}
        if override_reason:
            response['override_reason'] = override_reason
        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

# -------------------------------------------------------
# Quick sanity-check endpoint: /test
# -------------------------------------------------------
@app.route('/test')
def test():
    """
    Tests two contrasting cases directly — visit http://localhost:5000/test
    to verify the model produces different results for different inputs.
    """
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': 'Model not found.'})

    pipeline = joblib.load(MODEL_PATH)

    # Case 1: High CIBIL, high income → should be Approved
    case_approved = {
        'no_of_dependents': 1.0, 'education': 'Graduate', 'self_employed': 'No',
        'income_annum': 9000000.0, 'loan_amount': 10000000.0, 'loan_term': 12.0,
        'cibil_score': 800.0, 'residential_assets_value': 5000000.0,
        'commercial_assets_value': 2000000.0, 'luxury_assets_value': 3000000.0,
        'bank_asset_value': 1000000.0
    }

    # Case 2: Low CIBIL, low income → should be Rejected
    case_rejected = {
        'no_of_dependents': 5.0, 'education': 'Not Graduate', 'self_employed': 'Yes',
        'income_annum': 300000.0, 'loan_amount': 10000000.0, 'loan_term': 20.0,
        'cibil_score': 310.0, 'residential_assets_value': 100000.0,
        'commercial_assets_value': 0.0, 'luxury_assets_value': 100000.0,
        'bank_asset_value': 50000.0
    }

    results = []
    for label, case in [('High-CIBIL (expect Approved)', case_approved),
                         ('Low-CIBIL  (expect Rejected)', case_rejected)]:
        df = pd.DataFrame([case])
        df = feature_engineering(df)
        pred   = pipeline.predict(df[ALL_COLS])[0]
        proba  = pipeline.predict_proba(df[ALL_COLS])[0]
        status = 'Approved' if pred == 1 else 'Rejected'
        results.append({
            'case': label,
            'prediction': status,
            'approval_probability': f"{round(proba[1]*100, 1)}%",
            'cibil_score': case['cibil_score']
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
