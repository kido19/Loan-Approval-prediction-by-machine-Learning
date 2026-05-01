import requests

BASE = 'http://localhost:5000'

cases = [
    ('BAD  - CIBIL=300, income=200k, loan=10M (expect REJECT)', {
        'education': 'Graduate', 'self_employed': 'Yes', 'no_of_dependents': 0,
        'loan_term': 2, 'income_annum': 200000, 'loan_amount': 9999998,
        'cibil_score': 300, 'residential_assets_value': 0,
        'commercial_assets_value': 2000, 'luxury_assets_value': 300000, 'bank_asset_value': 56765
    }),
    ('GOOD - CIBIL=800, income=9M, loan=10M (expect APPROVE)', {
        'education': 'Graduate', 'self_employed': 'No', 'no_of_dependents': 1,
        'loan_term': 12, 'income_annum': 9000000, 'loan_amount': 10000000,
        'cibil_score': 800, 'residential_assets_value': 5000000,
        'commercial_assets_value': 2000000, 'luxury_assets_value': 3000000, 'bank_asset_value': 1000000
    }),
    ('MID  - CIBIL=650, income=5M, loan=8M (borderline)', {
        'education': 'Graduate', 'self_employed': 'No', 'no_of_dependents': 2,
        'loan_term': 10, 'income_annum': 5000000, 'loan_amount': 8000000,
        'cibil_score': 650, 'residential_assets_value': 3000000,
        'commercial_assets_value': 1000000, 'luxury_assets_value': 2000000, 'bank_asset_value': 500000
    }),
    ('EDGE - CIBIL=490, income=3M, loan=6M (expect REJECT - CIBIL<500)', {
        'education': 'Not Graduate', 'self_employed': 'Yes', 'no_of_dependents': 3,
        'loan_term': 20, 'income_annum': 3000000, 'loan_amount': 6000000,
        'cibil_score': 490, 'residential_assets_value': 500000,
        'commercial_assets_value': 0, 'luxury_assets_value': 500000, 'bank_asset_value': 100000
    }),
]

for label, payload in cases:
    r = requests.post(BASE + '/predict', json=payload)
    d = r.json()
    pred = d.get('prediction', 'ERROR')
    prob = d.get('probability', '?')
    reason = d.get('override_reason', '')
    print(label)
    print("  Result  :", pred, str(prob) + "%")
    if reason:
        print("  Override:", reason)
    print()
