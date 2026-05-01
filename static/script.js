// Dataset training ranges — values outside these may reduce accuracy
const DATASET_RANGES = {
    income_annum:              { min: 200000,   max: 9900000,   label: 'Annual Income' },
    loan_amount:               { min: 600000,   max: 39500000,  label: 'Loan Amount' },
    cibil_score:               { min: 300,      max: 900,       label: 'CIBIL Score' },
    residential_assets_value:  { min: -100000,  max: 29900000,  label: 'Residential Assets' },
    commercial_assets_value:   { min: 0,        max: 19000000,  label: 'Commercial Assets' },
    luxury_assets_value:       { min: 300000,   max: 38200000,  label: 'Luxury Assets' },
    bank_asset_value:          { min: 0,        max: 14400000,  label: 'Bank Asset Value' },
};

document.getElementById('loan-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData.entries());

    const numericFields = [
        'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
        'cibil_score', 'residential_assets_value', 'commercial_assets_value',
        'luxury_assets_value', 'bank_asset_value'
    ];
    numericFields.forEach(field => {
        if (data[field] !== undefined) data[field] = Number(data[field]);
    });

    // Check for out-of-range inputs
    const warnings = [];
    for (const [field, range] of Object.entries(DATASET_RANGES)) {
        const val = data[field];
        if (val !== undefined && (val < range.min || val > range.max)) {
            warnings.push(`${range.label} (${val.toLocaleString()}) is outside training range [${range.min.toLocaleString()} – ${range.max.toLocaleString()}]`);
        }
    }

    const btn = document.getElementById('predict-btn');
    btn.innerHTML = '⏳ Predicting...';
    btn.disabled = true;

    const resultContainer = document.getElementById('result-container');
    const resultText      = document.getElementById('result-text');
    const resultSub       = document.getElementById('result-sub');
    resultContainer.className = 'hidden';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        resultContainer.className = '';

        if (result.error) {
            resultText.innerText = '⚠️ Error';
            resultSub.innerText  = result.error;
            resultContainer.classList.add('error');
        } else {
            const icon = result.prediction === 'Approved' ? '✅' : '❌';
            // Fix: use toFixed(1) to always show exactly 1 decimal place
            const prob = parseFloat(result.probability).toFixed(1);
            resultText.innerText = `${icon} ${result.prediction}`;

            let subText = `Approval Probability: ${prob}%`;
            if (result.override_reason) {
                subText += `\n\n⚠️ ${result.override_reason}`;
            }
            if (warnings.length > 0 && !result.override_reason) {
                subText += `\n\n⚠️ Note: Some inputs are outside the model's training range — prediction accuracy may be reduced:\n• ${warnings.join('\n• ')}`;
            }
            resultSub.innerText = subText;
            resultContainer.classList.add(
                result.prediction === 'Approved' ? 'approved' : 'rejected'
            );
        }

        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

    } catch (err) {
        resultText.innerText = '⚠️ Network Error';
        resultSub.innerText  = 'Could not reach the server. Please try again.';
        resultContainer.className = 'error';
        console.error('Prediction error:', err);
    } finally {
        btn.innerHTML = 'Predict Approval';
        btn.disabled  = false;
    }
});
