# 💰 Loan Approval Prediction App

A premium, high-performance Machine Learning web application designed to predict the likelihood of loan approval based on financial data. Built with **Flask** and **XGBoost**, featuring a modern glassmorphism interface.

![Luxury Interface](static/gold_dollar.png)



## 🚀 Features
- **Machine Learning Powered:** Uses state-of-the-art XGBoost algorithm for high accuracy.
- **Intelligent Preprocessing:** Automatic scaling, encoding, and feature engineering (Total Assets, Loan-to-Income ratio).
- **Business Rule Overrides:** Integrated "hard rules" (like CIBIL score thresholds) to mirror real-world banking safety standards.
- **Premium UI:** Dark-themed, glassmorphism design with smooth animations and responsive layout.
- **Interactive Results:** Displays both the approval status and the probability percentage.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** XGBoost, Scikit-learn, Pandas, Numpy, SMOTE (for class balancing)
- **Frontend:** Vanilla HTML5, CSS3 (Custom Glassmorphism), JavaScript

## 📂 Project Structure
- `app.py`: Main Flask server and business logic.
- `train.py`: Machine Learning pipeline and model training script.
- `best_pipeline.pkl`: The trained and serialized model ready for production.
- `static/`: CSS styles, JavaScript logic, and UI assets.
- `templates/`: HTML structure.
- `eda.py`: Exploratory Data Analysis script.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kido19/Loan-Approval-prediction-by-machine-Learning.git
   cd Loan-Approval-prediction-by-machine-Learning
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## 🧠 Model Details
The model was trained on the `loan_approval_dataset` and optimized using **GridSearchCV**.
- **Best Algorithm:** XGBoost (Gradient Boosting)
- **Key Features:** CIBIL Score, Annual Income, Loan Amount, Asset-to-Loan Ratio.
- **Validation:** Evaluated using F1-Score to ensure a balance between Precision and Recall.

---
