💳 Credit Card Approval Prediction System
A complete Machine Learning web application that predicts whether a credit card application will be approved or rejected based on applicant demographic and financial details.
The system is built using Logistic Regression and deployed with an interactive Streamlit UI.
📌 Project Overview
Financial institutions receive thousands of credit card applications daily. Manual evaluation is time-consuming and prone to bias.
This project automates the credit approval decision process using machine learning, ensuring consistency, speed, and transparency.
The application allows users to:
Enter applicant details
Get instant approval or rejection
View approval probability
Download a professional decision report
🚀 Features
End-to-end Machine Learning pipeline
Handles imbalanced data
Feature encoding & scaling
Trained Logistic Regression model
Interactive Streamlit web interface
Approval probability score
Downloadable decision report
Clean, professional UI
Ready for cloud deployment
🧠 Machine Learning Details
Algorithm Used: Logistic Regression
Reason: Interpretable, efficient, and well-suited for binary classification
Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
Preprocessing:
One-Hot Encoding for categorical features
Standard Scaling for numerical features
Class Imbalance Handling: Balanced dataset before training
📂 Project Structure

credit-card-approval-ml/
│
├── app.py                # Streamlit web application
├── main.py               # Model training & evaluation
├── model.pkl             # Trained ML model
├── scaler.pkl            # Feature scaler
├── columns.pkl           # Feature alignment file
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── data/
    └── credit_card_approval.csv
🖥️ Web Application (Streamlit)
The Streamlit interface allows users to input:
Gender
Car ownership
Property ownership
Total income
Number of children
Age
Years of working experience
Family members
Output:
Approval / Rejection decision
Probability score
Downloadable official decision report
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/divyansh-dev8/credit-card-approval-ml.git
cd credit-card-approval-ml
2️⃣ Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Run the application
streamlit run app.py
🌐 Live Deployment
This application is deployed using Streamlit Cloud.
👉 (Add your live URL here once deployed)
Exampe

https://credit-card-approval-ml.streamlit.app
📊 Sample Output
Decision: APPROVED / REJECTED
Approval Probability: e.g. 94.03%
Downloadable Report: Official decision note from the credit card company
🔮 Future Enhancements
Feature importance visualization
Advanced models (Random Forest, XGBoost)
Explainable AI (SHAP values)
User authentication
Database integration
API version for bank systems
⚠️ Disclaimer
This project is built for educational and demonstration purposes only.
It does not represent the actual approval criteria of any financial institution.
👤 Author
Divyansh
Data Science & Machine Learning Enthusiast
GitHub: https://github.com/divyansh-dev8
⭐ If you like this project
Give it a ⭐ on GitHub and share your feedback!
