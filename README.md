# 📊 Customer Churn Prediction App

This project is a **machine learning-powered web application** built with **Streamlit** that predicts whether a customer is likely to churn (leave a telecom service). The model is trained on the popular [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 🚀 Features

- 🔍 Predict customer churn based on 20+ telecom service features
- 📈 Visualize churn probability using an interactive bar chart
- ⚖️ Balanced dataset using SMOTE for better model performance
- 🌐 Web app built with Streamlit for easy user interaction

---

## 📁 Project Structure

churn_project/
│
├── app.py # Streamlit app
├── model_training.py # Script to train model
├── churn_model.pkl # Trained Decision Tree model
├── encoders.pkl # Label encoders for categorical data
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset
├── requirements.txt # Required Python packages
└── README.md # Project documentation

---

## 🧠 Machine Learning Pipeline

- **Data Cleaning:** Handled missing values, dropped unnecessary columns
- **Encoding:** Used `LabelEncoder` for categorical features
- **Balancing:** Used `SMOTE` to handle class imbalance
- **Model:** Trained a 'XGBoost (or Decision Tree)`
- **Evaluation:** Achieved ~0.82 F1 Score on the test set

- 🛠️ Technologies Used
Python

Pandas, scikit-learn, imbalanced-learn

Streamlit (for building the interactive app)

Joblib (for model persistence)

📦 Dataset Used
Telco Customer Churn Dataset

## ⚙️ How to Run the App

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/churn-predictor.git
cd churn-predictor
 Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run the App
streamlit run app.py
Then open http://localhost:8501 in your browser.

🧪 Sample Input for Testing
Try this input to test a high-churn-risk customer:

Feature	                       Value
Gender                    	   Female
Senior Citizen	               1
Partner	                       No
Dependents	                   No
Tenure	                        1
Internet Service	            Fiber optic
Online Security	                No
Streaming TV	                  Yes
Payment Method              	 Electronic check
Monthly Charges        	         90
Total Charges	                   90
...	...
Expected churn probability:    Above 70%

👨‍💻 Author
Made with ❤️ by [Esha Jha]

Feel free to ⭐ the repo and share your feedback!

📃 License
This project is open-source and free to use under the MIT License.
