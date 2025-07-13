import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# 1. Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Clean data
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# 3. Encode target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 4. Label encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 5. Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6. Balance dataset with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# 7. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 8. Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate model
y_pred = model.predict(X_test)
print("✅ F1 Score:", f1_score(y_test, y_pred))
print("📊 Training features:", X.columns.tolist())

# 10. Save model and encoders
joblib.dump(model, "churn_model.pkl")
joblib.dump(encoders, "encoders.pkl")
print("✅ Model and encoders saved!")
