import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ==========================================
# 1. LOAD YOUR KAGGLE DATASET
# ==========================================
file_path = r"C:\Users\Lenovo\Downloads\archive (3)\housing.csv"
df = pd.read_csv(file_path)

print("--- Data Processing Started ---")

# Handle missing values if any exist (like in total_bedrooms)
df = df.dropna()

# Convert categorical 'ocean_proximity' column to numbers using One-Hot Encoding
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Define Features (X) - Drop the target column
X = df.drop(columns=['median_house_value'])

# ==========================================
# 2. DEFINE TARGETS (REGRESSION & CLASSIFICATION)
# ==========================================
# Target A: Continuous value for Regression
y_reg = df['median_house_value']

# Target B: Create a binary category for Classification (1 if house is above median price, 0 if below)
median_price = df['median_house_value'].median()
y_class = (df['median_house_value'] > median_price).astype(int)

# ==========================================
# 3. SPLIT AND SCALE THE DATA
# ==========================================
# Split for Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Split for Classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Scale features for the Linear & Logistic models
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)

# ==========================================
# 4. REGRESSION MODELS (Linear Regression & Random Forest)
# ==========================================
print("\n" + "="*40)
print("TRAINING REGRESSION MODELS...")
print("="*40)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
lr_preds = lr_model.predict(X_test_reg_scaled)

# Model 2: Random Forest Regressor
rf_reg_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf_reg_model.fit(X_train_reg, y_train_reg)
rf_preds = rf_reg_model.predict(X_test_reg)

# Evaluate Regression Results
print(f"Linear Regression R² Score: {r2_score(y_test_reg, lr_preds):.4f}")
print(f"Random Forest Regressor R² Score: {r2_score(y_test_reg, rf_preds):.4f}")

# ==========================================
# 5. CLASSIFICATION MODEL (Logistic Regression)
# ==========================================
print("\n" + "="*40)
print("TRAINING CLASSIFICATION MODEL...")
print("="*40)

# Model 3: Logistic Regression
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_cls_scaled, y_train_cls)
clf_preds = clf_model.predict(X_test_cls_scaled)

# Evaluate Classification Results
print(f"Classification Accuracy: {accuracy_score(y_test_cls, clf_preds):.4f}\n")
print("Detailed Classification Report:")
print(classification_report(y_test_cls, clf_preds))