# --------------------------------------------
# 📊 Big Sales Prediction - YBI Internship Project
# --------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ------------------- 1. Data Loading -------------------
print("✅ Step 1: Loading dataset...")

# Dataset Kaggle se download karein: https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
# Agar local mein file hai toh niche path dena. Nahi toh mai sample data generate karta hoon.
try:
    # Ye maan ke chal rahe hain ki aapne 'Train.csv' download karke same folder mein rakha hai
    df = pd.read_csv('Train.csv')
    print(f"✅ Dataset loaded! Shape: {df.shape}")
except FileNotFoundError:
    print("⚠️ Train.csv nahi mila! Mai abhi dummy data bana raha hoon (demo ke liye).")
    print("   Real project ke liye Kaggle se dataset download karein.")
    # Dummy data (practice ke liye)
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Item_Identifier': ['ID' + str(i) for i in range(n)],
        'Item_Weight': np.random.uniform(5, 20, n),
        'Item_Fat_Content': np.random.choice(['Low Fat', 'Regular'], n),
        'Item_Visibility': np.random.uniform(0, 0.3, n),
        'Item_Type': np.random.choice(['Dairy', 'Meat', 'Snacks', 'Fruits'], n),
        'Item_MRP': np.random.uniform(30, 300, n),
        'Outlet_Identifier': np.random.choice(['OUT001', 'OUT002', 'OUT003'], n),
        'Outlet_Establishment_Year': np.random.choice([1990, 1995, 2000, 2005], n),
        'Outlet_Size': np.random.choice(['Small', 'Medium', 'High'], n),
        'Outlet_Location_Type': np.random.choice(['Tier 1', 'Tier 2', 'Tier 3'], n),
        'Outlet_Type': np.random.choice(['Grocery Store', 'Supermarket Type1', 'Supermarket Type2'], n),
        'Item_Outlet_Sales': np.random.uniform(100, 1000, n)  # Target
    })

# ------------------- 2. Data Exploration (EDA) -------------------
print("\n✅ Step 2: Exploratory Data Analysis")
print(df.head(3))
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Missing values fill karna
if 'Item_Weight' in df.columns:
    df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
if 'Outlet_Size' in df.columns:
    df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)

# Data Cleaning: 'Item_Fat_Content' ko standardize karo
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
})

# ------------------- 3. Feature Engineering -------------------
print("\n✅ Step 3: Feature Engineering")

# Outlet age (current year - establishment year)
df['Outlet_Years'] = 2024 - df['Outlet_Establishment_Year']

# Categorical columns ko encode karna (Label Encoding)
categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type',
                    'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

encoder = LabelEncoder()
for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

# Drop columns jo kaam nahi aayenge (agar real data hai toh)
# 'Item_Identifier' drop kar dete hain kyunki ye unique ID hai
if 'Item_Identifier' in df.columns:
    df.drop('Item_Identifier', axis=1, inplace=True)
if 'Outlet_Establishment_Year' in df.columns:
    df.drop('Outlet_Establishment_Year', axis=1, inplace=True)

# ------------------- 4. Train-Test Split -------------------
print("\n✅ Step 4: Splitting data...")

X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ------------------- 5. Model Building (Random Forest) -------------------
print("\n✅ Step 5: Training Random Forest Regressor...")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 R² Score: {r2:.4f}")

# ------------------- 6. Feature Importance (kaunsi feature sabse important hai) -------------------
print("\n✅ Step 6: Feature Importance")
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('Feature Importance for Sales Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature importance graph saved as 'feature_importance.png'")

# ------------------- 7. Save the model -------------------
import joblib
joblib.dump(rf, 'model.pkl')
print("\n✅ Model saved as 'model.pkl'")

# ------------------- 8. Final Result (Certificate ke liye summary) -------------------
print("\n" + "="*50)
print("🎯 PROJECT SUMMARY")
print("="*50)
print(f"✅ Best Model: Random Forest Regressor")
print(f"✅ R² Score on Test Data: {r2:.4f} (Acha model hai agar R² > 0.6 hai)")
print(f"✅ RMSE: {rmse:.2f}")
print("✅ Most important feature:", importance.iloc[0]['Feature'])
print("="*50)
print("\n🎉 Project ready for submission! Ye code complete hai.")