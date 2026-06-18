import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
df = pd.read_csv('student-por.csv', sep=';')

# 2. Target variable pass_status (1 if G3 >= 10, else 0)
df['pass_status'] = (df['G3'] >= 10).astype(int)

# 3. Select Simplified Features
features = [
    'age', 'studytime', 'failures', 'absences', 'G1', 'G2', 
    'schoolsup', 'famsup', 'paid', 'internet', 'higher', 'romantic'
]

X = df[features].copy()
y = df['pass_status']

# Encode text categorical values
for col in ['schoolsup', 'famsup', 'paid', 'internet', 'higher', 'romantic']:
    X[col] = X[col].map({'yes': 1, 'no': 0}).fillna(0)

X = X.fillna(0)

# 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 5. Serialize
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
