import pandas as pd

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import joblib

import os

# CELL 3: Load and Prepare Data
# Load dataset
df = pd.read_csv('heart_data.csv')
df = df.drop("Unnamed: 0", axis=1)

# Display data
print(df.head())

# CELL 4: Visualize Data
# Plot relationships
sns.lmplot(x='biking', y='heart.disease', data=df)
sns.lmplot(x='smoking', y='heart.disease', data=df)

# CELL 5: Train Model
# Prepare data
X = df.drop('heart.disease', axis=1)
y = df['heart.disease']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
score = model.score(X_train, y_train)
print(f"Model R² score: {score}")

# CELL 6: Save Model Safely
# Create models directory if needed
os.makedirs('models', exist_ok=True)

# Save with joblib (preferred for scikit-learn)
joblib.dump(model, 'models/model.joblib')

# Save with pickle (not recommended for version mismatch issues)
# with open('models/model.pkl', 'wb') as f:
#     pickle.dump(model, f)