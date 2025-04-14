# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
 
# Set random seed for reproducibility
np.random.seed(42)
 
# ------------------------------------------------------------------------------
# 2.1 Acquire, Clean, and Preprocess Data
# ------------------------------------------------------------------------------
 
# (a) Data Acquisition
# Load the dataset (in the same directory as midterm2025.py)
df = pd.read_csv("diabetespredictionDS.csv")
# Alternative full path if needed:
# df = pd.read_csv("C:/Users/ketch/Documents/GitHub/Midterm_2025/diabeticspredictionDS.csv")
print("Dataset loaded successfully. Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
 
# (b) Data Cleaning
# Check for missing values
print("\nMissing values:\n", df.isnull().sum())
 
# Handle 'No Info' in smoking_history by replacing with NaN, then impute with mode
df['smoking_history'] = df['smoking_history'].replace("No Info", np.nan)
print("\nMissing values after replacing 'No Info' with NaN:\n", df['smoking_history'].isnull().sum())
df['smoking_history'] = df['smoking_history'].fillna(df['smoking_history'].mode()[0])
print("\nMissing values after imputation:\n", df['smoking_history'].isnull().sum())
 
# Remove duplicates if any
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)
 
# Encode categorical variables
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])
le_smoking = LabelEncoder()
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])
print("\nGender encoding:", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
print("Smoking history encoding:", dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_))))
 
# Define numerical columns for EDA
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
 
# (c) Data Preprocessing
# Separate features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']
 
# No scaling applied (Random Forest doesn't need it; Logistic Regression can work without for simplicity)
 
# ------------------------------------------------------------------------------
# 2.2 Perform Exploratory Data Analysis (EDA) and Visualize Key Insights
# ------------------------------------------------------------------------------
 
# (a) Exploratory Data Analysis
# Basic statistics
print("\nBasic statistics:\n", df[numerical_cols].describe())
 
# Correlation matrix for numerical columns
correlation_matrix = df[numerical_cols + ['diabetes']].corr()
print("\nCorrelation matrix:\n", correlation_matrix)
 
# Target variable distribution
print("\nDiabetes distribution:\n", df['diabetes'].value_counts(normalize=True))
 
# Identify outliers in 'blood_glucose_level' using IQR
Q1 = df['blood_glucose_level'].quantile(0.25)
Q3 = df['blood_glucose_level'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['blood_glucose_level'] < (Q1 - 1.5 * IQR)) | (df['blood_glucose_level'] > (Q3 + 1.5 * IQR))]
print("\nNumber of outliers in 'blood_glucose_level':", len(outliers))
 
# (b) Data Visualization
# Histogram of 'age'
plt.figure(figsize=(8, 6))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
 
# Pie chart of 'diabetes'
plt.figure(figsize=(8, 6))
df['diabetes'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
plt.title("Diabetes Outcome Distribution")
plt.ylabel('')
plt.legend(['No (0)', 'Yes (1)'], loc='best')
plt.show()
 
# Scatter plot of 'blood_glucose_level' vs 'HbA1c_level' colored by 'diabetes'
plt.figure(figsize=(10, 8))
sns.scatterplot(x='blood_glucose_level', y='HbA1c_level', hue='diabetes', data=df, palette=['lightcoral', 'lightgreen'], alpha=0.6)
plt.title("Blood Glucose Level vs HbA1c Level by Diabetes Outcome")
plt.xlabel("Blood Glucose Level")
plt.ylabel("HbA1c Level")
plt.legend(title='Diabetes', labels=['No (0)', 'Yes (1)'])
plt.show()
 
# ------------------------------------------------------------------------------
# 2.3 Build and Evaluate a Machine Learning Model
# ------------------------------------------------------------------------------
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
 
# (a) Model Building
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
y_pred_log_reg_proba = log_reg.predict_proba(X_test)[:, 1]  # Probabilities for RMSE
 
# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
y_pred_rf_proba = rf_clf.predict_proba(X_test)[:, 1]  # Probabilities for RMSE
 
# (b) Model Evaluation
# Logistic Regression evaluation
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_log_reg_proba))
print("\nLogistic Regression - Accuracy:", log_reg_accuracy)
print("Logistic Regression - RMSE:", log_reg_rmse)
 
# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_proba))
print("\nRandom Forest - Accuracy:", rf_accuracy)
print("Random Forest - RMSE:", rf_rmse)
 
# Interpretation
print("\nModel Evaluation Summary:")
print(f"Logistic Regression: Accuracy = {log_reg_accuracy:.2f}, RMSE = {log_reg_rmse:.2f}")
print(f"Random Forest: Accuracy = {rf_accuracy:.2f}, RMSE = {rf_rmse:.2f}")
print("Accuracy shows the percentage of correct diabetes predictions.")
print("RMSE measures the error between predicted probabilities and true labels (0 or 1).")
print("Random Forest often outperforms Logistic Regression due to its ability to capture complex patterns.")

#code runs fine checked by Jae Cho