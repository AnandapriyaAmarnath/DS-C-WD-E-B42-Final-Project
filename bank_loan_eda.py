import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load CSV
df = pd.read_csv("bank_loans.csv")

# ----- EDA -----
print("Shape of dataset:", df.shape)
print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLoan Status count:")
print(df['Loan_Status'].value_counts())

# Plot Loan Status distribution
sns.countplot(x='Loan_Status', data=df)
plt.title("Loan Approval Distribution")
plt.show()

# Plot Applicant Income distribution
sns.histplot(df['ApplicantIncome'], bins=10, kde=True)
plt.title("Applicant Income Distribution")
plt.show()

# Loan Status vs Credit History
sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
plt.title("Loan Status vs Credit History")
plt.show()

# ----- Machine Learning -----
# Encode categorical columns
le = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train on entire dataset (small dataset)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predictions on same data
y_pred = model.predict(X)

# Evaluation
print("Accuracy on entire dataset:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))






