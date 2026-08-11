# 1. Import libraries
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Read CSV file
data = pd.read_csv("titanic_dataset.csv")

# Select input and output
X = data[['Age', 'Fare', 'Sex', 'Pclass', 'sibsp', 'Parch', 'Embarked']]
y = data['Survived']

# Handle missing values
X = X.fillna(X.mean())

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Fit the model - Algorithm
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix-Logistic Regression")
plt.show()