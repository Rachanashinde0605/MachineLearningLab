# 1. Library Import

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Read Data

df = pd.read_csv("diabetes_prediction_dataset.csv")

print(df.head())


# 3. EDA

print("\nShape of Dataset:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Distribution:")
print(df["diabetes"].value_counts())


# 4. Data Preprocessing

# Convert categorical columns into numbers
le = LabelEncoder()

df["gender"] = le.fit_transform(df["gender"])
df["smoking_history"] = le.fit_transform(df["smoking_history"])


# Separate input and output

X = df.drop("diabetes", axis=1)
y = df["diabetes"]


# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Cross Validation

model = GaussianNB()

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(cv_scores)

print("Mean CV Accuracy:", cv_scores.mean())


# 6. Model Training

model.fit(X_train, y_train)


# 7. Prediction

y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred[:20])


# 8. Evaluation Metrics

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))