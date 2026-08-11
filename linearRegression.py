# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 2. Load external dataset
data = pd.read_csv("Commonwealth_Medalists_1930_2026.csv")

# Display first 5 rows
print(data.head())


# 3. Select input and output
X = data[["YearsExperience"]]   # Independent variable
y = data["Salary"]              # Dependent variable


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Create Linear Regression model
model = LinearRegression()


# 6. Train the model
model.fit(X_train, y_train)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Check model performance
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 9. Predict salary for a new value
years = [[5]]

predicted_salary = model.predict(years)

print("Predicted salary:", predicted_salary[0])


# 10. Visualize results
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")

plt.show()