import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
try:
    data = pd.read_csv("blood.csv")
    print("Dataset loaded successfully!\n")
    print(data.head())
except FileNotFoundError:
    print("Error: The CSV file was not found.")
    exit()

# Extract features and target
try:
    X = data[["Glucose", "BloodPressure"]]
    y = data["Outcome"]
except KeyError as e:
    print(f"Error: Missing column - {e}")
    exit()

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Predict for user input
try:
    glucose = float(input("\nEnter Glucose value: "))
    bp = float(input("Enter Blood Pressure value: "))
    user_input = pd.DataFrame([[glucose, bp]], columns=["Glucose", "BloodPressure"])
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        print("Predicted Outcome: Positive (Diabetes likely)")
    else:
        print("Predicted Outcome: Negative (No Diabetes)")
except ValueError:
    print("Invalid input for Glucose or Blood Pressure.")
except Exception as e:
    print(f"Error: {e}")
