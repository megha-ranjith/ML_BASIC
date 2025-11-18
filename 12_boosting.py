import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data  
y = iris.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=32)

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Manual prediction function
def manual_prediction():
    print("\nPlease enter the following details for prediction:")
    sepal_length = float(input("Sepal Length (cm): "))
    sepal_width = float(input("Sepal Width (cm): "))
    petal_length = float(input("Petal Length (cm): "))
    petal_width = float(input("Petal Width (cm): "))
    
    # Create a new data point with the user's input
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict the class of the entered data
    predicted_class = model.predict(user_input)
    predicted_species = iris.target_names[predicted_class[0]]
    print(f"\nPredicted species: {predicted_species}")

# Ask the user if they want to make a manual prediction
manual = input("\nWould you like to make a manual prediction? (y/n): ").strip().lower()
if manual == 'y':
    manual_prediction()
else:
    print("\nExit!")
