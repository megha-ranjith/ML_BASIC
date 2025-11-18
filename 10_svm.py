import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data  # features
y = iris.target  # target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=28)

# Create an SVM model (Support Vector Classifier)
model = SVC(kernel='linear') 

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Print correct predictions
print("Correct Predictions:")
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print(f"Sample {i+1}: Actual={iris.target_names[y_test[i]]}, Predicted={iris.target_names[y_pred[i]]}")

# Print wrong predictions
print("\nWrong Predictions:")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"Sample {i+1}: Actual={iris.target_names[y_test[i]]}, Predicted={iris.target_names[y_pred[i]]}")

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Manual prediction function
def manual_prediction():
    while True:
        print("\nPlease enter the following values for the Iris flower:")
        # Get user input for each feature
        sepal_length = float(input("Sepal Length (in cm): "))
        sepal_width = float(input("Sepal Width (in cm): "))
        petal_length = float(input("Petal Length (in cm): "))
        petal_width = float(input("Petal Width (in cm): "))

        # Create a numpy array with the input values
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict the class using the trained model
        prediction = model.predict(input_features)

        # Output the predicted class
        print(f"\nPredicted class: {iris.target_names[prediction[0]]}")

        # if user want to make another prediction
        continue_prediction = input("\nDo you want to make another prediction? (yes/no): ").lower()
        if continue_prediction != 'yes':
            print("Exiting prediction mode. Thank you!")
            break

# Call the manual prediction function
manual_prediction()
