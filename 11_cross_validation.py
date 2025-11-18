import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels

# Initialize the classifier (Random Forest as an example)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Arrays to store performance metrics for each fold
accuracies = []
precisions = []
recalls = []
f_scores = []

# 5-fold cross-validation
fold = 1
for train_index, test_index in kf.split(X, y):
    # Split data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict the labels on the test set
    y_pred = clf.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f_score = f1_score(y_test, y_pred, average='weighted')

    # Append the metrics to the respective lists
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f_scores.append(f_score)

    # Print metrics for the current fold
    print(f"Fold {fold}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F-Score: {f_score:.4f}")
    print()
    fold += 1

# Calculate and display the average performance metrics across all folds
print("Average Performance Metrics:")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average F-Score: {np.mean(f_scores):.4f}")
