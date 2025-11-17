import csv
import random
import math
from collections import Counter


def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader) 
        for row in reader:
            features = list(map(float, row[:-1]))  
            label = row[-1]  
            dataset.append((features, label))
    return dataset


def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def train_test_split(dataset, test_size=0.2):
    shuffled = dataset[:]
    random.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_index], shuffled[split_index:]


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        return [self._predict(x) for x in X_test]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def accuracy_score(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, labels):
    matrix = [[0]*len(labels) for _ in range(len(labels))]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        matrix[i][j] += 1
    return matrix

def main():
    dataset = load_csv('iris.csv')

    X = [item[0] for item in dataset]
    y = [item[1] for item in dataset]

    train_data, test_data = train_test_split(dataset, test_size=0.2)
    X_train = [item[0] for item in train_data]
    y_train = [item[1] for item in train_data]
    X_test = [item[0] for item in test_data]
    y_test = [item[1] for item in test_data]

    knn = KNearestNeighbors(k=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%\n")


    print(f"{'Original Label':<15}{'Predicted Label':<20}{'Status'}")
    for original, predicted in zip(y_test, y_pred):
        status = "correct" if original == predicted else "wrong"
        print(f"{original:<15}{predicted:<20}{status}")


    labels = sorted(set(y))
    matrix = confusion_matrix(y_test, y_pred, labels)

    print("\n\nConfusion Matrix:")
    print(f"{'':<15}", end="")
    for label in labels:
        print(f"{label:<15}", end="")
    print()
    for i, label in enumerate(labels):
        print(f"{label:<15}", end="")
        for j in range(len(labels)):
            print(f"{matrix[i][j]:<15}", end="")
        print()

 
    predict_user_input(knn)

def predict_user_input(knn):
    print("\nEnter features for a new Iris sample:")
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))

        sample = [sepal_length, sepal_width, petal_length, petal_width]
        prediction = knn._predict(sample)

        print(f"Predicted class for the input sample: {prediction}")
    except ValueError:
        print("Invalid input! Please enter numeric values.")

if __name__ == "__main__":
    main()