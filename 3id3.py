import pandas as pd
import math
import numpy as np

data = pd.read_csv("climate.csv")
features = list(data.columns)
if "answer" in features:
    features.remove("answer")
else:
    raise Exception("Column 'answer' not found in data")

class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

def entropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row["answer"] == "yes":
            pos += 1
        else:
            neg += 1
    total = pos + neg
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / total
        n = neg / total
        return -(p * math.log(p, 2) + n * math.log(n, 2))

def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (len(subdata) / len(examples)) * sub_e
    return gain

def ID3(examples, attrs):
    root = Node()
    max_gain = -1
    max_feat = None

    unique_labels = examples['answer'].unique()
    if len(unique_labels) == 1:
        root.isLeaf = True
        root.pred = unique_labels[0]
        return root

    if len(attrs) == 0:
        root.isLeaf = True
        root.pred = examples["answer"].mode()[0]
        return root

    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature

    if max_feat is None:
        root.isLeaf = True
        root.pred = examples["answer"].mode()[0]
        return root

    root.value = max_feat
    uniq = np.unique(examples[max_feat])

    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if entropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata["answer"])[0]
            root.children.append(newNode)
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            child = ID3(subdata, new_attrs)
            dummyNode.children.append(child)
            root.children.append(dummyNode)
    return root

def printTree(root: Node, depth=0):
    indent = "\t" * depth
    if root.isLeaf:
        print(f"{indent}{root.value} -> {root.pred}")
    else:
        print(f"{indent}{root.value}")
        for child in root.children:
            printTree(child, depth + 1)

def classify(root: Node, new):
    path = []
    node = root
    while not node.isLeaf:
        attr = node.value
        path.append(attr)
        found_child = False
        for child in node.children:
            if child.value == new.get(attr, None):
                path.append(child.value)
                if child.isLeaf:
                    print("Decision Path:", " -> ".join(path))
                    print("Predicted Label for new example", new, "is:", child.pred)
                    return child.pred
                else:
                    node = child.children[0]
                    found_child = True
                    break
        if not found_child:
            print("Path not found for the given input.")
            return None
    print("Predicted Label:", node.pred)
    return node.pred

def get_user_input(features, data):
    new = {}
    for feature in features:
        options = data[feature].unique()
        options_str = ", ".join(map(str, options))
        val = input(f"Enter value for {feature} ({options_str}): ")
        new[feature] = val
    return new

if __name__ == "__main__":
    root = ID3(data, features)
    print("Decision Tree is:")
    printTree(root)
    print("------------------")

    new = get_user_input(features, data)
    classify(root, new)
