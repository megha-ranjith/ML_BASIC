import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
true_labels = iris.target

#K-Means 
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

#EM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)

#Evaluation metrics
silhouette_kmeans = silhouette_score(X, kmeans_labels)
silhouette_gmm = silhouette_score(X, gmm_labels)

ami_kmeans = adjusted_mutual_info_score(true_labels, kmeans_labels)
ami_gmm = adjusted_mutual_info_score(true_labels, gmm_labels)

cm_kmeans = confusion_matrix(true_labels, kmeans_labels)
cm_gmm = confusion_matrix(true_labels, gmm_labels)

print("Silhouette Score for K-Means: {:.2f}".format(silhouette_kmeans))
print("Silhouette Score for GMM: {:.2f}".format(silhouette_gmm))
print("Adjusted Mutual Information for K-Means: {:.2f}".format(ami_kmeans))
print("Adjusted Mutual Information for GMM: {:.2f}".format(ami_gmm))
print("\nConfusion Matrix for K-Means:\n", cm_kmeans)
print("\nConfusion Matrix for GMM:\n", cm_gmm)

#Visualize actual, k-Means, and GMM labels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', s=40)
axes[0].set_title("True Species")
axes[0].set_xlabel('Sepal length')
axes[0].set_ylabel('Sepal width')

axes[1].scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=40)
axes[1].set_title("K-Means Clustering")
axes[1].set_xlabel('Sepal length')
axes[1].set_ylabel('Sepal width')

axes[2].scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', s=40)
axes[2].set_title("Gaussian Mixture Model Clustering")
axes[2].set_xlabel('Sepal length')
axes[2].set_ylabel('Sepal width')

plt.tight_layout()
plt.show()