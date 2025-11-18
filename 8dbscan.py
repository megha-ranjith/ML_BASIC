import math
import numpy as np

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def region_query(points, point_idx, epsilon):
    neighbors = []
    for i, point in enumerate(points):
        if euclidean_distance(points[point_idx], point) <= epsilon:
            neighbors.append(i)
    return neighbors

def expand_cluster(points, labels, point_idx, neighbors, cluster_id, epsilon, min_points, visited):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            neighbors_of_neighbor = region_query(points, neighbor_idx, epsilon)
            if len(neighbors_of_neighbor) >= min_points:
                for nb in neighbors_of_neighbor:
                    if nb not in neighbors:
                        neighbors.append(nb)
        if labels[neighbor_idx] == -1:
            labels[neighbor_idx] = cluster_id
        i += 1

def dbscan(points, epsilon, min_points):
    n_points = len(points)
    labels = [-1] * n_points  # -1 for noise unless assigned to a cluster
    visited = [False] * n_points
    cluster_id = 0

    for point_idx in range(n_points):
        if visited[point_idx]:
            continue
        visited[point_idx] = True
        neighbors = region_query(points, point_idx, epsilon)
        if len(neighbors) < min_points:
            labels[point_idx] = -1  # Noise
        else:
            expand_cluster(points, labels, point_idx, neighbors, cluster_id, epsilon, min_points, visited)
            cluster_id += 1
    return labels

def get_core_border_noise(points, labels, epsilon, min_points):
    core_points, border_points, noise_points = [], [], []
    n_points = len(points)
    for idx in range(n_points):
        if labels[idx] == -1:
            noise_points.append(points[idx])
        else:
            neighbors = region_query(points, idx, epsilon)
            if len(neighbors) >= min_points:
                core_points.append(points[idx])
            else:
                border_points.append(points[idx])
    return core_points, border_points, noise_points

def main():
    epsilon = float(input("Enter epsilon (ε) value (radius): "))
    min_points = int(input("Enter MinPts (minimum points to form a cluster): "))
    num_points = int(input("Enter the number of points: "))
    points = []
    print("Enter points (each as space-separated x y):")
    for _ in range(num_points):
        pt = tuple(map(float, input().split()))
        points.append(pt)

    labels = dbscan(points, epsilon, min_points)
    core_points, border_points, noise_points = get_core_border_noise(points, labels, epsilon, min_points)

    print("\nCluster Assignments:")
    for i, label in enumerate(labels):
        print(f"Point {points[i]} => Cluster {label if label != -1 else 'Noise'}")
    print("\nCore Points:")
    for pt in core_points:
        print(pt)
    if border_points:
        print("\nBorder Points:")
        for pt in border_points:
            print(pt)
    print("\nNoise Points:")
    for pt in noise_points:
        print(pt)

if __name__ == "__main__":
    main()
