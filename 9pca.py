import numpy as np
from sklearn.decomposition import PCA

def perform_pca():
    # Ask the user for the input dimensions and data points
    current_dimension = int(input("Enter current dimension (number of features): "))
    num_data_points = int(input("Enter the number of data points: "))
    
    # Initialize an empty list and collect the data
    data = []
    print("Enter each data point as space-separated values:")
    for i in range(num_data_points):
        while True:
            raw = input(f"Data point {i+1}: ").strip().split()
            try:
                data_point = [float(x) for x in raw]
                if len(data_point) != current_dimension:
                    print(f"  Error: Must enter exactly {current_dimension} values.")
                    continue
                break
            except ValueError:
                print("  Error: Please enter only numeric values.")
        data.append(data_point)
    data = np.array(data)

    # Choose number of PCA components
    reduced_dimension = int(input(f"Enter desired reduced dimension (<= {current_dimension}): "))
    if reduced_dimension > current_dimension or reduced_dimension < 1:
        print("Error: Dimension out of range.")
        return

    # Perform PCA
    pca = PCA(n_components=reduced_dimension)
    reduced_data = pca.fit_transform(data)

    print("\nPrincipal Components (Eigenvectors):")
    print(pca.components_)
    print("\nExplained Variance Ratio (per component):")
    print(pca.explained_variance_ratio_)
    print("\nReduced Data (after PCA transformation):")
    print(reduced_data)

# Execute
if __name__ == "__main__":
    perform_pca()
