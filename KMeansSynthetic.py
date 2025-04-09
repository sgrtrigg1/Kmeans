import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic dataset with 327 samples and 300 features
x = np.random.randn(327, 300)


# -----------------------------
# KMeansSynthetic
# -----------------------------
def initialSelection(x, k):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of centroids to select.
    Output:
      - A numpy array of shape (k, features) containing k randomly chosen initial centroids.
    """
    Y = x[np.random.choice(x.shape[0], k, replace=False)]  # choose k random objects
    return Y


def assignClustersIds(x, k, Y):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of clusters.
      - Y: numpy array of centroids (shape: (k, features)).
    Output:
      - A numpy array of cluster indices (one per data point) indicating the nearest centroid.
    """
    clusters = np.zeros(x.shape[0], dtype=int)  # Empty vector for cluster ids
    for i in range(x.shape[0]):  # loop over all 327 objects
        d = np.linalg.norm(x[i] - Y, axis=1)  # Distance of points to centroids
        clusters[i] = np.argmin(d)  # Assign point to the closest centroid
    return clusters


def computeClusterRepresentratives(clusters, k):
    """
    Input:
      - clusters: numpy array of cluster assignments for each data point.
      - k: integer, number of clusters.
    Output:
      - A numpy array of shape (k, features) with updated centroids computed as the mean of each cluster.
    """
    Y_new = np.zeros((k, x.shape[1]))  # new
    for i in range(k):  # loop over all centroids
        Y_new[i] = np.mean(
            x[clusters == i], axis=0
        )  # Move the centroid to the mean of the cluster
    return Y_new


def KMeans(x, k, MaxIter):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of clusters.
      - MaxIter: integer, maximum number of iterations.
    Output:
      - Y: numpy array of final centroids.
      - clusters: numpy array of cluster assignments for each data point.
    """
    Y = initialSelection(x, k)  # Starting centroids
    for i in range(MaxIter):  # Repeating until convergence
        clusters = assignClustersIds(x, k, Y)  # Assign points to a cluster
        Y_new = computeClusterRepresentratives(
            clusters, k
        )  # Update centroids' position
        if np.all(clusters == assignClustersIds(x, k, Y_new)):  # Check for convergence
            Y = Y_new
            break
        Y = Y_new  # Update centroids ... and repeat
    return Y, clusters


def computeSillhouette(x, clusters):
    """
    Input:
      - x: numpy array of data points.
      - clusters: numpy array of cluster assignments.
    Output:
      - score: float representing the mean silhouette coefficient (clustering quality).
    """
    c = []  # silhouette coefficients
    for i in range(x.shape[0]):  # loop over all 327 objects
        d = np.linalg.norm(x - x[i], axis=1)  # Distance of points to other points
        # a_i Cohesion: mean distance to points in the same cluster
        clstr = clusters == clusters[i]  # identify objects in the same cluster
        clstr[i] = False  # Exclude distance from point to itself
        a_i = np.mean(d[clstr]) if np.sum(clstr) > 0 else 0
        # b_i Separation: smallest mean distance to points in any other cluster
        b_i = np.inf  # initialize b_i to infinity
        for cl in np.unique(clusters):  # consider each cluster once
            if cl == clusters[i]:
                continue  # skip the same cluster
            mean_dist = np.mean(d[clusters == cl])
            if mean_dist < b_i:
                b_i = mean_dist
        # silhouette coefficient for the i-th object
        sc_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        c.append(sc_i)
    score = np.mean(c)  # mean silhouette coefficient = silhouette score
    return score


def plotShilhouettee(x, K, max_iter=100):
    """
    Input:
      - x: numpy array of data points.
      - K: iterable of integers, representing different numbers of clusters to try.
      - max_iter: integer, maximum number of iterations for KMeans.
    Output:
      - Saves a plot of Silhouette Score vs K as "Q2Plot.png".
    """
    scores = []
    for k in K:
        if k == 1:
            scores.append(0)  # Silhouette score for k=1 is not defined
            continue
        Y, clusters = KMeans(x, k, max_iter)
        score = computeSillhouette(x, clusters)
        scores.append(score)

    plt.figure()
    plt.plot(list(K), scores, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs K (Synthetic)")
    plt.savefig("Q2Plot.png")


def clustername(x, k):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of clusters.
    Output:
      - names: list of strings, where each string names the cluster (e.g., "Cluster 0") for each data point.
    """
    # Run KMeans on x with k clusters (using max_iter=100)
    _, clusters = KMeans(x, k, 100)
    # Create a list of names for each data point based on its cluster assignment
    names = ["Cluster " + str(label) for label in clusters]
    return names


K = range(1, 10)
plotShilhouettee(x, K)
