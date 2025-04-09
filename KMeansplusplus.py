import numpy as np
import matplotlib.pyplot as plt

x = []
try:
    with open("dataset", "r") as f:
        for i in f:
            data = i.strip().split(" ")
            x.append(data[1:])
except Exception as e:
    print("Error reading dataset", e)
    exit(1)
x = np.array(x, dtype=float)


# -----------------------------
# KMeans using KMeans++ initialization
# -----------------------------
def initialSelection(x, k):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of centroids to select.
    Output:
      - numpy array of shape (k, features) with initial centroids using KMeans++.
    """
    # KMeans++ initialization:
    # 1. Choose the first centroid randomly.
    n_samples = x.shape[0]
    centroids = []
    idx = np.random.randint(n_samples)
    centroids.append(x[idx])
    # 2. For each subsequent centroid, choose one with probability proportional to the squared distance.
    for _ in range(1, k):
        # Compute the squared distance from each point to the nearest centroid
        dist_sq = np.array(
            [min([np.sum((xi - c) ** 2) for c in centroids]) for xi in x]
        )
        probs = dist_sq / dist_sq.sum()
        new_idx = np.random.choice(n_samples, p=probs)
        centroids.append(x[new_idx])
    return np.array(centroids)


def assignClustersIds(x, k, Y):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of clusters.
      - Y: numpy array of centroids (shape: (k, features)).
    Output:
      - numpy array of cluster indices for each data point.
    """
    clusters = np.zeros(x.shape[0], dtype=int)  # Empty vector for cluster ids
    for i in range(x.shape[0]):  # loop over all data points
        d = np.linalg.norm(x[i] - Y, axis=1)  # Distance of point i to each centroid
        clusters[i] = np.argmin(d)  # Assign point to the closest centroid
    return clusters


def computeClusterRepresentratives(clusters, k):
    """
    Input:
      - clusters: numpy array of cluster assignments for each data point.
      - k: integer, number of clusters.
    Output:
      - numpy array of shape (k, features) with updated centroids computed as the mean of each cluster.
    """
    Y_new = np.zeros((k, x.shape[1]))  # new centroids array
    for i in range(k):  # loop over each centroid
        Y_new[i] = np.mean(
            x[clusters == i], axis=0
        )  # Compute the mean of points in cluster i
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
    Y = initialSelection(x, k)  # Starting centroids using KMeans++ initialization
    for i in range(MaxIter):  # Repeating until convergence
        clusters = assignClustersIds(x, k, Y)  # Assign points to clusters
        Y_new = computeClusterRepresentratives(clusters, k)  # Update centroids
        if np.all(clusters == assignClustersIds(x, k, Y_new)):  # Check for convergence
            Y = Y_new
            break
        Y = Y_new  # Update centroids and repeat
    return Y, clusters


def computeSillhouette(x, clusters):
    """
    Input:
      - x: numpy array of data points.
      - clusters: numpy array of cluster assignments.
    Output:
      - score: float, the mean silhouette coefficient for the clustering.
    """
    c = []  # silhouette coefficients
    for i in range(x.shape[0]):  # loop over all data points
        d = np.linalg.norm(x - x[i], axis=1)  # Distance of points to point i
        # a_i: Cohesion (mean distance to other points in the same cluster)
        clstr = clusters == clusters[i]  # identify points in the same cluster
        clstr[i] = False  # Exclude the distance from the point to itself
        a_i = np.mean(d[clstr]) if np.sum(clstr) > 0 else 0
        # b_i: Separation (smallest mean distance to points in any other cluster)
        b_i = np.inf  # initialize b_i to infinity
        for cl in np.unique(clusters):  # for each unique cluster
            if cl == clusters[i]:
                continue  # skip the same cluster
            mean_dist = np.mean(d[clusters == cl])
            if mean_dist < b_i:
                b_i = mean_dist
        # silhouette coefficient for point i
        sc_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        c.append(sc_i)
    score = np.mean(c)  # mean silhouette coefficient
    return score


def plotShilhouettee(x, K, max_iter=100):
    """
    Input:
      - x: numpy array of data points.
      - K: iterable of integers, different numbers of clusters to try.
      - max_iter: integer, maximum number of iterations for KMeans.
    Output:
      - Saves a plot of Silhouette Score vs K as "Q3Plot.png".
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
    plt.title("Silhouette Score vs K (K++ ; dataset)")
    plt.savefig("Q3Plot.png")


def clustername(x, k):
    """
    Input:
      - x: numpy array of data points.
      - k: integer, number of clusters.
    Output:
      - names: list of strings with the cluster name (e.g., "Cluster 0") for each data point.
    """
    # Run KMeans on x with k clusters (using max_iter=100)
    _, clusters = KMeans(x, k, 100)
    # Create a list of names for each data point based on its cluster assignment
    names = ["Cluster " + str(label) for label in clusters]
    return names


K = range(1, 10)
plotShilhouettee(x, K)
