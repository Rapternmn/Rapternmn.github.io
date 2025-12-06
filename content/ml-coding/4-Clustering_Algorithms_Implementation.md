+++
title = "Clustering Algorithms"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Complete implementation of clustering algorithms from scratch using Python and NumPy. Covers K-Means, K-Means++, hierarchical clustering, DBSCAN, distance metrics, and evaluation methods."
+++

---

## Introduction

Clustering is an unsupervised learning technique that groups similar data points together. Unlike classification, clustering doesn't require labeled data - it discovers patterns and structures in the data automatically.

In this guide, we'll implement several clustering algorithms from scratch using Python and NumPy, including K-Means, K-Means++, and hierarchical clustering.

---

## Mathematical Foundation

### Distance Metrics

#### Euclidean Distance

```
d(x, y) = √(Σ(xᵢ - yᵢ)²)
```

Most common distance metric for continuous features.

#### Manhattan Distance

```
d(x, y) = Σ|xᵢ - yᵢ|
```

Also called L1 distance, less sensitive to outliers.

#### Cosine Distance

```
d(x, y) = 1 - (x · y) / (||x|| * ||y||)
```

Measures angle between vectors, good for high-dimensional data.

#### Hamming Distance

```
d(x, y) = Σ(xᵢ ≠ yᵢ)
```

For categorical/binary data - counts mismatches.

### K-Means Objective

Minimize within-cluster sum of squares (WCSS):

```
J = Σᵢ Σⱼ ||xᵢ - μⱼ||²
```

Where:
- `xᵢ` is data point `i`
- `μⱼ` is centroid of cluster `j`
- Sum over all points `i` assigned to cluster `j`

---

## Implementation 1: K-Means Clustering

### Basic K-Means

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        """
        Initialize K-Means clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (default: 3)
        max_iters : int
            Maximum number of iterations (default: 100)
        tol : float
            Tolerance for convergence (default: 1e-4)
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None  # Within-cluster sum of squares
    
    def _euclidean_distance(self, x, y):
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _initialize_centroids(self, X):
        """
        Initialize centroids randomly.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        
        Returns:
        --------
        centroids : numpy array
            Initial centroids of shape (k, n)
        """
        np.random.seed(self.random_state)
        m, n = X.shape
        
        # Randomly select k data points as initial centroids
        indices = np.random.choice(m, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        centroids : numpy array
            Current centroids
        
        Returns:
        --------
        labels : numpy array
            Cluster assignments for each point
        """
        m = X.shape[0]
        labels = np.zeros(m, dtype=int)
        
        for i in range(m):
            # Calculate distances to all centroids
            distances = [self._euclidean_distance(X[i], centroid) 
                        for centroid in centroids]
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids to be the mean of points in each cluster.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        labels : numpy array
            Cluster assignments
        
        Returns:
        --------
        centroids : numpy array
            Updated centroids
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Get all points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid as mean of cluster points
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep previous centroid or reinitialize
                centroids[k] = np.random.randn(n_features)
        
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """
        Calculate within-cluster sum of squares (inertia).
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        labels : numpy array
            Cluster assignments
        centroids : numpy array
            Centroids
        
        Returns:
        --------
        inertia : float
            Within-cluster sum of squares
        """
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means clustering to data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        # Iterate until convergence
        for iteration in range(self.max_iters):
            # Assign points to nearest centroids
            self.labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, self.labels)
            
            # Check for convergence
            centroid_shift = np.sum([self._euclidean_distance(self.centroids[k], new_centroids[k])
                                    for k in range(self.n_clusters)])
            
            self.centroids = new_centroids
            
            # Calculate inertia
            self.inertia_ = self._calculate_inertia(X, self.labels, self.centroids)
            
            if centroid_shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict(self, X):
        """
        Predict cluster assignments for new data.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        labels : numpy array
            Cluster assignments
        """
        return self._assign_clusters(X, self.centroids)
```

### Usage Example

```python
# Generate sample data
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    random_state=42,
    cluster_std=0.60
)

# Create and fit K-Means
kmeans = KMeans(n_clusters=4, max_iters=100, random_state=42)
kmeans.fit(X)

# Get predictions
labels = kmeans.labels_

# Visualize results
plt.figure(figsize=(12, 5))

# Plot 1: True clusters
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot 2: Predicted clusters
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title(f'K-Means Clusters (Inertia: {kmeans.inertia_:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
```

---

## Implementation 2: K-Means++ Initialization

K-Means++ improves initialization by selecting centroids that are far apart, leading to better convergence.

```python
class KMeansPlusPlus(KMeans):
    """
    K-Means with K-Means++ initialization.
    """
    def _initialize_centroids(self, X):
        """
        Initialize centroids using K-Means++ algorithm.
        
        Steps:
        1. Choose first centroid randomly
        2. For each remaining centroid:
           - Calculate distance from each point to nearest existing centroid
           - Choose next centroid with probability proportional to distance²
        """
        np.random.seed(self.random_state)
        m, n = X.shape
        
        # Initialize first centroid randomly
        centroids = np.zeros((self.n_clusters, n))
        centroids[0] = X[np.random.randint(m)]
        
        # Select remaining centroids
        for k in range(1, self.n_clusters):
            # Calculate distances from each point to nearest existing centroid
            distances = np.zeros(m)
            for i in range(m):
                # Find minimum distance to any existing centroid
                min_dist = min([self._euclidean_distance(X[i], centroids[j]) 
                               for j in range(k)])
                distances[i] = min_dist ** 2  # Square the distance
            
            # Choose next centroid with probability proportional to distance²
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            
            # Find index where cumulative probability exceeds random value
            next_centroid_idx = np.searchsorted(cumulative_probs, r)
            centroids[k] = X[next_centroid_idx]
        
        return centroids
```

### Usage Example

```python
# Compare K-Means vs K-Means++
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                       random_state=42, cluster_std=0.60)

# Standard K-Means
kmeans_standard = KMeans(n_clusters=4, max_iters=100, random_state=42)
kmeans_standard.fit(X)

# K-Means++
kmeans_plus = KMeansPlusPlus(n_clusters=4, max_iters=100, random_state=42)
kmeans_plus.fit(X)

print(f"Standard K-Means Inertia: {kmeans_standard.inertia_:.2f}")
print(f"K-Means++ Inertia: {kmeans_plus.inertia_:.2f}")
```

---

## Implementation 3: Hierarchical Clustering (Agglomerative)

Hierarchical clustering builds a tree of clusters by iteratively merging the closest clusters.

```python
class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='ward'):
        """
        Initialize Hierarchical Clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to form
        linkage : str
            Linkage criterion: 'ward', 'complete', 'average', 'single'
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.distance_matrix = None
    
    def _euclidean_distance(self, x, y):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        m = X.shape[0]
        distance_matrix = np.zeros((m, m))
        
        for i in range(m):
            for j in range(i + 1, m):
                dist = self._euclidean_distance(X[i], X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _ward_linkage(self, cluster1, cluster2, X):
        """
        Calculate Ward linkage distance between two clusters.
        
        Ward linkage minimizes within-cluster variance.
        """
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)
        
        # Ward distance
        n1 = len(cluster1)
        n2 = len(cluster2)
        ward_dist = (n1 * n2 / (n1 + n2)) * self._euclidean_distance(centroid1, centroid2) ** 2
        
        return ward_dist
    
    def _complete_linkage(self, cluster1, cluster2, X):
        """
        Complete linkage: maximum distance between any two points in clusters.
        """
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        max_dist = 0
        for p1 in points1:
            for p2 in points2:
                dist = self._euclidean_distance(p1, p2)
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def _average_linkage(self, cluster1, cluster2, X):
        """
        Average linkage: average distance between all pairs of points.
        """
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        total_dist = 0
        count = 0
        for p1 in points1:
            for p2 in points2:
                total_dist += self._euclidean_distance(p1, p2)
                count += 1
        
        return total_dist / count if count > 0 else 0
    
    def _single_linkage(self, cluster1, cluster2, X):
        """
        Single linkage: minimum distance between any two points in clusters.
        """
        points1 = X[cluster1]
        points2 = X[cluster2]
        
        min_dist = float('inf')
        for p1 in points1:
            for p2 in points2:
                dist = self._euclidean_distance(p1, p2)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_linkage(self, cluster1, cluster2, X):
        """Calculate linkage distance based on specified method."""
        if self.linkage == 'ward':
            return self._ward_linkage(cluster1, cluster2, X)
        elif self.linkage == 'complete':
            return self._complete_linkage(cluster1, cluster2, X)
        elif self.linkage == 'average':
            return self._average_linkage(cluster1, cluster2, X)
        elif self.linkage == 'single':
            return self._single_linkage(cluster1, cluster2, X)
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X):
        """
        Fit hierarchical clustering using agglomerative approach.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        """
        m = X.shape[0]
        
        # Start with each point as its own cluster
        clusters = [[i] for i in range(m)]
        
        # Continue merging until we have n_clusters
        while len(clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_i, merge_j = None, None
            
            # Find two closest clusters
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._calculate_linkage(clusters[i], clusters[j], X)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            if merge_i is not None and merge_j is not None:
                clusters[merge_i].extend(clusters[merge_j])
                clusters.pop(merge_j)
        
        # Assign labels
        self.labels = np.zeros(m, dtype=int)
        for label, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = label
        
        return self
    
    def predict(self, X):
        """
        Note: Hierarchical clustering doesn't have a direct predict method
        for new data. This would require retraining or using a different approach.
        For now, we'll return the labels from fit.
        """
        return self.labels
```

---

## Implementation 4: DBSCAN (Density-Based Clustering)

DBSCAN groups points that are closely packed together, marking outliers as noise.

```python
class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN clustering.
        
        Parameters:
        -----------
        eps : float
            Maximum distance between two samples to be considered neighbors
        min_samples : int
            Minimum number of samples in a neighborhood to form a cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
    
    def _euclidean_distance(self, x, y):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _get_neighbors(self, point_idx, X):
        """
        Find all neighbors of a point within eps distance.
        
        Parameters:
        -----------
        point_idx : int
            Index of the point
        X : numpy array
            Feature matrix
        
        Returns:
        --------
        neighbors : list
            Indices of neighboring points
        """
        neighbors = []
        for i in range(len(X)):
            if i != point_idx:
                dist = self._euclidean_distance(X[point_idx], X[i])
                if dist <= self.eps:
                    neighbors.append(i)
        return neighbors
    
    def fit(self, X):
        """
        Fit DBSCAN clustering.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix of shape (m, n)
        """
        m = X.shape[0]
        self.labels = np.full(m, -1)  # -1 means unvisited/noise
        cluster_id = 0
        
        for point_idx in range(m):
            # Skip if already processed
            if self.labels[point_idx] != -1:
                continue
            
            # Get neighbors
            neighbors = self._get_neighbors(point_idx, X)
            
            # Check if point is noise (fewer than min_samples neighbors)
            if len(neighbors) < self.min_samples:
                self.labels[point_idx] = -1  # Mark as noise
                continue
            
            # Start new cluster
            self.labels[point_idx] = cluster_id
            
            # Expand cluster
            seed_set = neighbors.copy()
            i = 0
            while i < len(seed_set):
                neighbor_idx = seed_set[i]
                
                # If neighbor was marked as noise, add to cluster
                if self.labels[neighbor_idx] == -1:
                    self.labels[neighbor_idx] = cluster_id
                
                # If neighbor is unvisited, process it
                if self.labels[neighbor_idx] == -1:
                    self.labels[neighbor_idx] = cluster_id
                    
                    # Get neighbors of this neighbor
                    neighbor_neighbors = self._get_neighbors(neighbor_idx, X)
                    
                    # If it's a core point, add its neighbors to seed set
                    if len(neighbor_neighbors) >= self.min_samples:
                        seed_set.extend(neighbor_neighbors)
                
                i += 1
            
            cluster_id += 1
        
        return self
    
    def predict(self, X):
        """Return cluster labels."""
        return self.labels
```

---

## Evaluation Metrics

### Silhouette Score

Measures how similar a point is to its own cluster compared to other clusters:

```python
def silhouette_score(X, labels):
    """
    Calculate silhouette score for clustering.
    
    Parameters:
    -----------
    X : numpy array
        Feature matrix
    labels : numpy array
        Cluster assignments
    
    Returns:
    --------
    score : float
        Average silhouette score (-1 to 1, higher is better)
    """
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))
    
    m = X.shape[0]
    silhouette_scores = np.zeros(m)
    
    for i in range(m):
        cluster_i = labels[i]
        
        # Calculate average distance to points in same cluster
        same_cluster_points = X[labels == cluster_i]
        if len(same_cluster_points) > 1:
            a_i = np.mean([euclidean_distance(X[i], point) 
                          for point in same_cluster_points if not np.array_equal(X[i], point)])
        else:
            a_i = 0
        
        # Calculate minimum average distance to other clusters
        other_clusters = [c for c in np.unique(labels) if c != cluster_i]
        if len(other_clusters) > 0:
            b_i_values = []
            for other_cluster in other_clusters:
                other_cluster_points = X[labels == other_cluster]
                avg_dist = np.mean([euclidean_distance(X[i], point) 
                                   for point in other_cluster_points])
                b_i_values.append(avg_dist)
            b_i = min(b_i_values)
        else:
            b_i = 0
        
        # Silhouette score for point i
        if max(a_i, b_i) > 0:
            silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_scores[i] = 0
    
    return np.mean(silhouette_scores)
```

### Adjusted Rand Index (ARI)

Measures similarity between two clusterings:

```python
def adjusted_rand_index(labels_true, labels_pred):
    """
    Calculate Adjusted Rand Index.
    
    Parameters:
    -----------
    labels_true : numpy array
        True cluster labels
    labels_pred : numpy array
        Predicted cluster labels
    
    Returns:
    --------
    ari : float
        Adjusted Rand Index (-1 to 1, 1 = perfect match)
    """
    from scipy.special import comb
    
    # Create contingency table
    n = len(labels_true)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    contingency = np.zeros((len(classes), len(clusters)))
    for i, class_label in enumerate(classes):
        for j, cluster_label in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == class_label) & 
                                      (labels_pred == cluster_label))
    
    # Calculate sums
    sum_comb_c = sum(comb(n_c, 2) for n_c in np.sum(contingency, axis=1))
    sum_comb_k = sum(comb(n_k, 2) for n_k in np.sum(contingency, axis=0))
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())
    
    # Calculate ARI
    expected_index = sum_comb_c * sum_comb_k / comb(n, 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    
    return ari
```

---

## Complete Example: Comparing Clustering Algorithms

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

# Generate different datasets
datasets = {
    'Blobs': make_blobs(n_samples=300, centers=4, n_features=2, 
                        random_state=42, cluster_std=0.60)[0],
    'Moons': make_moons(n_samples=300, noise=0.05, random_state=42)[0],
    'Circles': make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)[0]
}

# Test different algorithms
fig, axes = plt.subplots(len(datasets), 3, figsize=(15, 12))

for row, (dataset_name, X) in enumerate(datasets.items()):
    # K-Means
    kmeans = KMeans(n_clusters=4, max_iters=100, random_state=42)
    kmeans.fit(X)
    labels_kmeans = kmeans.labels_
    
    # K-Means++
    kmeans_plus = KMeansPlusPlus(n_clusters=4, max_iters=100, random_state=42)
    kmeans_plus.fit(X)
    labels_kmeans_plus = kmeans_plus.labels_
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X)
    labels_dbscan = dbscan.labels_
    
    # Plot results
    axes[row, 0].scatter(X[:, 0], X[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.6)
    axes[row, 0].set_title(f'K-Means - {dataset_name}')
    axes[row, 0].grid(True)
    
    axes[row, 1].scatter(X[:, 0], X[:, 1], c=labels_kmeans_plus, cmap='viridis', alpha=0.6)
    axes[row, 1].set_title(f'K-Means++ - {dataset_name}')
    axes[row, 1].grid(True)
    
    axes[row, 2].scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', alpha=0.6)
    axes[row, 2].set_title(f'DBSCAN - {dataset_name}')
    axes[row, 2].grid(True)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **K-Means**: 
   - Simple, fast, works well for spherical clusters
   - Sensitive to initialization
   - Requires specifying number of clusters

2. **K-Means++**: 
   - Better initialization than random
   - Often converges faster and to better solutions

3. **Hierarchical Clustering**: 
   - Creates dendrogram (tree structure)
   - No need to specify number of clusters upfront
   - Can be slow for large datasets

4. **DBSCAN**: 
   - Can find clusters of arbitrary shape
   - Identifies noise/outliers
   - Doesn't require number of clusters

---

## Interview Tips

1. **K-Means Algorithm**: Explain assignment and update steps
2. **Initialization**: Why K-Means++ is better than random
3. **Convergence**: How to detect when algorithm has converged
4. **Distance Metrics**: When to use Euclidean vs others
5. **Evaluation**: Silhouette score, inertia, ARI
6. **Limitations**: What types of clusters each algorithm can/can't handle

---

## Time Complexity

- **K-Means**: O(m * k * n * i) where m=samples, k=clusters, n=features, i=iterations
- **K-Means++**: O(m * k * n) for initialization, same as K-Means for iterations
- **Hierarchical**: O(m³) for naive implementation, O(m² log m) with optimizations
- **DBSCAN**: O(m²) worst case, O(m log m) with spatial indexing

---

## Choosing the Right Algorithm

| Algorithm | Best For | Limitations |
|-----------|----------|-------------|
| **K-Means** | Spherical clusters, known k | Sensitive to initialization, assumes clusters are similar size |
| **Hierarchical** | Unknown k, hierarchical structure | Slow for large datasets |
| **DBSCAN** | Arbitrary shapes, noise detection | Sensitive to eps and min_samples parameters |
| **GMM** | Overlapping clusters | Assumes Gaussian distribution |

---

## References

- K-Means algorithm and convergence
- K-Means++ initialization
- Hierarchical clustering and linkage methods
- DBSCAN density-based clustering
- Clustering evaluation metrics

