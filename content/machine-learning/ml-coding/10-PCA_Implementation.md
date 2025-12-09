+++
title = "Principal Component Analysis (PCA)"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Complete implementation of Principal Component Analysis from scratch using Python and NumPy. Covers eigenvalue decomposition, variance maximization, dimensionality reduction, and practical examples."
+++

---

## Introduction

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique that finds the directions of maximum variance in high-dimensional data. It projects data onto a lower-dimensional space while preserving as much information as possible.

In this guide, we'll implement PCA from scratch using Python and NumPy, covering eigenvalue decomposition, variance maximization, and dimensionality reduction.

---

## Mathematical Foundation

### Goal

Find a linear transformation that:
1. Maximizes variance in the projected space
2. Minimizes reconstruction error
3. Removes correlation between features

### Covariance Matrix

For centered data (mean = 0), the covariance matrix is:

```
C = (1/(m-1)) * XᵀX
```

Where:
- `X` is the data matrix of shape (m, n)
- `m` is the number of samples
- `n` is the number of features

### Eigenvalue Decomposition

PCA finds the eigenvectors and eigenvalues of the covariance matrix:

```
C * v = λ * v
```

Where:
- `v` are eigenvectors (principal components)
- `λ` are eigenvalues (variance along each component)

### Principal Components

The principal components are the eigenvectors sorted by decreasing eigenvalues:

```
PC₁, PC₂, ..., PCₙ
```

Where `PC₁` has the highest variance.

### Projection

To reduce dimensionality to `k` dimensions:

```
X_reduced = X * W
```

Where `W` is a matrix of the first `k` principal components (shape: n × k).

### Explained Variance

The proportion of variance explained by component `i`:

```
explained_variance_ratio[i] = λᵢ / Σλⱼ
```

### Reconstruction

To reconstruct original data from reduced representation:

```
X_reconstructed = X_reduced * Wᵀ
```

---

## Implementation 1: Basic PCA

```python
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components=None):
        """
        Initialize PCA.
        
        Parameters:
        -----------
        n_components : int or None
            Number of components to keep. If None, keeps all components.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
    
    def fit(self, X):
        """
        Fit PCA on the data.
        
        Parameters:
        -----------
        X : numpy array
            Data matrix of shape (m, n)
        """
        # Center the data (subtract mean)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Calculate covariance matrix
        # C = (1/(m-1)) * XᵀX
        m, n = X_centered.shape
        cov_matrix = (1 / (m - 1)) * np.dot(X_centered.T, X_centered)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store components
        if self.n_components is None:
            self.n_components = n
        
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)
        self.singular_values_ = np.sqrt(self.explained_variance_ * (m - 1))
        
        return self
    
    def transform(self, X):
        """
        Transform data to lower-dimensional space.
        
        Parameters:
        -----------
        X : numpy array
            Data matrix of shape (m, n)
        
        Returns:
        --------
        X_transformed : numpy array
            Transformed data of shape (m, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        """
        Fit PCA and transform data.
        
        Parameters:
        -----------
        X : numpy array
            Data matrix of shape (m, n)
        
        Returns:
        --------
        X_transformed : numpy array
            Transformed data of shape (m, n_components)
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Reconstruct original data from transformed data.
        
        Parameters:
        -----------
        X_transformed : numpy array
            Transformed data of shape (m, n_components)
        
        Returns:
        --------
        X_reconstructed : numpy array
            Reconstructed data of shape (m, n)
        """
        X_reconstructed = np.dot(X_transformed, self.components_.T)
        return X_reconstructed + self.mean_
    
    def get_cumulative_variance(self):
        """
        Get cumulative explained variance ratio.
        
        Returns:
        --------
        cumulative_variance : numpy array
            Cumulative explained variance ratio
        """
        return np.cumsum(self.explained_variance_ratio_)
```

### Usage Example

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {pca.get_cumulative_variance()}")

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], label='Setosa', alpha=0.7)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label='Versicolor', alpha=0.7)
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], label='Virginica', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Iris Dataset')
plt.legend()
plt.grid(True)

# Plot explained variance
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Implementation 2: PCA using SVD (More Numerically Stable)

Singular Value Decomposition (SVD) is more numerically stable than eigenvalue decomposition:

```python
class PCA_SVD:
    def __init__(self, n_components=None):
        """
        Initialize PCA using SVD.
        
        Parameters:
        -----------
        n_components : int or None
            Number of components to keep
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
    
    def fit(self, X):
        """
        Fit PCA using SVD.
        
        Parameters:
        -----------
        X : numpy array
            Data matrix of shape (m, n)
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD: X = U * Σ * Vᵀ
        # U: left singular vectors (m × m)
        # Σ: singular values (diagonal matrix)
        # Vᵀ: right singular vectors (n × n)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Principal components are the right singular vectors (Vᵀ)
        # Sort by singular values (already sorted in descending order)
        if self.n_components is None:
            self.n_components = min(X.shape[1], X.shape[0])
        
        self.components_ = Vt[:self.n_components].T
        self.singular_values_ = s[:self.n_components]
        
        # Explained variance from singular values
        m = X.shape[0]
        self.explained_variance_ = (self.singular_values_ ** 2) / (m - 1)
        total_variance = np.sum(s ** 2) / (m - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """Transform data to lower-dimensional space."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        """Fit and transform."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """Reconstruct original data."""
        X_reconstructed = np.dot(X_transformed, self.components_.T)
        return X_reconstructed + self.mean_
```

---

## Implementation 3: PCA with Variance Threshold

Select components that explain a certain percentage of variance:

```python
class PCA_VarianceThreshold:
    def __init__(self, variance_threshold=0.95):
        """
        Initialize PCA with variance threshold.
        
        Parameters:
        -----------
        variance_threshold : float
            Minimum cumulative variance to retain (0 to 1)
        """
        self.variance_threshold = variance_threshold
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
    
    def fit(self, X):
        """
        Fit PCA and select components based on variance threshold.
        
        Parameters:
        -----------
        X : numpy array
            Data matrix of shape (m, n)
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Calculate explained variance
        m = X.shape[0]
        explained_variance = (s ** 2) / (m - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find number of components needed
        self.n_components_ = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        
        # Store components
        self.components_ = Vt[:self.n_components_].T
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        
        print(f"Selected {self.n_components_} components to explain "
              f"{cumulative_variance[self.n_components_ - 1]:.2%} of variance")
        
        return self
    
    def transform(self, X):
        """Transform data."""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        """Fit and transform."""
        return self.fit(X).transform(X)
```

---

## Complete Example: Dimensionality Reduction and Reconstruction

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load digits dataset (64 features = 8x8 pixels)
digits = load_digits()
X = digits.data
y = digits.target

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with different numbers of components
n_components_list = [2, 10, 30, 64]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, n_comp in enumerate(n_components_list):
    # Fit PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    # Reconstruct
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = scaler.inverse_transform(X_reconstructed)
    
    # Visualize original and reconstructed
    axes[0, idx].imshow(X[0].reshape(8, 8), cmap='gray')
    axes[0, idx].set_title(f'Original (64 features)')
    axes[0, idx].axis('off')
    
    axes[1, idx].imshow(X_reconstructed[0].reshape(8, 8), cmap='gray')
    explained_var = np.sum(pca.explained_variance_ratio_)
    axes[1, idx].set_title(f'Reconstructed ({n_comp} components, {explained_var:.1%} variance)')
    axes[1, idx].axis('off')

plt.tight_layout()
plt.show()

# Plot explained variance
pca_full = PCA(n_components=None)
pca_full.fit(X_scaled)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         pca_full.explained_variance_ratio_, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Component')
plt.grid(True)

plt.subplot(1, 2, 2)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **Centering is Essential**: Always center (subtract mean) before applying PCA
2. **Standardization**: Often beneficial to standardize features before PCA
3. **Variance Maximization**: PCA finds directions of maximum variance
4. **Orthogonal Components**: Principal components are orthogonal (uncorrelated)
5. **Information Loss**: Reducing dimensions always loses some information
6. **SVD vs Eigenvalue Decomposition**: SVD is more numerically stable
7. **Explained Variance**: Use cumulative variance to choose number of components

---

## When to Use PCA

| Scenario | Suitable? | Notes |
|----------|-----------|-------|
| **High-dimensional data** | ✅ Yes | Reduces dimensionality effectively |
| **Multicollinearity** | ✅ Yes | Removes correlation between features |
| **Visualization** | ✅ Yes | Reduce to 2D/3D for visualization |
| **Noise reduction** | ✅ Yes | Removes low-variance (noisy) components |
| **Feature extraction** | ✅ Yes | Creates new uncorrelated features |
| **Interpretability** | ❌ No | Principal components are hard to interpret |
| **Non-linear relationships** | ❌ No | Only captures linear relationships |
| **Sparse data** | ⚠️ Maybe | Consider sparse PCA variants |

---

## Interview Tips

When implementing PCA in interviews:

1. **Explain Variance Maximization**: PCA finds directions of maximum variance
2. **Covariance Matrix**: Understand how covariance matrix relates to variance
3. **Eigenvalue Decomposition**: Know the mathematical foundation
4. **SVD Alternative**: Mention SVD as more stable approach
5. **Centering**: Always emphasize the importance of centering data
6. **Dimensionality Selection**: Explain how to choose number of components
7. **Limitations**: Mention linearity assumption and interpretability issues
8. **Applications**: Image compression, feature extraction, visualization

---

## Time Complexity

- **Eigenvalue Decomposition**: O(n³) where n = number of features
- **SVD**: O(min(m²n, mn²)) where m = samples, n = features
- **Transform**: O(m * n * k) where k = number of components
- **Space Complexity**: O(n²) for covariance matrix, O(n * k) for components

---

## Advantages

1. **Dimensionality Reduction**: Reduces number of features
2. **Noise Reduction**: Removes low-variance components
3. **Uncorrelated Features**: Creates orthogonal components
4. **Linear Transformation**: Simple and interpretable mathematically
5. **No Labels Required**: Unsupervised method

---

## Disadvantages

1. **Linearity Assumption**: Only captures linear relationships
2. **Interpretability**: Principal components are hard to interpret
3. **Scale Dependent**: Sensitive to feature scaling
4. **Information Loss**: Reducing dimensions loses information
5. **Computational Cost**: Expensive for very large datasets

---

## Applications

1. **Image Compression**: Reduce image dimensions while preserving quality
2. **Face Recognition**: Eigenfaces using PCA
3. **Gene Expression Analysis**: Reduce thousands of genes to few components
4. **Data Visualization**: Project high-dimensional data to 2D/3D
5. **Feature Engineering**: Create new features from original ones
6. **Noise Filtering**: Remove components with low variance

---

## References

* Eigenvalue decomposition and singular value decomposition
* Variance maximization and information theory
* Dimensionality reduction techniques
* Covariance and correlation matrices
* Linear algebra fundamentals

