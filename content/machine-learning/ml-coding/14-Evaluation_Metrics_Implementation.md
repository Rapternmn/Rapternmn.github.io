+++
title = "Evaluation Metrics"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 14
description = "Complete implementation of evaluation metrics from scratch using Python and NumPy. Covers classification metrics (accuracy, precision, recall, F1, ROC-AUC), regression metrics (MSE, MAE, R²), and cross-validation techniques."
+++

---

## Introduction

Evaluation metrics are essential tools for assessing the performance of machine learning models. Different metrics are appropriate for different tasks - classification requires different metrics than regression, and imbalanced datasets need special consideration.

In this guide, we'll implement various evaluation metrics from scratch using Python and NumPy, covering classification, regression, and cross-validation techniques.

---

## Classification Metrics

### Confusion Matrix

The foundation for many classification metrics:

```
                Predicted
              Negative  Positive
Actual Negative   TN      FP
       Positive   FN      TP
```

Where:
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

---

## Implementation 1: Classification Metrics

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class ClassificationMetrics:
    """Implementation of classification evaluation metrics."""
    
    def __init__(self, y_true, y_pred):
        """
        Initialize with true and predicted labels.
        
        Parameters:
        -----------
        y_true : numpy array
            True class labels
        y_pred : numpy array
            Predicted class labels
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.confusion_matrix = self._compute_confusion_matrix()
    
    def _compute_confusion_matrix(self):
        """Compute confusion matrix."""
        classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        n_classes = len(classes)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(self.y_true)):
            true_idx = np.where(classes == self.y_true[i])[0][0]
            pred_idx = np.where(classes == self.y_pred[i])[0][0]
            cm[true_idx, pred_idx] += 1
        
        return cm
    
    def accuracy(self):
        """
        Calculate accuracy.
        
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        """
        return np.mean(self.y_true == self.y_pred)
    
    def precision(self, class_idx=1):
        """
        Calculate precision for a specific class.
        
        Precision = TP / (TP + FP)
        
        Parameters:
        -----------
        class_idx : int
            Index of class (0 for negative, 1 for positive in binary)
        
        Returns:
        --------
        precision : float
            Precision score
        """
        if self.confusion_matrix.shape[0] == 2:
            # Binary classification
            if class_idx == 1:
                TP = self.confusion_matrix[1, 1]
                FP = self.confusion_matrix[0, 1]
            else:
                TP = self.confusion_matrix[0, 0]
                FP = self.confusion_matrix[1, 0]
            
            denominator = TP + FP
            return TP / denominator if denominator > 0 else 0.0
        else:
            # Multi-class: calculate for specific class
            TP = self.confusion_matrix[class_idx, class_idx]
            FP = np.sum(self.confusion_matrix[:, class_idx]) - TP
            denominator = TP + FP
            return TP / denominator if denominator > 0 else 0.0
    
    def recall(self, class_idx=1):
        """
        Calculate recall (sensitivity) for a specific class.
        
        Recall = TP / (TP + FN)
        
        Parameters:
        -----------
        class_idx : int
            Index of class
        
        Returns:
        --------
        recall : float
            Recall score
        """
        if self.confusion_matrix.shape[0] == 2:
            # Binary classification
            if class_idx == 1:
                TP = self.confusion_matrix[1, 1]
                FN = self.confusion_matrix[1, 0]
            else:
                TP = self.confusion_matrix[0, 0]
                FN = self.confusion_matrix[0, 1]
            
            denominator = TP + FN
            return TP / denominator if denominator > 0 else 0.0
        else:
            # Multi-class
            TP = self.confusion_matrix[class_idx, class_idx]
            FN = np.sum(self.confusion_matrix[class_idx, :]) - TP
            denominator = TP + FN
            return TP / denominator if denominator > 0 else 0.0
    
    def specificity(self):
        """
        Calculate specificity (True Negative Rate).
        
        Specificity = TN / (TN + FP)
        """
        if self.confusion_matrix.shape[0] != 2:
            raise ValueError("Specificity only defined for binary classification")
        
        TN = self.confusion_matrix[0, 0]
        FP = self.confusion_matrix[0, 1]
        denominator = TN + FP
        return TN / denominator if denominator > 0 else 0.0
    
    def f1_score(self, class_idx=1):
        """
        Calculate F1 score.
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Parameters:
        -----------
        class_idx : int
            Index of class
        
        Returns:
        --------
        f1 : float
            F1 score
        """
        precision = self.precision(class_idx)
        recall = self.recall(class_idx)
        
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def f_beta_score(self, beta, class_idx=1):
        """
        Calculate F-beta score.
        
        F-beta = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
        
        Parameters:
        -----------
        beta : float
            Beta parameter (beta > 1 emphasizes recall, < 1 emphasizes precision)
        class_idx : int
            Index of class
        
        Returns:
        --------
        f_beta : float
            F-beta score
        """
        precision = self.precision(class_idx)
        recall = self.recall(class_idx)
        
        denominator = (beta ** 2) * precision + recall
        return (1 + beta ** 2) * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def classification_report(self):
        """Generate classification report."""
        classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        n_classes = len(classes)
        
        print("Classification Report:")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 40)
        
        for i, cls in enumerate(classes):
            prec = self.precision(i)
            rec = self.recall(i)
            f1 = self.f1_score(i)
            print(f"{cls:<10} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
        
        # Overall accuracy
        acc = self.accuracy()
        print("-" * 40)
        print(f"{'Accuracy':<10} {acc:<10.4f}")


class ROC_AUC:
    """ROC Curve and AUC calculation."""
    
    def __init__(self, y_true, y_scores):
        """
        Initialize with true labels and prediction scores.
        
        Parameters:
        -----------
        y_true : numpy array
            True binary labels (0 or 1)
        y_scores : numpy array
            Prediction scores (probabilities)
        """
        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)
    
    def roc_curve(self):
        """
        Calculate ROC curve.
        
        Returns:
        --------
        fpr : numpy array
            False Positive Rates
        tpr : numpy array
            True Positive Rates
        thresholds : numpy array
            Threshold values
        """
        # Sort by scores (descending)
        sorted_indices = np.argsort(self.y_scores)[::-1]
        y_true_sorted = self.y_true[sorted_indices]
        y_scores_sorted = self.y_scores[sorted_indices]
        
        # Count positives and negatives
        n_positive = np.sum(self.y_true == 1)
        n_negative = np.sum(self.y_true == 0)
        
        if n_positive == 0 or n_negative == 0:
            raise ValueError("ROC curve requires both positive and negative samples")
        
        # Initialize
        tpr = [0.0]
        fpr = [0.0]
        thresholds = [y_scores_sorted[0] + 1]  # Start above max score
        
        TP = 0
        FP = 0
        
        # Calculate TPR and FPR for each threshold
        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == 1:
                TP += 1
            else:
                FP += 1
            
            tpr.append(TP / n_positive)
            fpr.append(FP / n_negative)
            thresholds.append(y_scores_sorted[i])
        
        # Add endpoint
        tpr.append(1.0)
        fpr.append(1.0)
        thresholds.append(y_scores_sorted[-1] - 1)
        
        return np.array(fpr), np.array(tpr), np.array(thresholds)
    
    def auc(self):
        """
        Calculate Area Under ROC Curve (AUC).
        
        Uses trapezoidal rule for integration.
        
        Returns:
        --------
        auc : float
            AUC score (0 to 1)
        """
        fpr, tpr, _ = self.roc_curve()
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        fpr, tpr, _ = self.roc_curve()
        auc_score = self.auc()
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()


class PrecisionRecallCurve:
    """Precision-Recall curve calculation."""
    
    def __init__(self, y_true, y_scores):
        """
        Initialize with true labels and scores.
        
        Parameters:
        -----------
        y_true : numpy array
            True binary labels
        y_scores : numpy array
            Prediction scores
        """
        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)
    
    def precision_recall_curve(self):
        """
        Calculate precision-recall curve.
        
        Returns:
        --------
        precision : numpy array
            Precision values
        recall : numpy array
            Recall values
        thresholds : numpy array
            Threshold values
        """
        sorted_indices = np.argsort(self.y_scores)[::-1]
        y_true_sorted = self.y_true[sorted_indices]
        y_scores_sorted = self.y_scores[sorted_indices]
        
        n_positive = np.sum(self.y_true == 1)
        
        precision = []
        recall = []
        thresholds = [y_scores_sorted[0] + 1]
        
        TP = 0
        FP = 0
        
        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == 1:
                TP += 1
            else:
                FP += 1
            
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0
            rec = TP / n_positive if n_positive > 0 else 0
            
            precision.append(prec)
            recall.append(rec)
            thresholds.append(y_scores_sorted[i])
        
        return np.array(precision), np.array(recall), np.array(thresholds)
    
    def average_precision(self):
        """
        Calculate Average Precision (AP).
        
        Returns:
        --------
        ap : float
            Average precision score
        """
        precision, recall, _ = self.precision_recall_curve()
        
        # Calculate AP using trapezoidal rule
        ap = 0.0
        for i in range(1, len(recall)):
            ap += (recall[i] - recall[i-1]) * precision[i]
        
        return ap
```

---

## Implementation 2: Regression Metrics

```python
class RegressionMetrics:
    """Implementation of regression evaluation metrics."""
    
    def __init__(self, y_true, y_pred):
        """
        Initialize with true and predicted values.
        
        Parameters:
        -----------
        y_true : numpy array
            True target values
        y_pred : numpy array
            Predicted target values
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
    
    def mse(self):
        """
        Calculate Mean Squared Error.
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        """
        return np.mean((self.y_true - self.y_pred) ** 2)
    
    def rmse(self):
        """
        Calculate Root Mean Squared Error.
        
        RMSE = √MSE
        """
        return np.sqrt(self.mse())
    
    def mae(self):
        """
        Calculate Mean Absolute Error.
        
        MAE = (1/n) * Σ|y_true - y_pred|
        """
        return np.mean(np.abs(self.y_true - self.y_pred))
    
    def r2_score(self):
        """
        Calculate R² (Coefficient of Determination).
        
        R² = 1 - (SS_res / SS_tot)
        where:
        SS_res = Σ(y_true - y_pred)²
        SS_tot = Σ(y_true - y_mean)²
        """
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def adjusted_r2_score(self, n_features):
        """
        Calculate Adjusted R².
        
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
        where n = samples, k = features
        
        Parameters:
        -----------
        n_features : int
            Number of features used in model
        """
        r2 = self.r2_score()
        n = len(self.y_true)
        
        if n - n_features - 1 <= 0:
            return r2
        
        return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    
    def mean_absolute_percentage_error(self):
        """
        Calculate Mean Absolute Percentage Error.
        
        MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
        """
        epsilon = 1e-10  # Avoid division by zero
        return 100 * np.mean(np.abs((self.y_true - self.y_pred) / (self.y_true + epsilon)))
    
    def report(self):
        """Print regression metrics report."""
        print("Regression Metrics:")
        print(f"  MSE:  {self.mse():.4f}")
        print(f"  RMSE: {self.rmse():.4f}")
        print(f"  MAE:  {self.mae():.4f}")
        print(f"  R²:   {self.r2_score():.4f}")
```

---

## Implementation 3: Cross-Validation

```python
class CrossValidation:
    """Implementation of cross-validation techniques."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        """
        Initialize cross-validator.
        
        Parameters:
        -----------
        n_splits : int
            Number of folds
        shuffle : bool
            Whether to shuffle data before splitting
        random_state : int
            Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def k_fold_split(self, X, y):
        """
        Generate K-fold splits.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        
        Yields:
        -------
        train_indices : numpy array
            Training set indices
        test_indices : numpy array
            Test set indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            yield train_indices, test_indices
    
    def cross_val_score(self, X, y, model, scoring='accuracy'):
        """
        Perform cross-validation and return scores.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Target vector
        model : object
            Model with fit() and predict() methods
        scoring : str
            Scoring metric: 'accuracy', 'mse', 'r2'
        
        Returns:
        --------
        scores : numpy array
            Scores for each fold
        """
        scores = []
        
        for train_idx, test_idx in self.k_fold_split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate score
            if scoring == 'accuracy':
                score = np.mean(y_pred == y_test)
            elif scoring == 'mse':
                score = np.mean((y_test - y_pred) ** 2)
            elif scoring == 'r2':
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                raise ValueError(f"Unknown scoring: {scoring}")
            
            scores.append(score)
        
        return np.array(scores)
    
    def stratified_k_fold_split(self, X, y):
        """
        Generate stratified K-fold splits (for classification).
        
        Maintains class distribution in each fold.
        
        Parameters:
        -----------
        X : numpy array
            Feature matrix
        y : numpy array
            Class labels
        
        Yields:
        -------
        train_indices : numpy array
            Training set indices
        test_indices : numpy array
            Test set indices
        """
        classes = np.unique(y)
        n_samples = len(X)
        
        # Group indices by class
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for cls in classes:
                np.random.shuffle(class_indices[cls])
        
        fold_indices = [[] for _ in range(self.n_splits)]
        
        # Distribute samples from each class across folds
        for cls in classes:
            cls_indices = class_indices[cls]
            fold_size = len(cls_indices) // self.n_splits
            
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else len(cls_indices)
                fold_indices[i].extend(cls_indices[start:end])
        
        # Generate splits
        for i in range(self.n_splits):
            test_indices = np.array(fold_indices[i])
            train_indices = np.concatenate([fold_indices[j] for j in range(self.n_splits) if j != i])
            train_indices = np.array(train_indices)
            
            yield train_indices, test_indices
```

---

## Usage Examples

```python
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Example 1: Classification Metrics
print("=== Classification Metrics ===")
X_clf, y_clf = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Train a simple model (example)
from sklearn.linear_model import LogisticRegression
model_clf = LogisticRegression()
model_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = model_clf.predict(X_test_clf)
y_proba_clf = model_clf.predict_proba(X_test_clf)[:, 1]

# Calculate metrics
metrics = ClassificationMetrics(y_test_clf, y_pred_clf)
print(f"Accuracy: {metrics.accuracy():.4f}")
print(f"Precision: {metrics.precision():.4f}")
print(f"Recall: {metrics.recall():.4f}")
print(f"F1-Score: {metrics.f1_score():.4f}")
metrics.classification_report()

# ROC-AUC
roc = ROC_AUC(y_test_clf, y_proba_clf)
auc_score = roc.auc()
print(f"\nAUC Score: {auc_score:.4f}")

# Example 2: Regression Metrics
print("\n=== Regression Metrics ===")
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=5,
    noise=10,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LinearRegression
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

# Calculate metrics
reg_metrics = RegressionMetrics(y_test_reg, y_pred_reg)
reg_metrics.report()

# Example 3: Cross-Validation
print("\n=== Cross-Validation ===")
cv = CrossValidation(n_splits=5, shuffle=True, random_state=42)
scores = cv.cross_val_score(X_clf, y_clf, model_clf, scoring='accuracy')
print(f"CV Scores: {scores}")
print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

---

## Key Takeaways

1. **Classification Metrics**:
   - Accuracy: Overall correctness
   - Precision: Of predicted positives, how many are correct
   - Recall: Of actual positives, how many were found
   - F1: Harmonic mean of precision and recall
   - ROC-AUC: Overall classifier performance across thresholds

2. **Regression Metrics**:
   - MSE/RMSE: Penalizes large errors
   - MAE: Average error magnitude
   - R²: Proportion of variance explained

3. **Cross-Validation**: Provides robust performance estimates

4. **Metric Selection**: Choose based on problem type and class imbalance

---

## When to Use Each Metric

| Metric | Use Case |
|--------|----------|
| **Accuracy** | Balanced classes, equal cost of errors |
| **Precision** | High cost of false positives (spam detection) |
| **Recall** | High cost of false negatives (disease detection) |
| **F1-Score** | Balance between precision and recall |
| **ROC-AUC** | Overall classifier performance, class imbalance |
| **MSE/RMSE** | Regression, penalize large errors |
| **MAE** | Regression, equal weight to all errors |
| **R²** | Regression, proportion of variance explained |

---

## Interview Tips

When discussing evaluation metrics in interviews:

1. **Confusion Matrix**: Foundation for classification metrics
2. **Precision vs Recall**: Trade-off and when to prioritize each
3. **ROC-AUC**: Understand what it measures and when to use it
4. **Cross-Validation**: Why it's important and different types
5. **Class Imbalance**: How metrics behave with imbalanced data
6. **Regression Metrics**: Understand what each measures
7. **Metric Selection**: Justify metric choice for specific problems

---

## References

* Confusion matrix and classification metrics
* ROC curve and AUC calculation
* Precision-recall curves
* Cross-validation techniques
* Regression evaluation metrics

