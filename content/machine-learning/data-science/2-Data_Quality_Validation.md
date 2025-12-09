+++
title = "Data Quality & Validation"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 9
description = "Comprehensive guide to data quality, validation, and data profiling. Covers data profiling, data validation, outlier detection, missing data handling, data drift detection, and schema validation."
+++


## 1. Data Quality Overview

### What is Data Quality?

**Definition**: Measure of how well data serves its intended purpose.

**Dimensions of Data Quality**:

1. **Completeness**: Percentage of non-missing values
2. **Accuracy**: Data correctly represents real-world entities
3. **Consistency**: Data is consistent across sources
4. **Validity**: Data conforms to defined rules
5. **Timeliness**: Data is up-to-date
6. **Uniqueness**: No duplicate records

### Why Data Quality Matters

- **Garbage In, Garbage Out**: Poor data → poor models
- **Business Impact**: Wrong decisions, lost revenue
- **Model Performance**: Data quality affects model accuracy
- **Trust**: Stakeholders need reliable data

---

## 2. Data Profiling

### What is Data Profiling?

**Definition**: Process of examining data to understand its structure, content, and quality.

### Profiling Techniques

#### 1. Statistical Summary

**For Numerical Columns**:
- Count, mean, median, mode
- Min, max, range
- Standard deviation, variance
- Percentiles (25th, 50th, 75th, 95th)
- Skewness, kurtosis

**For Categorical Columns**:
- Value counts
- Unique values
- Most frequent values
- Frequency distribution

#### 2. Data Type Validation

- Check expected vs actual types
- Detect type mismatches
- Identify mixed types

#### 3. Range Checks

- Min/max values
- Detect values outside expected range
- Identify impossible values (negative age, etc.)

#### 4. Pattern Detection

- Format validation (email, phone, date)
- Regular expression matching
- Identify anomalies in patterns

#### 5. Relationship Analysis

- Foreign key relationships
- Referential integrity
- Cross-column dependencies

---

## 3. Data Validation

### Validation Rules

#### 1. Type Validation

**Check**: Data matches expected type

**Examples**:
- Integer fields contain integers
- Date fields contain valid dates
- Boolean fields contain True/False

#### 2. Range Validation

**Check**: Values within acceptable range

**Examples**:
- Age: 0-150
- Percentage: 0-100
- Price: > 0

#### 3. Format Validation

**Check**: Data matches expected format

**Examples**:
- Email: `user@domain.com`
- Phone: `(XXX) XXX-XXXX`
- Date: `YYYY-MM-DD`

#### 4. Business Rule Validation

**Check**: Data follows business logic

**Examples**:
- Order date ≤ Ship date
- Total = Sum of line items
- Status transitions are valid

#### 5. Referential Integrity

**Check**: Foreign keys reference valid records

**Example**: User ID in orders table exists in users table

---

### Validation Framework

#### Schema Validation

**Definition**: Validate data structure matches schema.

**Tools**:
- JSON Schema
- Avro Schema
- Great Expectations
- Pandera

#### Data Quality Checks

1. **Completeness**: Check missing values
2. **Uniqueness**: Check duplicates
3. **Validity**: Check format, range, rules
4. **Consistency**: Check across sources
5. **Accuracy**: Check against ground truth (if available)

---

## 4. Outlier Detection

### What are Outliers?

**Definition**: Data points that deviate significantly from the rest.

**Types**:
- **Point Outliers**: Individual anomalous points
- **Contextual Outliers**: Anomalous in specific context
- **Collective Outliers**: Anomalous as a group

### Detection Methods

#### 1. Statistical Methods

**Z-Score**:
```
z = (x - μ) / σ
```

**Rule**: |z| > 3 → outlier

**IQR Method**:
- Q1, Q3: 25th and 75th percentiles
- IQR = Q3 - Q1
- Outliers: < Q1 - 1.5×IQR or > Q3 + 1.5×IQR

#### 2. Distance-Based Methods

**K-Nearest Neighbors**:
- Points with few nearby neighbors are outliers

**Local Outlier Factor (LOF)**:
- Measures local density deviation

#### 3. Isolation Forest

**Idea**: Outliers are easier to isolate.

**Method**: Random forests that isolate points.

**Advantages**: Handles high-dimensional data, no distribution assumptions.

#### 4. DBSCAN

**Clustering-based**: Points not in any cluster are outliers.

### Handling Outliers

**Options**:
1. **Remove**: If clearly errors
2. **Transform**: Log, winsorize
3. **Cap**: Set to min/max
4. **Investigate**: May be valid but rare
5. **Separate model**: Model outliers separately

**Decision**: Depends on whether outlier is error or valid rare case.

---

## 5. Missing Data Handling

### Types of Missingness

#### 1. MCAR (Missing Completely At Random)

**Definition**: Missingness independent of observed and unobserved data.

**Example**: Random data loss

**Impact**: Unbiased, easier to handle

#### 2. MAR (Missing At Random)

**Definition**: Missingness depends on observed data only.

**Example**: Higher income → more likely to report age

**Impact**: Can be handled with proper methods

#### 3. MNAR (Missing Not At Random)

**Definition**: Missingness depends on unobserved data.

**Example**: People with high income less likely to report income

**Impact**: Most challenging, may introduce bias

### Handling Strategies

#### 1. Deletion

**Listwise Deletion**: Remove rows with any missing values
- **Use**: MCAR, small % missing
- **Risk**: Loss of data

**Pairwise Deletion**: Use available data for each analysis
- **Use**: Different variables missing
- **Risk**: Inconsistent sample sizes

#### 2. Imputation

**Mean/Median/Mode**: Replace with central tendency
- **Use**: Numerical, small % missing
- **Risk**: Underestimates variance

**Forward/Backward Fill**: Use previous/next value
- **Use**: Time series, ordered data
- **Risk**: Assumes no change

**Model-Based**: Predict missing values
- **KNN Imputation**: Use k nearest neighbors
- **Regression**: Predict from other variables
- **Use**: When relationships exist
- **Risk**: Overfitting, data leakage

#### 3. Indicator Variables

**Missing Indicator**: Add binary flag for missing
- **Use**: When missingness is informative
- **Advantage**: Preserves information about missingness

---

## 6. Data Drift Detection

### What is Data Drift?

**Definition**: Change in data distribution over time.

**Types**:
- **Concept Drift**: Target relationship changes (P(Y|X) changes)
- **Data Drift**: Input distribution changes (P(X) changes)
- **Covariate Shift**: P(X) changes, P(Y|X) same
- **Label Drift**: P(Y) changes

### Why Detect Drift?

- **Model Performance**: Models degrade when data changes
- **Data Quality**: Detect data pipeline issues
- **Business Changes**: Understand changing patterns
- **Early Warning**: Detect issues before model fails

### Real-World Scenarios

**Scenario 1: E-commerce Recommendation System**
- User behavior changes (mobile vs desktop)
- Seasonal patterns (holiday shopping)
- New product categories introduced

**Scenario 2: Fraud Detection**
- Fraudsters adapt their strategies
- New fraud patterns emerge
- Legitimate user behavior evolves

**Scenario 3: Credit Scoring**
- Economic conditions change
- Lending criteria evolve
- Customer demographics shift

---

### Detection Methods

#### 1. Statistical Tests

##### Kolmogorov-Smirnov Test (Numerical Features)

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift_ks(reference_data, current_data, feature_name, alpha=0.05):
    """
    Detect drift using Kolmogorov-Smirnov test.
    
    Args:
        reference_data: Training/reference data
        current_data: Current/production data
        feature_name: Name of feature to check
        alpha: Significance level
    
    Returns:
        dict with drift status and statistics
    """
    # Perform KS test
    statistic, p_value = ks_2samp(reference_data, current_data)
    
    # Determine drift
    is_drift = p_value < alpha
    
    result = {
        'feature': feature_name,
        'drift_detected': is_drift,
        'ks_statistic': statistic,
        'p_value': p_value,
        'significance_level': alpha
    }
    
    return result

# Example usage
reference = df_train['age'].values
current = df_production['age'].values
result = detect_drift_ks(reference, current, 'age')
print(f"Drift detected: {result['drift_detected']}")
print(f"KS statistic: {result['ks_statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
```

##### Chi-Square Test (Categorical Features)

```python
from scipy.stats import chi2_contingency
import pandas as pd

def detect_drift_chi2(reference_data, current_data, feature_name, alpha=0.05):
    """
    Detect drift in categorical features using Chi-Square test.
    """
    # Get value counts
    ref_counts = pd.Series(reference_data).value_counts()
    curr_counts = pd.Series(current_data).value_counts()
    
    # Align indices (handle missing categories)
    all_categories = ref_counts.index.union(curr_counts.index)
    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    curr_counts = curr_counts.reindex(all_categories, fill_value=0)
    
    # Create contingency table
    contingency = pd.DataFrame({
        'reference': ref_counts,
        'current': curr_counts
    }).T
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    is_drift = p_value < alpha
    
    result = {
        'feature': feature_name,
        'drift_detected': is_drift,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof
    }
    
    return result

# Example usage
reference_cat = df_train['category'].values
current_cat = df_production['category'].values
result = detect_drift_chi2(reference_cat, current_cat, 'category')
```

##### Mann-Whitney U Test (Non-parametric)

```python
from scipy.stats import mannwhitneyu

def detect_drift_mannwhitney(reference_data, current_data, feature_name, alpha=0.05):
    """
    Detect drift using Mann-Whitney U test (non-parametric).
    Good for non-normal distributions.
    """
    statistic, p_value = mannwhitneyu(reference_data, current_data, alternative='two-sided')
    
    is_drift = p_value < alpha
    
    result = {
        'feature': feature_name,
        'drift_detected': is_drift,
        'u_statistic': statistic,
        'p_value': p_value
    }
    
    return result
```

#### 2. Distance Metrics

##### Population Stability Index (PSI)

```python
import numpy as np
import pandas as pd

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI).
    
    PSI = Σ((Actual% - Expected%) × ln(Actual% / Expected%))
    
    Interpretation:
    - PSI < 0.1: No significant drift
    - PSI 0.1-0.25: Moderate drift
    - PSI > 0.25: Significant drift
    
    Args:
        expected: Reference/training data
        actual: Current/production data
        buckets: Number of bins for numerical data
    """
    # For numerical data, create bins
    if pd.api.types.is_numeric_dtype(pd.Series(expected)):
        # Create bins based on reference data
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints[0] = breakpoints[0] - 0.001  # Include minimum
        breakpoints[-1] = breakpoints[-1] + 0.001  # Include maximum
        
        # Bin the data
        expected_binned = pd.cut(expected, breakpoints, include_lowest=True)
        actual_binned = pd.cut(actual, breakpoints, include_lowest=True)
    else:
        # For categorical, use categories directly
        expected_binned = expected
        actual_binned = actual
    
    # Calculate percentages
    expected_counts = pd.Series(expected_binned).value_counts(normalize=True)
    actual_counts = pd.Series(actual_binned).value_counts(normalize=True)
    
    # Align indices
    all_bins = expected_counts.index.union(actual_counts.index)
    expected_pct = expected_counts.reindex(all_bins, fill_value=0)
    actual_pct = actual_counts.reindex(all_bins, fill_value=0)
    
    # Avoid division by zero
    expected_pct = expected_pct + 1e-10
    actual_pct = actual_pct + 1e-10
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi

def interpret_psi(psi):
    """Interpret PSI value."""
    if psi < 0.1:
        return "No significant drift"
    elif psi < 0.25:
        return "Moderate drift - investigate"
    else:
        return "Significant drift - action required"

# Example usage
reference_feature = df_train['age'].values
current_feature = df_production['age'].values

psi_value = calculate_psi(reference_feature, current_feature)
interpretation = interpret_psi(psi_value)

print(f"PSI: {psi_value:.4f}")
print(f"Interpretation: {interpretation}")
```

##### Wasserstein Distance (Earth Mover's Distance)

```python
from scipy.stats import wasserstein_distance

def detect_drift_wasserstein(reference_data, current_data, feature_name, threshold=0.1):
    """
    Detect drift using Wasserstein distance.
    Measures the minimum "work" to transform one distribution to another.
    """
    distance = wasserstein_distance(reference_data, current_data)
    
    # Normalize by standard deviation for relative comparison
    normalized_distance = distance / np.std(reference_data)
    
    is_drift = normalized_distance > threshold
    
    result = {
        'feature': feature_name,
        'drift_detected': is_drift,
        'wasserstein_distance': distance,
        'normalized_distance': normalized_distance,
        'threshold': threshold
    }
    
    return result
```

#### 3. Model-Based Methods

##### Drift Detection Using Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def detect_drift_classifier(reference_data, current_data, threshold=0.6):
    """
    Detect drift by training a classifier to distinguish reference vs current data.
    
    If classifier can easily distinguish (high accuracy), drift is present.
    """
    # Label data: 0 = reference, 1 = current
    X = np.vstack([reference_data, current_data])
    y = np.hstack([np.zeros(len(reference_data)), np.ones(len(current_data))])
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    
    # High accuracy = easy to distinguish = drift present
    is_drift = mean_accuracy > threshold
    
    result = {
        'drift_detected': is_drift,
        'classification_accuracy': mean_accuracy,
        'threshold': threshold
    }
    
    return result
```

##### Feature Importance Changes

```python
def detect_drift_feature_importance(model_train, model_current, feature_names, threshold=0.2):
    """
    Detect drift by comparing feature importance between models.
    """
    importance_train = model_train.feature_importances_
    importance_current = model_current.feature_importances_
    
    # Calculate relative change
    importance_change = np.abs(importance_current - importance_train) / (importance_train + 1e-10)
    
    # Find features with significant changes
    drifted_features = []
    for i, (name, change) in enumerate(zip(feature_names, importance_change)):
        if change > threshold:
            drifted_features.append({
                'feature': name,
                'change_ratio': change,
                'train_importance': importance_train[i],
                'current_importance': importance_current[i]
            })
    
    result = {
        'drift_detected': len(drifted_features) > 0,
        'drifted_features': drifted_features,
        'threshold': threshold
    }
    
    return result
```

---

### Comprehensive Feature Drift Detection

```python
import pandas as pd
import numpy as np
from typing import Dict, List

class FeatureDriftDetector:
    """
    Comprehensive feature drift detection system.
    """
    
    def __init__(self, reference_data: pd.DataFrame, alpha=0.05, psi_threshold=0.25):
        """
        Initialize detector with reference data.
        
        Args:
            reference_data: Training/reference dataset
            alpha: Significance level for statistical tests
            psi_threshold: PSI threshold for drift detection
        """
        self.reference_data = reference_data
        self.alpha = alpha
        self.psi_threshold = psi_threshold
        self.drift_results = {}
    
    def detect_drift(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect drift for all features in current data.
        
        Returns:
            DataFrame with drift detection results
        """
        results = []
        
        for col in current_data.columns:
            if col not in self.reference_data.columns:
                results.append({
                    'feature': col,
                    'drift_detected': True,
                    'method': 'new_feature',
                    'message': 'New feature not in reference data'
                })
                continue
            
            ref_data = self.reference_data[col].dropna()
            curr_data = current_data[col].dropna()
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue
            
            # Detect drift based on data type
            if pd.api.types.is_numeric_dtype(ref_data):
                result = self._detect_numerical_drift(ref_data, curr_data, col)
            else:
                result = self._detect_categorical_drift(ref_data, curr_data, col)
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _detect_numerical_drift(self, ref_data, curr_data, feature_name):
        """Detect drift in numerical features."""
        # KS test
        from scipy.stats import ks_2samp
        ks_stat, ks_pvalue = ks_2samp(ref_data, curr_data)
        ks_drift = ks_pvalue < self.alpha
        
        # PSI
        psi = calculate_psi(ref_data, curr_data)
        psi_drift = psi > self.psi_threshold
        
        # Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        mw_stat, mw_pvalue = mannwhitneyu(ref_data, curr_data, alternative='two-sided')
        mw_drift = mw_pvalue < self.alpha
        
        # Overall drift
        drift_detected = ks_drift or psi_drift or mw_drift
        
        return {
            'feature': feature_name,
            'drift_detected': drift_detected,
            'ks_drift': ks_drift,
            'ks_pvalue': ks_pvalue,
            'psi_drift': psi_drift,
            'psi_value': psi,
            'mannwhitney_drift': mw_drift,
            'mannwhitney_pvalue': mw_pvalue,
            'method': 'numerical'
        }
    
    def _detect_categorical_drift(self, ref_data, curr_data, feature_name):
        """Detect drift in categorical features."""
        # Chi-square test
        from scipy.stats import chi2_contingency
        
        ref_counts = pd.Series(ref_data).value_counts()
        curr_counts = pd.Series(curr_data).value_counts()
        
        all_categories = ref_counts.index.union(curr_counts.index)
        ref_counts = ref_counts.reindex(all_categories, fill_value=0)
        curr_counts = curr_counts.reindex(all_categories, fill_value=0)
        
        contingency = pd.DataFrame({
            'reference': ref_counts,
            'current': curr_counts
        }).T
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        chi2_drift = p_value < self.alpha
        
        # PSI
        psi = calculate_psi(ref_data, curr_data)
        psi_drift = psi > self.psi_threshold
        
        drift_detected = chi2_drift or psi_drift
        
        return {
            'feature': feature_name,
            'drift_detected': drift_detected,
            'chi2_drift': chi2_drift,
            'chi2_pvalue': p_value,
            'psi_drift': psi_drift,
            'psi_value': psi,
            'method': 'categorical'
        }
    
    def generate_report(self, current_data: pd.DataFrame) -> str:
        """Generate human-readable drift report."""
        results = self.detect_drift(current_data)
        
        drifted_features = results[results['drift_detected'] == True]
        
        report = f"""
        Feature Drift Detection Report
        ==============================
        
        Total Features Checked: {len(results)}
        Features with Drift: {len(drifted_features)}
        
        Drifted Features:
        """
        
        for _, row in drifted_features.iterrows():
            report += f"\n- {row['feature']} ({row['method']})"
            if 'psi_value' in row:
                report += f" - PSI: {row['psi_value']:.4f}"
            if 'ks_pvalue' in row:
                report += f" - KS p-value: {row['ks_pvalue']:.4f}"
        
        return report

# Example usage
detector = FeatureDriftDetector(df_train)
drift_results = detector.detect_drift(df_production)
print(detector.generate_report(df_production))
```

---

### Monitoring Drift Over Time

```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DriftMonitor:
    """Monitor feature drift over time."""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_history = []
    
    def check_drift(self, current_data: pd.DataFrame, timestamp=None):
        """Check drift and record in history."""
        if timestamp is None:
            timestamp = datetime.now()
        
        detector = FeatureDriftDetector(self.reference_data)
        results = detector.detect_drift(current_data)
        
        drift_summary = {
            'timestamp': timestamp,
            'total_features': len(results),
            'drifted_features': len(results[results['drift_detected'] == True]),
            'drift_percentage': (results['drift_detected'].sum() / len(results)) * 100
        }
        
        self.drift_history.append(drift_summary)
        return drift_summary
    
    def plot_drift_trend(self):
        """Plot drift trend over time."""
        if not self.drift_history:
            print("No drift history available")
            return
        
        df_history = pd.DataFrame(self.drift_history)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df_history['timestamp'], df_history['drift_percentage'], marker='o')
        plt.xlabel('Time')
        plt.ylabel('Percentage of Features with Drift')
        plt.title('Feature Drift Trend Over Time')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Example usage
monitor = DriftMonitor(df_train)

# Check drift daily
for date in pd.date_range(start='2024-01-01', end='2024-01-31', freq='D'):
    daily_data = df_production[df_production['date'] == date]
    monitor.check_drift(daily_data, timestamp=date)

monitor.plot_drift_trend()
```

---

### Handling Drift

#### 1. Retrain Model

```python
def handle_drift_retrain(model, reference_data, current_data, retrain_threshold=0.25):
    """
    Retrain model when drift is detected.
    """
    detector = FeatureDriftDetector(reference_data)
    results = detector.detect_drift(current_data)
    
    drift_percentage = (results['drift_detected'].sum() / len(results)) * 100
    
    if drift_percentage > retrain_threshold:
        # Combine old and new data
        combined_data = pd.concat([reference_data, current_data], ignore_index=True)
        
        # Retrain model
        X = combined_data.drop('target', axis=1)
        y = combined_data['target']
        model.fit(X, y)
        
        return {
            'action': 'retrained',
            'drift_percentage': drift_percentage,
            'new_training_size': len(combined_data)
        }
    else:
        return {
            'action': 'no_action',
            'drift_percentage': drift_percentage
        }
```

#### 2. Adaptive Models (Online Learning)

```python
from sklearn.linear_model import SGDClassifier

class AdaptiveModel:
    """Model that adapts to drift using online learning."""
    
    def __init__(self):
        self.model = SGDClassifier(loss='log_loss', learning_rate='adaptive')
        self.is_fitted = False
    
    def partial_fit(self, X, y):
        """Update model with new data."""
        if not self.is_fitted:
            self.model.partial_fit(X, y, classes=np.unique(y))
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
```

#### 3. Ensemble Approach

```python
from sklearn.ensemble import VotingClassifier

def create_ensemble_models(reference_data, current_data):
    """
    Create ensemble of old and new models.
    """
    # Train model on reference data
    model_old = RandomForestClassifier(n_estimators=100, random_state=42)
    X_ref = reference_data.drop('target', axis=1)
    y_ref = reference_data['target']
    model_old.fit(X_ref, y_ref)
    
    # Train model on current data
    model_new = RandomForestClassifier(n_estimators=100, random_state=42)
    X_curr = current_data.drop('target', axis=1)
    y_curr = current_data['target']
    model_new.fit(X_curr, y_curr)
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('old', model_old),
            ('new', model_new)
        ],
        voting='soft',
        weights=[0.3, 0.7]  # Weight new model more
    )
    
    return ensemble
```

---

### Best Practices

1. **Establish Baseline**: Use training data as reference
2. **Monitor Continuously**: Check drift regularly (daily/weekly)
3. **Set Thresholds**: Define acceptable drift levels
4. **Track History**: Maintain drift history for trend analysis
5. **Automate Alerts**: Set up alerts for significant drift
6. **Investigate Root Causes**: Understand why drift occurs
7. **Document Actions**: Track what actions were taken
8. **Test Before Retraining**: Validate new models before deployment

---

### Tools for Drift Detection

- **Evidently AI**: Open-source ML monitoring
- **NannyML**: Post-deployment data science
- **Alibi Detect**: Drift detection algorithms
- **Great Expectations**: Data validation and monitoring
- **WhyLabs**: ML observability platform

---

## 7. Schema Validation

### What is Schema Validation?

**Definition**: Ensuring data structure matches expected schema.

### Schema Components

1. **Column Names**: Expected columns present
2. **Data Types**: Each column has correct type
3. **Constraints**: Nullable, unique, range
4. **Relationships**: Foreign keys, dependencies

### Validation Tools

#### Great Expectations

**Features**:
- Define expectations
- Validate batches
- Generate reports
- Data documentation

#### Pandera

**Python library**:
- Type validation
- Range checks
- Custom validators
- Integration with pandas

#### JSON Schema / Avro

**Schema Definition**:
- Define structure
- Validate JSON/data
- Version control

---

## 8. Data Quality Metrics

### Completeness

**Formula**:
```
Completeness = (Non-missing values / Total values) × 100%
```

**Target**: > 95% (depends on use case)

### Accuracy

**Definition**: Percentage of correct values.

**Measurement**: Compare to ground truth (if available).

**Challenge**: Often no ground truth available.

### Consistency

**Definition**: Data is consistent across sources.

**Metrics**:
- Duplicate rate
- Referential integrity violations
- Cross-source discrepancies

### Validity

**Formula**:
```
Validity = (Valid records / Total records) × 100%
```

**Definition**: Records passing all validation rules.

### Uniqueness

**Formula**:
```
Uniqueness = (Unique records / Total records) × 100%
```

**Target**: 100% (no duplicates)

### Timeliness

**Definition**: Data freshness, latency.

**Metrics**:
- Time since last update
- Data age
- Update frequency

---

## 9. Best Practices

### Data Quality Framework

1. **Define Standards**: What is acceptable quality?
2. **Profile Regularly**: Continuous monitoring
3. **Validate Early**: Catch issues at source
4. **Document Issues**: Track quality over time
5. **Automate Checks**: Reduce manual effort
6. **Set Alerts**: Notify on quality degradation

### Validation Pipeline

1. **Ingestion Checks**: Validate on ingestion
2. **Transformation Checks**: Validate after transformations
3. **Output Checks**: Validate before model training
4. **Monitoring**: Continuous quality monitoring

### Data Quality Dashboard

**Metrics to Track**:
- Completeness by column
- Validity rate
- Duplicate rate
- Drift indicators
- Error rates

### Handling Quality Issues

1. **Prevent**: Fix at source
2. **Detect**: Early detection
3. **Correct**: Automated correction when possible
4. **Document**: Track all issues
5. **Alert**: Notify stakeholders

---

## Quick Reference

### Data Quality Dimensions

- **Completeness**: % non-missing
- **Accuracy**: Correctness
- **Consistency**: Cross-source agreement
- **Validity**: Follows rules
- **Timeliness**: Up-to-date
- **Uniqueness**: No duplicates

### Common Checks

- Type validation
- Range validation
- Format validation
- Business rule validation
- Referential integrity

### Tools

- **Profiling**: pandas-profiling, ydata-profiling
- **Validation**: Great Expectations, Pandera
- **Monitoring**: Evidently AI, WhyLabs
- **Drift**: Alibi Detect, NannyML

---

*Last Updated: 2024*

