+++
title = "Data Preprocessing & Cleaning"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Comprehensive guide to data preprocessing and cleaning. Covers handling missing values, outlier detection and treatment, data transformation, encoding categorical variables, feature scaling, and data cleaning best practices."
+++

---

## Introduction

Data preprocessing and cleaning is a crucial step in the data science pipeline. Raw data is often messy, incomplete, and inconsistent. Proper preprocessing ensures that your data is ready for analysis and modeling, which directly impacts model performance.

**Key Principle**: Garbage in, garbage out. Clean, well-preprocessed data leads to better models.

---

## 1. Handling Missing Values

### Understanding Missing Data

**Types of Missingness**:
- **MCAR (Missing Completely At Random)**: Missingness is independent of observed and unobserved data
- **MAR (Missing At Random)**: Missingness depends only on observed data
- **MNAR (Missing Not At Random)**: Missingness depends on unobserved data

### Detection Methods

```python
import pandas as pd
import numpy as np

# Check for missing values
df.isnull().sum()
df.isnull().mean()  # Percentage of missing values

# Visualize missing data
import seaborn as sns
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
```

### Handling Strategies

#### 1. Deletion Methods

**Listwise Deletion (Complete Case Analysis)**:
- Remove rows with any missing values
- **Use when**: Missing data is MCAR and < 5% of data
- **Pros**: Simple, no assumptions
- **Cons**: Loss of information, potential bias

```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows where specific columns have missing values
df_clean = df.dropna(subset=['column1', 'column2'])
```

**Pairwise Deletion**:
- Use available data for each analysis
- **Use when**: Different variables have different missing patterns
- **Pros**: Retains more data
- **Cons**: Inconsistent sample sizes across analyses

#### 2. Imputation Methods

**Mean/Median/Mode Imputation**:
- Replace missing values with mean (numerical) or mode (categorical)
- **Use when**: Missing data is MCAR, small percentage
- **Pros**: Simple, preserves sample size
- **Cons**: Reduces variance, may introduce bias

```python
# Mean imputation
df['column'].fillna(df['column'].mean(), inplace=True)

# Median imputation (robust to outliers)
df['column'].fillna(df['column'].median(), inplace=True)

# Mode imputation (categorical)
df['column'].fillna(df['column'].mode()[0], inplace=True)
```

**Forward Fill / Backward Fill**:
- Use previous/next value to fill missing
- **Use when**: Time series data, ordered sequences
- **Pros**: Preserves temporal patterns
- **Cons**: May propagate errors

```python
# Forward fill
df['column'].fillna(method='ffill', inplace=True)

# Backward fill
df['column'].fillna(method='bfill', inplace=True)
```

**K-Nearest Neighbors (KNN) Imputation**:
- Use k most similar observations to impute
- **Use when**: Missing data has patterns, relationships exist
- **Pros**: Uses relationships between variables
- **Cons**: Computationally expensive, sensitive to k

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

**Regression Imputation**:
- Predict missing values using other variables
- **Use when**: Strong relationships between variables
- **Pros**: Uses variable relationships
- **Cons**: Underestimates variance, may overfit

```python
from sklearn.linear_model import LinearRegression
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(estimator=LinearRegression())
df_imputed = imputer.fit_transform(df)
```

**Advanced: Multiple Imputation**:
- Create multiple imputed datasets, analyze each, combine results
- **Use when**: Missing data is MAR, need uncertainty estimates
- **Pros**: Accounts for imputation uncertainty
- **Cons**: Complex, computationally intensive

---

## 2. Outlier Detection & Treatment

### What are Outliers?

**Definition**: Data points that significantly differ from other observations.

**Types**:
- **Univariate Outliers**: Extreme values in one variable
- **Multivariate Outliers**: Unusual combinations of variables
- **Point Outliers**: Single anomalous points
- **Contextual Outliers**: Anomalous in specific context

### Detection Methods

#### 1. Statistical Methods

**Z-Score Method**:
```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['column']))
outliers = df[z_scores > 3]  # Threshold: 3 standard deviations
```

**IQR Method (Interquartile Range)**:
```python
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
```

**Modified Z-Score (MAD)**:
```python
median = df['column'].median()
mad = np.median(np.abs(df['column'] - median))
modified_z_score = 0.6745 * (df['column'] - median) / mad
outliers = df[np.abs(modified_z_score) > 3.5]
```

#### 2. Visualization Methods

**Box Plots**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(data=df, y='column')
plt.show()
```

**Scatter Plots**:
```python
plt.scatter(df['x'], df['y'])
plt.show()
```

**Histograms**:
```python
df['column'].hist(bins=50)
plt.show()
```

#### 3. Machine Learning Methods

**Isolation Forest**:
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(df)
outliers = df[outlier_labels == -1]
```

**DBSCAN Clustering**:
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(df)
outliers = df[clusters == -1]  # -1 indicates outliers
```

### Treatment Strategies

**1. Removal**:
- Delete outlier observations
- **Use when**: Outliers are errors, small percentage
- **Pros**: Simple
- **Cons**: Loss of information

```python
df_clean = df[z_scores <= 3]
```

**2. Capping/Winsorizing**:
- Replace outliers with threshold values
- **Use when**: Outliers are valid but extreme
- **Pros**: Retains data, reduces impact
- **Cons**: May lose information

```python
# Cap at 95th and 5th percentiles
lower_cap = df['column'].quantile(0.05)
upper_cap = df['column'].quantile(0.95)
df['column'] = df['column'].clip(lower=lower_cap, upper=upper_cap)
```

**3. Transformation**:
- Apply log, square root, or other transformations
- **Use when**: Skewed distributions
- **Pros**: Normalizes distribution
- **Cons**: Changes interpretation

```python
# Log transformation
df['column_log'] = np.log1p(df['column'])  # log1p handles zeros
```

**4. Separate Treatment**:
- Model outliers separately
- **Use when**: Outliers represent different population
- **Pros**: Preserves information
- **Cons**: More complex modeling

---

## 3. Data Transformation

### Scaling & Normalization

**Why Scale?**:
- Algorithms sensitive to feature scale (SVM, KNN, Neural Networks)
- Gradient descent converges faster
- Prevents features with large ranges from dominating

**Standardization (Z-score Normalization)**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# Mean = 0, Std = 1
```

**Min-Max Scaling**:
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
# Range: [0, 1]
```

**Robust Scaling**:
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
# Uses median and IQR (robust to outliers)
```

**Max Abs Scaling**:
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(df)
# Divides by max absolute value, range: [-1, 1]
```

### Encoding Categorical Variables

**One-Hot Encoding**:
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['category_column']])
```

**Label Encoding**:
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['encoded'] = encoder.fit_transform(df['category_column'])
```

**Ordinal Encoding**:
```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()
df[['encoded']] = encoder.fit_transform(df[['category_column']])
```

**Target Encoding (Mean Encoding)**:
```python
# Encode categories by their mean target value
mean_encoding = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(mean_encoding)
```

**Frequency Encoding**:
```python
# Encode by frequency of category
freq_encoding = df['category'].value_counts().to_dict()
df['category_freq'] = df['category'].map(freq_encoding)
```

### Handling Skewed Data

**Log Transformation**:
```python
df['log_column'] = np.log1p(df['column'])
```

**Square Root Transformation**:
```python
df['sqrt_column'] = np.sqrt(df['column'])
```

**Box-Cox Transformation**:
```python
from scipy.stats import boxcox

df['boxcox_column'], lambda_param = boxcox(df['column'] + 1)
```

**Yeo-Johnson Transformation**:
```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer(method='yeo-johnson')
df['transformed'] = transformer.fit_transform(df[['column']])
```

---

## 4. Feature Engineering

### Creating New Features

**Polynomial Features**:
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
df_poly = poly.fit_transform(df[['feature1', 'feature2']])
```

**Interaction Features**:
```python
df['interaction'] = df['feature1'] * df['feature2']
df['ratio'] = df['feature1'] / (df['feature2'] + 1e-8)
```

**Binning**:
```python
# Equal-width binning
df['binned'] = pd.cut(df['column'], bins=5)

# Equal-frequency binning
df['binned'] = pd.qcut(df['column'], q=5)
```

**Date/Time Features**:
```python
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
```

---

## 5. Data Cleaning Best Practices

### Workflow

1. **Understand Your Data**: Explore, visualize, understand domain
2. **Handle Missing Values**: Choose appropriate strategy
3. **Detect Outliers**: Identify and understand outliers
4. **Transform Data**: Scale, encode, transform as needed
5. **Feature Engineering**: Create new meaningful features
6. **Validate**: Check data quality after preprocessing

### Common Pitfalls

**1. Data Leakage**:
- Using future information to predict past
- **Solution**: Split data before preprocessing, fit on train only

```python
# WRONG: Fit on entire dataset
scaler.fit(X_all)
X_scaled = scaler.transform(X_all)

# CORRECT: Fit on training set only
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**2. Over-imputation**:
- Imputing too much data
- **Solution**: Consider if feature is useful if mostly missing

**3. Ignoring Domain Knowledge**:
- Not consulting domain experts
- **Solution**: Understand what outliers/missing values mean

**4. Not Documenting**:
- Not tracking preprocessing steps
- **Solution**: Document all transformations, save preprocessors

```python
import joblib

# Save preprocessor
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
```

---

## 6. Preprocessing Pipeline

### Using Scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Use in full pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

---

## Key Takeaways

1. **Data Quality First**: Clean data is essential for good models
2. **Understand Missingness**: Choose imputation strategy based on missing data mechanism
3. **Handle Outliers Carefully**: Understand if they're errors or valid extreme values
4. **Scale Appropriately**: Use scaling for algorithms that need it
5. **Encode Categorically**: Choose encoding method based on algorithm and data
6. **Avoid Data Leakage**: Always fit preprocessors on training data only
7. **Document Everything**: Track all preprocessing steps for reproducibility
8. **Domain Knowledge Matters**: Consult experts to understand your data

---

## Practice Problems

- **Kaggle**: Titanic, House Prices, Credit Card Fraud Detection
- **Real-world**: Clean a messy dataset with multiple data quality issues
- **Challenge**: Build a reusable preprocessing pipeline class

---

## References

- Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- Pandas Documentation: https://pandas.pydata.org/docs/
- "Handling Missing Data" by Little & Rubin
- "An Introduction to Statistical Learning" by James et al.

