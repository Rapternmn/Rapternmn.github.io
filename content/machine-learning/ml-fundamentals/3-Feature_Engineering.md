+++
title = "Feature Engineering"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 2
description = "A comprehensive guide to feature engineering techniques and best practices. Covers handling missing values, categorical encoding, feature scaling, feature selection, and feature creation."
+++


## 1. Handling Missing Values

### How do you handle missing values?

**Approaches:**

#### 1. Drop Missing Values

- **Use when**: Percentage of missing rows is very low (< 5%)
- **Method**: `df.dropna()`
- **Considerations**: May lose valuable data if not random

#### 2. Imputation (Fill Values)

**Numerical Columns:**
- **Mean**: Good for normally distributed data
  ```python
  df['age'].fillna(df['age'].mean())
  ```
- **Median**: Better for skewed data, robust to outliers
  ```python
  df['age'].fillna(df['age'].median())
  ```
- **Mode**: For discrete numerical features
- **Constant**: Use domain-specific default (e.g., 0, -1)

**Categorical Columns:**
- **Mode**: Most frequent category
  ```python
  df['city'].fillna(df['city'].mode()[0])
  ```
- **'Unknown'**: Create new category for missing
  ```python
  df['city'].fillna('Unknown')
  ```

#### 3. Model-Based Imputation

- Use a model (e.g., KNN, regression) to predict missing values
- **KNN Imputation**: Use k nearest neighbors to impute
- **Regression Imputation**: Train model on non-missing data to predict missing
- **Advantages**: More sophisticated, preserves relationships
- **Disadvantages**: More complex, can overfit

#### 4. Indicator Variable

- Add a flag: `was_missing = 1` if value was NaN else 0
- **Use Case**: When missingness itself is informative
- **Example**: Missing income might indicate unemployment

**Best Practice**: Impute using training data statistics only, avoid data leakage.

---

## 2. Categorical Feature Encoding

### How do you deal with categorical features?

**Approaches:**

#### 1. Label Encoding

- Assign integer to each category
- **Good for**: Tree-based models (Random Forest, XGBoost)
- **Limitation**: Implies ordinal relationship (may not be true)
- **Example**:
  ```python
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  df['category'] = le.fit_transform(df['category'])
  ```

#### 2. One-Hot Encoding

- Create binary columns for each category
- **Good for**: Linear models, neural networks
- **Limitation**: High cardinality creates many features (curse of dimensionality)
- **Example**:
  ```python
  pd.get_dummies(df['city'])
  ```

#### 3. Target Encoding / Mean Encoding

- Replace category with mean of target for that category
- **Advantages**: Captures target relationship, reduces dimensionality
- **Risk**: Data leakage — apply only on training folds during CV
- **Best Practice**: Use cross-validation to compute means per fold

#### 4. Frequency Encoding

- Replace with how often each category appears
- **Use Case**: High cardinality categorical features
- **Advantage**: No data leakage risk
- **Example**: Replace city name with frequency of that city

#### 5. Embeddings

- For high-cardinality categories (e.g., user_id, product_id)
- Use learned embeddings (common in DL/NLP)
- **Use Case**: When you have many categories and relationships matter
- **Example**: User embeddings in recommendation systems

---

## 3. Feature Importance

### What is feature importance, and how do you measure it?

**Feature Importance** = A measure of how useful a feature is in predicting the target.

**Ways to measure:**

#### 1. Tree-Based Models

- Feature importance is built-in (e.g., Random Forest, XGBoost)
- Based on how much each feature reduces impurity
- **Advantages**: Fast, model-specific
- **Limitations**: May be biased towards high-cardinality features

#### 2. Permutation Importance

- Shuffle one feature at a time and measure drop in model performance
- **Advantages**: Model-agnostic, intuitive
- **Method**: 
  1. Train model
  2. Shuffle feature values
  3. Measure performance drop
  4. Larger drop = more important feature

#### 3. SHAP Values

- Model-agnostic, explainable AI technique
- Quantifies feature impact on individual predictions
- **Advantages**: 
  - Explains individual predictions
  - Shows feature interactions
  - Model-agnostic
- **Use Case**: When interpretability is crucial

#### 4. Lasso Regression

- Coefficients are shrunk to zero → unused features are automatically pruned
- **L1 Regularization**: Encourages sparsity
- **Use Case**: Feature selection in linear models

---

## 4. Feature Scaling & Normalization

### When and why do you scale features?

**Why Scale:**
- Algorithms sensitive to feature scale (SVM, KNN, Neural Networks)
- Gradient descent converges faster
- Prevents features with large ranges from dominating

**Methods:**

#### 1. Standardization (Z-score Normalization)

- Transform to mean=0, std=1
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Use Case**: When data is approximately normally distributed
- **Example**: `StandardScaler()` in sklearn

#### 2. Min-Max Scaling

- Transform to range [0, 1]
- **Formula**: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$
- **Use Case**: When you need bounded range
- **Example**: `MinMaxScaler()` in sklearn

#### 3. Robust Scaling

- Uses median and IQR (robust to outliers)
- **Formula**: $x_{scaled} = \frac{x - median}{IQR}$
- **Use Case**: When data has outliers
- **Example**: `RobustScaler()` in sklearn

---

## 5. Feature Selection

### How do you select important features?

**Methods:**

#### 1. Filter Methods
- Statistical tests (chi-square, correlation)
- **Advantages**: Fast, model-independent
- **Example**: Remove features with low variance or high correlation

#### 2. Wrapper Methods
- Use model performance to select features
- **Examples**: Forward selection, backward elimination
- **Advantages**: Model-specific, considers interactions
- **Disadvantages**: Computationally expensive

#### 3. Embedded Methods
- Feature selection during model training
- **Examples**: Lasso, Ridge, Tree-based feature importance
- **Advantages**: Efficient, model-specific

---

## 6. Feature Creation

### How do you create new features?

**Techniques:**

#### 1. Domain Knowledge
- Create features based on domain expertise
- **Example**: Age groups, time-based features (hour of day, day of week)

#### 2. Polynomial Features
- Create interaction terms
- **Example**: $x_1 \times x_2$, $x_1^2$
- **Use Case**: Capture non-linear relationships

#### 3. Binning
- Convert continuous to categorical
- **Example**: Age → Age groups (0-18, 19-35, 36-50, 50+)
- **Use Case**: Handle non-linear relationships, reduce overfitting

#### 4. Aggregation
- Group-level statistics
- **Example**: User's average purchase amount, count of transactions
- **Use Case**: Capture group patterns

#### 5. Time-Based Features
- Extract components from datetime
- **Examples**: Hour, day of week, month, season
- **Use Case**: Capture temporal patterns

---

## Best Practices

1. **Avoid Data Leakage**: Compute statistics only on training data
2. **Handle Missing Values Early**: Before feature engineering
3. **Scale Features**: For algorithms sensitive to scale
4. **Feature Selection**: Remove redundant or irrelevant features
5. **Cross-Validation**: Use CV for target encoding and feature selection
6. **Document Transformations**: Keep track of all feature engineering steps

---

*Last Updated: 2024*
