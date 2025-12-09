+++
title = "Exploratory Data Analysis (EDA)"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Comprehensive guide to Exploratory Data Analysis. Covers univariate analysis, bivariate analysis, multivariate analysis, data visualization, statistical summaries, correlation analysis, and EDA best practices."
+++

---

## Introduction

**Exploratory Data Analysis (EDA)** is the process of analyzing datasets to summarize their main characteristics, often with visual methods. EDA helps you understand your data, discover patterns, detect anomalies, test hypotheses, and check assumptions before modeling.

**Key Principle**: "The greatest value of a picture is when it forces us to notice what we never expected to see." - John Tukey

---

## 1. EDA Overview

### Goals of EDA

1. **Understand Data Structure**: Shape, size, types, distributions
2. **Detect Patterns**: Trends, relationships, anomalies
3. **Identify Problems**: Missing values, outliers, inconsistencies
4. **Generate Hypotheses**: Form ideas about relationships
5. **Guide Modeling**: Inform feature engineering and model selection

### EDA Workflow

1. **Load & Inspect**: Load data, check shape, types, basic info
2. **Univariate Analysis**: Analyze individual variables
3. **Bivariate Analysis**: Analyze relationships between pairs
4. **Multivariate Analysis**: Analyze complex relationships
5. **Summary & Insights**: Document findings, prepare for modeling

---

## 2. Initial Data Inspection

### Basic Information

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic information
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Size: {df.size}")     # Total elements
print(f"Memory: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Data types
print(df.dtypes)

# Summary
df.info()

# First few rows
df.head()

# Last few rows
df.tail()

# Random sample
df.sample(5)
```

### Missing Values

```python
# Count missing values
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percentage': missing_percent
}).sort_values('Missing Count', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0])

# Visualize missing values
import missingno as msno
msno.matrix(df)
msno.bar(df)
```

### Duplicate Records

```python
# Check for duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")

# Remove duplicates (if needed)
df_clean = df.drop_duplicates()
```

---

## 3. Univariate Analysis

### Numerical Variables

#### Summary Statistics

```python
# Descriptive statistics
df.describe()

# Additional statistics
df.describe(include='all')
df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis

for col in numerical_cols:
    print(f"{col}:")
    print(f"  Skewness: {skew(df[col].dropna()):.2f}")
    print(f"  Kurtosis: {kurtosis(df[col].dropna()):.2f}")
```

#### Visualizations

**Histogram**:
```python
df['column'].hist(bins=50, figsize=(10, 6))
plt.title('Distribution of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

**KDE Plot (Kernel Density Estimation)**:
```python
df['column'].plot(kind='kde', figsize=(10, 6))
plt.title('Density Plot of Column')
plt.show()
```

**Box Plot**:
```python
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['column'])
plt.title('Box Plot of Column')
plt.show()
```

**Violin Plot**:
```python
plt.figure(figsize=(8, 6))
sns.violinplot(y=df['column'])
plt.title('Violin Plot of Column')
plt.show()
```

**Q-Q Plot (Quantile-Quantile)**:
```python
from scipy.stats import probplot

probplot(df['column'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.show()
```

### Categorical Variables

#### Summary Statistics

```python
# Value counts
df['category'].value_counts()

# Value counts with percentages
df['category'].value_counts(normalize=True) * 100

# Unique values
df['category'].nunique()
df['category'].unique()
```

#### Visualizations

**Bar Chart**:
```python
df['category'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title('Frequency of Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

**Horizontal Bar Chart**:
```python
df['category'].value_counts().plot(kind='barh', figsize=(10, 6))
plt.title('Frequency of Categories')
plt.show()
```

**Pie Chart**:
```python
df['category'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribution of Categories')
plt.ylabel('')
plt.show()
```

---

## 4. Bivariate Analysis

### Numerical vs Numerical

#### Scatter Plot

```python
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot: X vs Y')
plt.show()
```

#### Correlation Analysis

```python
# Correlation matrix
correlation_matrix = df[numerical_cols].corr()
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix')
plt.show()
```

#### Pair Plot

```python
sns.pairplot(df[numerical_cols], diag_kind='kde')
plt.show()
```

#### Regression Plot

```python
sns.regplot(x='x', y='y', data=df)
plt.title('Regression Plot: X vs Y')
plt.show()
```

### Numerical vs Categorical

#### Box Plot

```python
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='numerical', data=df)
plt.title('Numerical Variable by Category')
plt.xticks(rotation=45)
plt.show()
```

#### Violin Plot

```python
plt.figure(figsize=(12, 6))
sns.violinplot(x='category', y='numerical', data=df)
plt.title('Distribution by Category')
plt.xticks(rotation=45)
plt.show()
```

#### Grouped Statistics

```python
# Mean by category
df.groupby('category')['numerical'].mean()

# Multiple statistics
df.groupby('category')['numerical'].agg(['mean', 'median', 'std', 'count'])
```

### Categorical vs Categorical

#### Contingency Table

```python
# Cross-tabulation
pd.crosstab(df['category1'], df['category2'])

# With percentages
pd.crosstab(df['category1'], df['category2'], normalize='index') * 100
```

#### Stacked Bar Chart

```python
pd.crosstab(df['category1'], df['category2']).plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Category1 vs Category2')
plt.xlabel('Category1')
plt.ylabel('Count')
plt.legend(title='Category2')
plt.xticks(rotation=45)
plt.show()
```

#### Heatmap

```python
crosstab = pd.crosstab(df['category1'], df['category2'])
sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Category1 vs Category2')
plt.show()
```

---

## 5. Multivariate Analysis

### 3D Scatter Plot

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['x'], df['y'], df['z'], c=df['target'], cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot')
plt.show()
```

### Faceted Plots

```python
# FacetGrid
g = sns.FacetGrid(df, col='category', col_wrap=3)
g.map(plt.hist, 'numerical')
plt.show()
```

### Pair Plot with Hue

```python
sns.pairplot(df, hue='category', diag_kind='kde')
plt.show()
```

---

## 6. Advanced Visualizations

### Distribution Comparison

```python
# Compare distributions
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

df[df['category'] == 'A']['numerical'].hist(ax=axes[0], bins=30, alpha=0.7, label='A')
df[df['category'] == 'B']['numerical'].hist(ax=axes[0], bins=30, alpha=0.7, label='B')
axes[0].set_title('Histogram Comparison')
axes[0].legend()

df[df['category'] == 'A']['numerical'].plot(kind='kde', ax=axes[1], label='A')
df[df['category'] == 'B']['numerical'].plot(kind='kde', ax=axes[1], label='B')
axes[1].set_title('Density Comparison')
axes[1].legend()

plt.show()
```

### Time Series Analysis

```python
# Line plot for time series
df.set_index('date')['value'].plot(figsize=(12, 6))
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df.set_index('date')['value'], model='additive')
decomposition.plot()
plt.show()
```

### Heatmap for Correlation

```python
# Annotated correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Correlation Matrix (Lower Triangle)')
plt.show()
```

---

## 7. Statistical Tests

### Normality Tests

```python
from scipy.stats import shapiro, normaltest, anderson

# Shapiro-Wilk test (for small samples)
stat, p_value = shapiro(df['column'].sample(5000))  # Max 5000
print(f"Shapiro-Wilk: statistic={stat:.4f}, p-value={p_value:.4f}")

# D'Agostino's normality test
stat, p_value = normaltest(df['column'].dropna())
print(f"D'Agostino: statistic={stat:.4f}, p-value={p_value:.4f}")
```

### Correlation Tests

```python
from scipy.stats import pearsonr, spearmanr, kendalltau

# Pearson correlation
corr, p_value = pearsonr(df['x'], df['y'])
print(f"Pearson: correlation={corr:.4f}, p-value={p_value:.4f}")

# Spearman correlation (non-parametric)
corr, p_value = spearmanr(df['x'], df['y'])
print(f"Spearman: correlation={corr:.4f}, p-value={p_value:.4f}")
```

### Group Comparison Tests

```python
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway

# T-test (parametric)
stat, p_value = ttest_ind(df[df['group'] == 'A']['value'],
                          df[df['group'] == 'B']['value'])
print(f"T-test: statistic={stat:.4f}, p-value={p_value:.4f}")

# Mann-Whitney U test (non-parametric)
stat, p_value = mannwhitneyu(df[df['group'] == 'A']['value'],
                             df[df['group'] == 'B']['value'])
print(f"Mann-Whitney: statistic={stat:.4f}, p-value={p_value:.4f}")
```

---

## 8. EDA Best Practices

### 1. Start with Questions

- What are you trying to understand?
- What relationships might exist?
- What patterns are you looking for?

### 2. Iterative Process

- Explore → Hypothesize → Test → Refine
- Don't try to see everything at once
- Follow interesting leads

### 3. Document Everything

```python
# Create EDA report
eda_report = {
    'dataset_shape': df.shape,
    'missing_values': df.isnull().sum().to_dict(),
    'duplicate_rows': df.duplicated().sum(),
    'numerical_summary': df.describe().to_dict(),
    'categorical_summary': {col: df[col].value_counts().to_dict() 
                           for col in categorical_cols}
}
```

### 4. Visualize First, Then Analyze

- Visualizations reveal patterns statistics might miss
- Use multiple visualization types
- Don't rely on single view

### 5. Check Assumptions

- Normality for parametric tests
- Linearity for regression
- Independence of observations
- Homoscedasticity (equal variance)

### 6. Handle Outliers Before Analysis

- Identify outliers first
- Understand if they're errors or valid
- Decide on treatment strategy

### 7. Feature Engineering Insights

- EDA should inform feature engineering
- Identify which features might be useful
- Discover relationships to exploit

---

## 9. Automated EDA Tools

### Pandas Profiling

```python
from pandas_profiling import ProfileReport

profile = ProfileReport(df, title='EDA Report')
profile.to_file('eda_report.html')
```

### Sweetviz

```python
import sweetviz as sv

report = sv.analyze(df)
report.show_html('sweetviz_report.html')
```

### D-Tale

```python
import dtale

d = dtale.show(df)
d.open_browser()
```

---

## Key Takeaways

1. **EDA is Iterative**: Explore, hypothesize, test, refine
2. **Visualization is Key**: Use multiple plot types to understand data
3. **Understand Before Modeling**: EDA guides feature engineering and model selection
4. **Check Assumptions**: Verify statistical assumptions before modeling
5. **Document Findings**: Keep track of insights and patterns discovered
6. **Domain Knowledge Matters**: Combine EDA with domain expertise
7. **Automation Helps**: Use tools for initial exploration, but dive deep manually

---

## Practice Problems

- **Kaggle**: Titanic, House Prices, Customer Churn
- **Real-world**: Perform complete EDA on a dataset of your choice
- **Challenge**: Create an automated EDA report generator

---

## References

- "Exploratory Data Analysis" by John Tukey
- "The Visual Display of Quantitative Information" by Edward Tufte
- Matplotlib Documentation: https://matplotlib.org/
- Seaborn Documentation: https://seaborn.pydata.org/
- Pandas Documentation: https://pandas.pydata.org/docs/

