# Data Quality & Validation: Interview Q&A Guide

Comprehensive guide to data quality, validation, and data profiling for data science interviews.


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
- **Concept Drift**: Target relationship changes
- **Data Drift**: Input distribution changes
- **Covariate Shift**: P(X) changes, P(Y|X) same

### Why Detect Drift?

- **Model Performance**: Models degrade when data changes
- **Data Quality**: Detect data pipeline issues
- **Business Changes**: Understand changing patterns

### Detection Methods

#### 1. Statistical Tests

**Kolmogorov-Smirnov Test**: Compare distributions
**Chi-Square Test**: Compare categorical distributions
**Mann-Whitney U Test**: Compare medians

#### 2. Distance Metrics

**Population Stability Index (PSI)**:
```
PSI = Σ((Actual% - Expected%) × ln(Actual% / Expected%))
```

**Interpretation**:
- PSI < 0.1: No significant drift
- PSI 0.1-0.25: Moderate drift
- PSI > 0.25: Significant drift

#### 3. Model-Based

**Monitor Model Performance**: Accuracy, prediction distribution
**Feature Importance Changes**: Track feature importance over time

### Handling Drift

1. **Retrain Model**: Update with new data
2. **Adaptive Models**: Online learning
3. **Ensemble**: Combine old and new models
4. **Investigate**: Understand cause of drift

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

