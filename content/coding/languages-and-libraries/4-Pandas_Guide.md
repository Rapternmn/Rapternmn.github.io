+++
title = "Pandas: Data Manipulation & Analysis"
date = 2025-12-09T10:00:00+05:30
draft = false
weight = 4
description = "Comprehensive Pandas guide: DataFrame operations, data loading, manipulation, cleaning, groupby, merging, time series, and performance optimization for data analysis."
+++

---

## Introduction

**Pandas** is a powerful Python library for data manipulation and analysis. It provides data structures and operations for working with structured data, making it essential for data science, analytics, and data engineering tasks.

**Key Features**:
- **DataFrame**: Two-dimensional labeled data structure
- **Series**: One-dimensional labeled array
- **Data loading**: CSV, Excel, JSON, SQL, and more
- **Data manipulation**: Filtering, grouping, merging, pivoting
- **Data cleaning**: Handling missing values, duplicates
- **Time series**: Date/time operations

---

## Core Data Structures

### Series

**One-dimensional labeled array**:

```python
import pandas as pd
import numpy as np

# Create Series
s = pd.Series([1, 3, 5, 7, 9])
s = pd.Series([1, 3, 5, 7, 9], index=['a', 'b', 'c', 'd', 'e'])
s = pd.Series({'a': 1, 'b': 3, 'c': 5})

# Access elements
s[0]           # By position
s['a']          # By label
s[['a', 'c']]   # Multiple labels

# Attributes
s.index        # Index
s.values       # Values (numpy array)
s.dtype        # Data type
s.size         # Number of elements
s.name         # Series name

# Operations
s + 10         # Add scalar
s * 2          # Multiply scalar
s.sum()        # Sum
s.mean()       # Mean
s.max()         # Maximum
s.min()         # Minimum
```

### DataFrame

**Two-dimensional labeled data structure**:

```python
# Create DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['NYC', 'LA', 'Chicago', 'Houston']
})

# From list of dictionaries
data = [
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob', 'age': 30, 'city': 'LA'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
]
df = pd.DataFrame(data)

# From CSV
df = pd.read_csv('data.csv')

# Basic info
df.shape        # (rows, columns)
df.size         # Total elements
df.columns      # Column names
df.index        # Row index
df.dtypes       # Data types
df.info()       # Summary info
df.describe()   # Statistical summary
```

---

## Data Loading and Saving

### Reading Data

```python
# CSV
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', sep=',', header=0, index_col=0)
df = pd.read_csv('data.csv', nrows=1000)  # Read first 1000 rows
df = pd.read_csv('data.csv', usecols=['col1', 'col2'])  # Specific columns

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # First sheet

# JSON
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query('SELECT * FROM table_name', conn)
df = pd.read_sql('SELECT * FROM table_name', conn)

# Parquet (efficient binary format)
df = pd.read_parquet('data.parquet')

# HTML tables
df = pd.read_html('https://example.com/table.html')[0]
```

### Writing Data

```python
# CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', index=False, sep='\t')  # Tab-separated

# Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# JSON
df.to_json('output.json', orient='records')

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Parquet
df.to_parquet('output.parquet', compression='snappy')
```

---

## Data Selection and Filtering

### Column Selection

```python
# Single column (returns Series)
df['name']
df.name  # Same, but only works if column name is valid Python identifier

# Multiple columns (returns DataFrame)
df[['name', 'age']]

# Column by position
df.iloc[:, 0]        # First column
df.iloc[:, [0, 2]]   # First and third columns
```

### Row Selection

```python
# By label
df.loc[0]            # First row (Series)
df.loc[[0, 2]]       # First and third rows (DataFrame)

# By position
df.iloc[0]           # First row
df.iloc[0:3]         # First 3 rows
df.iloc[[0, 2, 4]]   # Specific rows

# First/last N rows
df.head(5)           # First 5 rows
df.tail(5)           # Last 5 rows
```

### Boolean Indexing

```python
# Single condition
df[df['age'] > 30]
df[df.age > 30]      # Same

# Multiple conditions
df[(df['age'] > 30) & (df['city'] == 'NYC')]
df[(df['age'] > 30) | (df['city'] == 'LA')]

# Using query (more readable)
df.query('age > 30 and city == "NYC"')
df.query('age > 30 or city == "LA"')

# isin
df[df['city'].isin(['NYC', 'LA'])]

# String operations
df[df['name'].str.startswith('A')]
df[df['name'].str.contains('Bob')]
df[df['email'].str.endswith('@gmail.com')]
```

### Advanced Selection

```python
# Using loc with conditions
df.loc[df['age'] > 30, ['name', 'age']]

# Using iloc for position-based
df.iloc[0:5, 0:3]    # First 5 rows, first 3 columns

# Using at and iat for single values
df.at[0, 'name']     # Fast single value access
df.iat[0, 0]         # Fast single value by position
```

---

## Data Manipulation

### Adding/Removing Columns

```python
# Add column
df['new_col'] = df['age'] * 2
df['new_col'] = 'constant_value'
df['new_col'] = df['col1'] + df['col2']

# Insert column at specific position
df.insert(1, 'middle_col', [1, 2, 3, 4])

# Remove columns
df.drop('col_name', axis=1, inplace=True)
df.drop(['col1', 'col2'], axis=1, inplace=True)

# Remove rows
df.drop(0, axis=0, inplace=True)  # Remove first row
df.drop([0, 2], axis=0, inplace=True)  # Remove multiple rows
```

### Renaming

```python
# Rename columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)
df.rename(columns={'old1': 'new1', 'old2': 'new2'}, inplace=True)

# Rename index
df.rename(index={0: 'first', 1: 'second'}, inplace=True)

# Set column names
df.columns = ['col1', 'col2', 'col3']
```

### Sorting

```python
# Sort by column
df.sort_values('age', ascending=True)
df.sort_values(['age', 'name'], ascending=[True, False])

# Sort by index
df.sort_index(ascending=False)

# Reset index after sorting
df.sort_values('age').reset_index(drop=True)
```

### Duplicates

```python
# Check for duplicates
df.duplicated()              # Boolean Series
df.duplicated(subset=['name'])  # Check specific columns

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['name'], keep='first')
df.drop_duplicates(subset=['name'], keep='last')
```

---

## GroupBy Operations

### Basic GroupBy

```python
# Group by single column
df.groupby('department')

# Group by multiple columns
df.groupby(['department', 'city'])

# Aggregate functions
df.groupby('department')['salary'].sum()
df.groupby('department')['salary'].mean()
df.groupby('department')['salary'].agg(['sum', 'mean', 'max', 'min'])

# Multiple aggregations
df.groupby('department').agg({
    'salary': ['sum', 'mean'],
    'age': 'mean',
    'name': 'count'
})
```

### Common GroupBy Operations

```python
# Count
df.groupby('department').size()        # Number of rows
df.groupby('department')['name'].count()  # Non-null values

# First/Last
df.groupby('department').first()
df.groupby('department').last()

# nth
df.groupby('department').nth(0)        # First row of each group

# Transform (returns same shape as original)
df['dept_avg'] = df.groupby('department')['salary'].transform('mean')

# Apply custom function
df.groupby('department').apply(lambda x: x['salary'].max() - x['salary'].min())

# Filter groups
df.groupby('department').filter(lambda x: len(x) > 5)  # Groups with >5 rows
```

### Pivot Tables

```python
# Create pivot table
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum'
)

# Multiple aggregations
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc=['sum', 'mean']
)

# With margins
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    margins=True
)
```

---

## Merging and Joining

### Merge

```python
# Inner join (default)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, left_on='key1', right_on='key2')

# Left join
pd.merge(df1, df2, on='key', how='left')

# Right join
pd.merge(df1, df2, on='key', how='right')

# Outer join
pd.merge(df1, df2, on='key', how='outer')

# Multiple keys
pd.merge(df1, df2, on=['key1', 'key2'])

# Suffixes for overlapping columns
pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
```

### Join

```python
# Join on index
df1.join(df2, how='left')

# Join on columns
df1.set_index('key').join(df2.set_index('key'), how='left')
```

### Concatenate

```python
# Concatenate along rows
pd.concat([df1, df2], axis=0)

# Concatenate along columns
pd.concat([df1, df2], axis=1)

# With keys (hierarchical index)
pd.concat([df1, df2], keys=['first', 'second'])

# Ignore index
pd.concat([df1, df2], ignore_index=True)
```

---

## Data Cleaning

### Handling Missing Values

```python
# Check for missing values
df.isna()              # Boolean DataFrame
df.isnull()            # Same as isna()
df.notna()             # Opposite
df.isna().sum()        # Count missing per column

# Drop missing values
df.dropna()            # Drop rows with any NaN
df.dropna(axis=1)      # Drop columns with any NaN
df.dropna(subset=['col1', 'col2'])  # Drop rows where specific columns are NaN
df.dropna(thresh=2)    # Keep rows with at least 2 non-null values

# Fill missing values
df.fillna(0)           # Fill with scalar
df.fillna({'col1': 0, 'col2': 'unknown'})  # Fill different columns differently
df.fillna(method='ffill')  # Forward fill
df.fillna(method='bfill')  # Backward fill
df.fillna(df.mean())   # Fill with mean

# Interpolate
df.interpolate()       # Linear interpolation
df.interpolate(method='polynomial', order=2)
```

### Data Type Conversion

```python
# Convert data types
df['age'] = df['age'].astype(int)
df['age'] = df['age'].astype('float64')
df = df.astype({'col1': int, 'col2': float})

# Convert to numeric (with error handling)
pd.to_numeric(df['col'], errors='coerce')  # Invalid -> NaN
pd.to_numeric(df['col'], errors='ignore')  # Invalid -> keep as is

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Convert to category (memory efficient)
df['category'] = df['category'].astype('category')
```

### String Operations

```python
# String methods (via .str accessor)
df['name'].str.upper()
df['name'].str.lower()
df['name'].str.strip()
df['name'].str.replace('old', 'new')
df['name'].str.split(' ')
df['name'].str.contains('pattern')
df['name'].str.startswith('A')
df['name'].str.endswith('Z')
df['name'].str.len()
df['name'].str.find('substring')

# Extract with regex
df['email'].str.extract(r'(\w+)@(\w+)\.(\w+)')
```

---

## Time Series Operations

### Date/Time Index

```python
# Create date range
pd.date_range('2024-01-01', '2024-12-31', freq='D')
pd.date_range('2024-01-01', periods=365, freq='D')

# Set date as index
df.set_index('date', inplace=True)

# Resample (downsample/upsample)
df.resample('D').sum()      # Daily
df.resample('W').mean()     # Weekly
df.resample('M').sum()      # Monthly
df.resample('Q').sum()      # Quarterly
df.resample('Y').sum()      # Yearly
```

### Date/Time Operations

```python
# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.day_name()

# Date arithmetic
df['date'] + pd.Timedelta(days=7)
df['date'] - pd.Timedelta(hours=2)
df['date'].diff()           # Difference between consecutive dates

# Date filtering
df[df['date'] >= '2024-01-01']
df[df['date'].between('2024-01-01', '2024-12-31')]
df[df['date'].dt.year == 2024]
df[df['date'].dt.month == 12]
```

---

## Advanced Operations

### Apply Functions

```python
# Apply to Series
df['age'].apply(lambda x: x * 2)
df['name'].apply(len)

# Apply to DataFrame rows
df.apply(lambda row: row['col1'] + row['col2'], axis=1)

# Apply to DataFrame columns
df.apply(lambda col: col.mean(), axis=0)

# Applymap (element-wise)
df.applymap(lambda x: x * 2 if pd.notna(x) else x)
```

### Map and Replace

```python
# Map (Series)
df['status'].map({'A': 'Active', 'I': 'Inactive'})

# Replace
df['status'].replace({'A': 'Active', 'I': 'Inactive'})
df.replace(0, np.nan)  # Replace value
df.replace([0, 1, 2], [10, 11, 12])  # Replace multiple values
```

### Cut and Qcut

```python
# Cut (bins)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 65, 100], 
                         labels=['Child', 'Young', 'Adult', 'Senior'])

# Qcut (quantile-based bins)
df['salary_quartile'] = pd.qcut(df['salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### Stack and Unstack

```python
# Stack (columns to rows)
df_stacked = df.stack()

# Unstack (rows to columns)
df_unstacked = df_stacked.unstack()
```

---

## Performance Optimization

### Efficient Data Types

```python
# Use appropriate dtypes
df['id'] = df['id'].astype('int32')      # Instead of int64
df['category'] = df['category'].astype('category')  # Categorical

# Check memory usage
df.memory_usage(deep=True)
```

### Vectorization

```python
# Use vectorized operations instead of loops
# Bad
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * 2

# Good
df['new_col'] = df['col1'] * 2
```

### Chunking Large Files

```python
# Read large files in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)
```

### Using Query

```python
# Query is often faster than boolean indexing
df.query('age > 30 and city == "NYC"')  # Faster
df[(df['age'] > 30) & (df['city'] == 'NYC')]  # Slower
```

---

## Common Patterns

### Conditional Logic

```python
# Using np.where
df['status'] = np.where(df['age'] >= 18, 'Adult', 'Minor')

# Using np.select (multiple conditions)
conditions = [
    df['age'] < 18,
    (df['age'] >= 18) & (df['age'] < 65),
    df['age'] >= 65
]
choices = ['Minor', 'Adult', 'Senior']
df['age_group'] = np.select(conditions, choices)

# Using apply
df['age_group'] = df['age'].apply(lambda x: 'Adult' if x >= 18 else 'Minor')
```

### Rolling Windows

```python
# Rolling mean
df['rolling_mean'] = df['sales'].rolling(window=7).mean()

# Rolling sum
df['rolling_sum'] = df['sales'].rolling(window=30).sum()

# Expanding window
df['expanding_mean'] = df['sales'].expanding().mean()

# Custom function
df['rolling_custom'] = df['sales'].rolling(window=7).apply(lambda x: x.max() - x.min())
```

### Cross Tabulation

```python
# Create cross tabulation
pd.crosstab(df['category'], df['status'])
pd.crosstab(df['category'], df['status'], margins=True)
pd.crosstab(df['category'], df['status'], normalize='index')  # Percentages
```

---

## Best Practices

### 1. Use Vectorized Operations

```python
# Good: Vectorized
df['new_col'] = df['col1'] * df['col2']

# Bad: Loop
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col1'] * df.loc[i, 'col2']
```

### 2. Avoid Chained Indexing

```python
# Bad: Chained indexing
df[df['age'] > 30]['name'] = 'New Name'

# Good: Use loc
df.loc[df['age'] > 30, 'name'] = 'New Name'
```

### 3. Use Appropriate Data Types

```python
# Use category for low-cardinality strings
df['status'] = df['status'].astype('category')

# Use int32 instead of int64 when possible
df['id'] = df['id'].astype('int32')
```

### 4. Handle Missing Values Early

```python
# Check for missing values first
print(df.isna().sum())

# Decide on strategy: drop, fill, or interpolate
df = df.dropna(subset=['critical_col'])
df['optional_col'] = df['optional_col'].fillna(0)
```

### 5. Use Copy When Needed

```python
# When creating new DataFrame from existing
df_new = df[df['age'] > 30].copy()  # Avoid SettingWithCopyWarning
```

---

## Summary

**Core Operations**:
- **Data Structures**: Series, DataFrame
- **Data Loading**: CSV, Excel, JSON, SQL, Parquet
- **Selection**: loc, iloc, boolean indexing, query
- **Manipulation**: groupby, pivot, merge, join, concat

**Data Cleaning**:
- **Missing Values**: dropna, fillna, interpolate
- **Data Types**: astype, to_numeric, to_datetime
- **Duplicates**: drop_duplicates
- **String Operations**: str accessor methods

**Advanced Topics**:
- **Time Series**: date_range, resample, dt accessor
- **Apply Functions**: apply, applymap, transform
- **Performance**: vectorization, chunking, efficient dtypes

**Best Practices**:
- Use vectorized operations
- Avoid chained indexing
- Use appropriate data types
- Handle missing values early
- Use copy when creating new DataFrames

Mastering Pandas enables efficient data manipulation and analysis for data science and engineering tasks!

