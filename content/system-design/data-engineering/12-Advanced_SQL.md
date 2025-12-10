+++
title = "Advanced SQL"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 12
description = "Advanced SQL: Window functions, CTEs, complex joins, subqueries, query optimization, analytical functions, and advanced SQL techniques for data engineering."
+++

---

## Introduction

Advanced SQL skills are essential for data engineering, enabling efficient data transformation, complex analytics, and optimized queries. This guide covers advanced SQL concepts and techniques used in data warehousing, ETL processes, and analytical systems.

---

## Window Functions

### What are Window Functions?

**Window Functions** perform calculations across a set of table rows related to the current row, without collapsing rows into a single output row.

**Key Characteristics**:
- Preserve all rows
- Calculate over a "window" of rows
- Use `OVER()` clause
- More efficient than self-joins

### Window Function Syntax

```sql
function_name() OVER (
    [PARTITION BY column]
    [ORDER BY column]
    [ROWS/RANGE BETWEEN ...]
)
```

### Common Window Functions

**1. Ranking Functions**

```sql
-- ROW_NUMBER: Sequential numbering
SELECT 
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- RANK: Ranking with gaps
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- DENSE_RANK: Ranking without gaps
SELECT 
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;
```

**2. Aggregate Window Functions**

```sql
-- Running totals
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) as running_total
FROM daily_sales;

-- Moving averages
SELECT 
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7d
FROM daily_sales;

-- Partitioned aggregates
SELECT 
    department,
    employee,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;
```

**3. Value Functions**

```sql
-- LAG: Previous row value
SELECT 
    date,
    sales,
    LAG(sales, 1) OVER (ORDER BY date) as prev_sales
FROM daily_sales;

-- LEAD: Next row value
SELECT 
    date,
    sales,
    LEAD(sales, 1) OVER (ORDER BY date) as next_sales
FROM daily_sales;

-- FIRST_VALUE: First value in window
SELECT 
    date,
    sales,
    FIRST_VALUE(sales) OVER (
        PARTITION BY month 
        ORDER BY date
    ) as month_first_sales
FROM daily_sales;

-- LAST_VALUE: Last value in window
SELECT 
    date,
    sales,
    LAST_VALUE(sales) OVER (
        PARTITION BY month 
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) as month_last_sales
FROM daily_sales;
```

### Window Frame Specifications

**ROWS vs RANGE**:
- **ROWS**: Physical rows
- **RANGE**: Logical values

**Frame Options**:
```sql
-- Current row and preceding
ROWS BETWEEN 2 PRECEDING AND CURRENT ROW

-- Current row and following
ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING

-- All preceding
ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

-- All following
ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING

-- Sliding window
ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
```

---

## Common Table Expressions (CTEs)

### What are CTEs?

**CTEs** (Common Table Expressions) are temporary result sets defined within a query that can be referenced multiple times.

**Benefits**:
- Improve readability
- Enable recursion
- Reusable subqueries
- Better organization

### Basic CTE Syntax

```sql
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;
```

### Multiple CTEs

```sql
WITH 
    sales_summary AS (
        SELECT 
            product_id,
            SUM(quantity) as total_sold
        FROM sales
        GROUP BY product_id
    ),
    product_info AS (
        SELECT 
            product_id,
            product_name,
            price
        FROM products
    )
SELECT 
    p.product_name,
    s.total_sold,
    s.total_sold * p.price as revenue
FROM sales_summary s
JOIN product_info p ON s.product_id = p.product_id;
```

### Recursive CTEs

**Use Cases**: Hierarchical data, sequences, graph traversal

```sql
WITH RECURSIVE employee_hierarchy AS (
    -- Anchor member
    SELECT 
        employee_id,
        manager_id,
        name,
        1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive member
    SELECT 
        e.employee_id,
        e.manager_id,
        e.name,
        eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy;
```

---

## Advanced Joins

### Join Types

**1. INNER JOIN**
```sql
SELECT *
FROM table1 t1
INNER JOIN table2 t2 ON t1.id = t2.id;
```

**2. LEFT JOIN (LEFT OUTER JOIN)**
```sql
SELECT *
FROM table1 t1
LEFT JOIN table2 t2 ON t1.id = t2.id;
```

**3. RIGHT JOIN (RIGHT OUTER JOIN)**
```sql
SELECT *
FROM table1 t1
RIGHT JOIN table2 t2 ON t1.id = t2.id;
```

**4. FULL OUTER JOIN**
```sql
SELECT *
FROM table1 t1
FULL OUTER JOIN table2 t2 ON t1.id = t2.id;
```

**5. CROSS JOIN**
```sql
SELECT *
FROM table1
CROSS JOIN table2;
```

### Self Joins

```sql
-- Find employees and their managers
SELECT 
    e.name as employee,
    m.name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```

### Multiple Table Joins

```sql
SELECT 
    o.order_id,
    c.customer_name,
    p.product_name,
    oi.quantity,
    oi.price
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id;
```

---

## Subqueries

### Types of Subqueries

**1. Scalar Subquery**
```sql
-- Returns single value
SELECT 
    name,
    salary,
    (SELECT AVG(salary) FROM employees) as avg_salary
FROM employees;
```

**2. Correlated Subquery**
```sql
-- References outer query
SELECT 
    e.name,
    e.salary,
    e.department
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department = e.department
);
```

**3. EXISTS Subquery**
```sql
-- Check existence
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
);
```

**4. IN Subquery**
```sql
-- Check membership
SELECT *
FROM products
WHERE category_id IN (
    SELECT category_id
    FROM categories
    WHERE active = true
);
```

**5. Derived Tables (Inline Views)**
```sql
SELECT *
FROM (
    SELECT 
        department,
        AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
) dept_avg
WHERE avg_salary > 50000;
```

---

## Advanced Aggregations

### GROUP BY Extensions

**1. GROUPING SETS**
```sql
SELECT 
    department,
    region,
    SUM(sales) as total_sales
FROM sales
GROUP BY GROUPING SETS (
    (department, region),
    (department),
    (region),
    ()
);
```

**2. ROLLUP**
```sql
-- Hierarchical aggregation
SELECT 
    year,
    quarter,
    month,
    SUM(sales) as total_sales
FROM sales
GROUP BY ROLLUP (year, quarter, month);
```

**3. CUBE**
```sql
-- All combinations
SELECT 
    department,
    region,
    SUM(sales) as total_sales
FROM sales
GROUP BY CUBE (department, region);
```

### Filtered Aggregations

```sql
-- Aggregate with conditions
SELECT 
    department,
    SUM(sales) as total_sales,
    SUM(sales) FILTER (WHERE region = 'North') as north_sales,
    COUNT(*) FILTER (WHERE sales > 1000) as high_sales_count
FROM sales
GROUP BY department;
```

---

## Pivoting and Unpivoting

### PIVOT

**Convert rows to columns**:

```sql
-- PostgreSQL example
SELECT *
FROM (
    SELECT department, employee, salary
    FROM employees
) AS source
PIVOT (
    AVG(salary)
    FOR department IN ('Sales', 'Engineering', 'Marketing')
) AS pivot_table;
```

**Using CASE for PIVOT**:
```sql
SELECT 
    employee,
    SUM(CASE WHEN department = 'Sales' THEN salary END) as sales_salary,
    SUM(CASE WHEN department = 'Engineering' THEN salary END) as eng_salary,
    SUM(CASE WHEN department = 'Marketing' THEN salary END) as mkt_salary
FROM employees
GROUP BY employee;
```

### UNPIVOT

**Convert columns to rows**:

```sql
SELECT 
    employee,
    department,
    salary
FROM (
    SELECT 
        employee,
        sales_salary,
        eng_salary,
        mkt_salary
    FROM employee_salaries
) AS source
UNPIVOT (
    salary FOR department IN (
        sales_salary AS 'Sales',
        eng_salary AS 'Engineering',
        mkt_salary AS 'Marketing'
    )
) AS unpivot_table;
```

---

## Query Optimization

### Index Usage

**1. Create Appropriate Indexes**
```sql
-- Single column index
CREATE INDEX idx_customer_id ON orders(customer_id);

-- Composite index
CREATE INDEX idx_customer_date ON orders(customer_id, order_date);

-- Partial index
CREATE INDEX idx_active_orders ON orders(order_date) 
WHERE status = 'active';
```

**2. Use Indexes in WHERE Clauses**
```sql
-- Good: Uses index
SELECT * FROM orders WHERE customer_id = 123;

-- Bad: Can't use index
SELECT * FROM orders WHERE UPPER(customer_name) = 'JOHN';
```

### Query Execution Plans

**EXPLAIN ANALYZE**:
```sql
EXPLAIN ANALYZE
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date > '2024-01-01';
```

**Key Metrics**:
- Execution time
- Rows scanned
- Index usage
- Join algorithms

### Optimization Techniques

**1. Avoid SELECT ***
```sql
-- Bad
SELECT * FROM large_table;

-- Good
SELECT id, name, email FROM large_table;
```

**2. Use LIMIT**
```sql
-- Limit result set
SELECT * FROM orders ORDER BY order_date DESC LIMIT 100;
```

**3. Filter Early**
```sql
-- Filter before join
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_date > '2024-01-01'  -- Filter early
AND c.status = 'active';
```

**4. Use EXISTS Instead of IN**
```sql
-- Better for large subqueries
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id
);
```

**5. Avoid Functions in WHERE**
```sql
-- Bad: Can't use index
SELECT * FROM orders 
WHERE YEAR(order_date) = 2024;

-- Good: Can use index
SELECT * FROM orders 
WHERE order_date >= '2024-01-01' 
AND order_date < '2025-01-01';
```

---

## Advanced SQL Patterns

### 1. Gaps and Islands

**Find gaps in sequences**:
```sql
WITH numbered AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY id) as rn
    FROM numbers
)
SELECT 
    n1.id + 1 as gap_start,
    n2.id - 1 as gap_end
FROM numbered n1
JOIN numbered n2 ON n1.rn = n2.rn - 1
WHERE n2.id - n1.id > 1;
```

**Find islands (consecutive groups)**:
```sql
WITH grouped AS (
    SELECT 
        value,
        value - ROW_NUMBER() OVER (ORDER BY value) as grp
    FROM numbers
)
SELECT 
    MIN(value) as island_start,
    MAX(value) as island_end,
    COUNT(*) as island_size
FROM grouped
GROUP BY grp;
```

### 2. Running Totals and Averages

```sql
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) as running_total,
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7d
FROM daily_sales;
```

### 3. Top N per Group

```sql
-- Top 3 employees per department
WITH ranked AS (
    SELECT 
        department,
        employee,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) as rn
    FROM employees
)
SELECT *
FROM ranked
WHERE rn <= 3;
```

### 4. Date/Time Operations

```sql
-- Date arithmetic
SELECT 
    order_date,
    order_date + INTERVAL '7 days' as week_later,
    DATE_TRUNC('month', order_date) as month_start,
    EXTRACT(YEAR FROM order_date) as year
FROM orders;

-- Date ranges
SELECT *
FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-12-31';
```

---

## Analytical Functions

### Statistical Functions

```sql
-- Percentiles
SELECT 
    department,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) as p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY salary) as p95
FROM employees
GROUP BY department;

-- Standard deviation
SELECT 
    department,
    AVG(salary) as avg_salary,
    STDDEV(salary) as stddev_salary
FROM employees
GROUP BY department;
```

### Time Series Functions

```sql
-- Period-over-period comparison
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(amount) as sales
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT 
    month,
    sales,
    LAG(sales, 1) OVER (ORDER BY month) as prev_month,
    sales - LAG(sales, 1) OVER (ORDER BY month) as change,
    (sales - LAG(sales, 1) OVER (ORDER BY month)) * 100.0 / 
        LAG(sales, 1) OVER (ORDER BY month) as pct_change
FROM monthly_sales;
```

---

## Best Practices

### 1. Code Organization

- Use CTEs for complex queries
- Add comments for complex logic
- Format queries consistently
- Use meaningful aliases

### 2. Performance

- Use indexes appropriately
- Filter early in queries
- Avoid unnecessary joins
- Use EXPLAIN to analyze plans

### 3. Readability

- Break complex queries into CTEs
- Use consistent naming
- Format for readability
- Document complex logic

### 4. Maintainability

- Use parameterized queries
- Avoid hard-coded values
- Version control SQL scripts
- Test queries thoroughly

---

## Common Pitfalls

### 1. NULL Handling

```sql
-- Handle NULLs properly
SELECT 
    COALESCE(sales, 0) as sales,
    NULLIF(division, 0) as safe_division
FROM sales_data;
```

### 2. Cartesian Products

```sql
-- Always specify join conditions
-- Bad: Missing join condition
SELECT * FROM table1, table2;

-- Good: Explicit join
SELECT * FROM table1 JOIN table2 ON table1.id = table2.id;
```

### 3. Aggregation Errors

```sql
-- All non-aggregated columns must be in GROUP BY
-- Bad
SELECT department, name, AVG(salary) FROM employees;

-- Good
SELECT department, AVG(salary) FROM employees GROUP BY department;
```

---

## Key Takeaways

- Window functions enable powerful analytical queries
- CTEs improve query readability and enable recursion
- Advanced joins handle complex relationships
- Subqueries provide flexible querying options
- Query optimization is crucial for performance
- Use appropriate patterns for common problems
- Follow best practices for maintainability
- Test and optimize queries regularly

