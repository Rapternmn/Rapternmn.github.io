+++
title = "SQL Essentials: Commonly Used Functions & Advanced Topics"
date = 2025-12-09T10:00:00+05:30
draft = false
weight = 3
description = "Comprehensive SQL guide: commonly used functions, window functions, CTEs, joins, subqueries, and advanced SQL techniques for data analysis and database operations."
+++

---

## Introduction

SQL (Structured Query Language) is the standard language for managing and querying relational databases. This guide covers commonly used SQL functions, advanced topics like window functions, and practical patterns for efficient data analysis.

**Key Topics**:
- Basic SQL operations (SELECT, INSERT, UPDATE, DELETE)
- Commonly used functions (string, date, aggregate, conditional)
- Joins and relationships
- Window functions
- CTEs and subqueries
- Query optimization

---

## Basic SQL Operations

### Data Definition Language (DDL)

**Create, alter, and drop database objects**:

```sql
-- Create table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_name_email ON users(name, email);  -- Composite index

-- Alter table
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users ALTER COLUMN age SET DEFAULT 0;
ALTER TABLE users DROP COLUMN phone;

-- Drop table
DROP TABLE IF EXISTS users;
```

### Data Manipulation Language (DML)

**Insert, update, and delete data**:

```sql
-- Insert single row
INSERT INTO users (name, email, age) 
VALUES ('John Doe', 'john@example.com', 30);

-- Insert multiple rows
INSERT INTO users (name, email, age) 
VALUES 
    ('Jane Smith', 'jane@example.com', 25),
    ('Bob Johnson', 'bob@example.com', 35);

-- Insert from select
INSERT INTO users (name, email, age)
SELECT name, email, age FROM temp_users;

-- Update
UPDATE users 
SET age = 31, email = 'john.new@example.com'
WHERE id = 1;

-- Delete
DELETE FROM users WHERE id = 1;
DELETE FROM users WHERE age < 18;

-- Truncate (faster than DELETE, no rollback)
TRUNCATE TABLE users;
```

### SELECT Basics

```sql
-- Basic select
SELECT * FROM users;
SELECT id, name, email FROM users;

-- Filtering
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
SELECT * FROM users WHERE name IN ('John', 'Jane', 'Bob');
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE age IS NOT NULL;

-- Sorting
SELECT * FROM users ORDER BY age DESC;
SELECT * FROM users ORDER BY name ASC, age DESC;

-- Limiting
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;  -- Pagination

-- Distinct
SELECT DISTINCT age FROM users;
SELECT DISTINCT name, email FROM users;
```

---

## Commonly Used Functions

### String Functions

```sql
-- Concatenation
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users;
SELECT first_name || ' ' || last_name AS full_name FROM users;  -- PostgreSQL

-- Length
SELECT LENGTH(name) AS name_length FROM users;
SELECT CHAR_LENGTH(name) AS name_length FROM users;

-- Case conversion
SELECT UPPER(name) AS upper_name FROM users;
SELECT LOWER(email) AS lower_email FROM users;
SELECT INITCAP(name) AS title_case FROM users;  -- PostgreSQL

-- Substring
SELECT SUBSTRING(email, 1, 5) AS prefix FROM users;
SELECT SUBSTRING(email FROM 1 FOR 5) AS prefix FROM users;  -- PostgreSQL
SELECT LEFT(name, 3) AS first_3_chars FROM users;  -- PostgreSQL
SELECT RIGHT(email, 10) AS domain FROM users;  -- PostgreSQL

-- Trimming
SELECT TRIM('  hello  ') AS trimmed;  -- 'hello'
SELECT LTRIM('  hello  ') AS left_trimmed;  -- 'hello  '
SELECT RTRIM('  hello  ') AS right_trimmed;  -- '  hello'
SELECT TRIM(BOTH 'x' FROM 'xxhelloxx');  -- 'hello'

-- Replace
SELECT REPLACE(email, '@gmail.com', '@company.com') AS new_email FROM users;

-- Position and find
SELECT POSITION('@' IN email) AS at_position FROM users;
SELECT STRPOS(email, '@') AS at_position FROM users;  -- PostgreSQL
SELECT LOCATE('@', email) AS at_position FROM users;  -- MySQL

-- Padding
SELECT LPAD(name, 20, ' ') AS left_padded FROM users;
SELECT RPAD(name, 20, ' ') AS right_padded FROM users;

-- Split
SELECT SPLIT_PART(email, '@', 1) AS username FROM users;  -- PostgreSQL
SELECT SUBSTRING_INDEX(email, '@', 1) AS username FROM users;  -- MySQL
```

### Date and Time Functions

```sql
-- Current date/time
SELECT CURRENT_DATE;
SELECT CURRENT_TIMESTAMP;
SELECT NOW();
SELECT CURRENT_TIME;

-- Extract components
SELECT EXTRACT(YEAR FROM created_at) AS year FROM users;
SELECT EXTRACT(MONTH FROM created_at) AS month FROM users;
SELECT EXTRACT(DAY FROM created_at) AS day FROM users;
SELECT EXTRACT(DOW FROM created_at) AS day_of_week FROM users;  -- PostgreSQL

-- Date arithmetic
SELECT created_at + INTERVAL '7 days' AS week_later FROM users;
SELECT created_at - INTERVAL '1 month' AS month_ago FROM users;
SELECT created_at + INTERVAL '2 hours' AS two_hours_later FROM users;

-- Date formatting
SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS formatted_date FROM users;  -- PostgreSQL
SELECT DATE_FORMAT(created_at, '%Y-%m-%d') AS formatted_date FROM users;  -- MySQL
SELECT FORMAT(created_at, 'yyyy-MM-dd') AS formatted_date FROM users;  -- SQL Server

-- Date truncation
SELECT DATE_TRUNC('month', created_at) AS month_start FROM users;  -- PostgreSQL
SELECT DATE_TRUNC('year', created_at) AS year_start FROM users;
SELECT DATE_TRUNC('day', created_at) AS day_start FROM users;

-- Age calculation
SELECT AGE(created_at) AS account_age FROM users;  -- PostgreSQL
SELECT DATEDIFF(NOW(), created_at) AS days_old FROM users;  -- MySQL
SELECT DATEDIFF(day, created_at, GETDATE()) AS days_old FROM users;  -- SQL Server

-- Date comparison
SELECT * FROM users WHERE created_at >= '2024-01-01';
SELECT * FROM users WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';
SELECT * FROM users WHERE DATE(created_at) = CURRENT_DATE;
```

### Aggregate Functions

```sql
-- Basic aggregates
SELECT COUNT(*) AS total_users FROM users;
SELECT COUNT(DISTINCT age) AS unique_ages FROM users;
SELECT SUM(amount) AS total_amount FROM orders;
SELECT AVG(price) AS avg_price FROM products;
SELECT MIN(price) AS min_price FROM products;
SELECT MAX(price) AS max_price FROM products;

-- Group by
SELECT department, COUNT(*) AS emp_count 
FROM employees 
GROUP BY department;

SELECT department, AVG(salary) AS avg_salary 
FROM employees 
GROUP BY department
HAVING AVG(salary) > 50000;  -- Filter groups

-- Multiple grouping
SELECT department, city, COUNT(*) AS emp_count
FROM employees
GROUP BY department, city;

-- Aggregate with expressions
SELECT 
    department,
    COUNT(*) AS total,
    AVG(salary) AS avg_salary,
    SUM(salary) AS total_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department;
```

### Conditional Functions

```sql
-- CASE statement
SELECT 
    name,
    CASE 
        WHEN age < 18 THEN 'Minor'
        WHEN age < 65 THEN 'Adult'
        ELSE 'Senior'
    END AS age_group
FROM users;

-- Simple CASE
SELECT 
    status,
    CASE status
        WHEN 'A' THEN 'Active'
        WHEN 'I' THEN 'Inactive'
        ELSE 'Unknown'
    END AS status_description
FROM accounts;

-- COALESCE (first non-null value)
SELECT COALESCE(phone, email, 'No contact') AS contact FROM users;

-- NULLIF (returns NULL if equal)
SELECT NULLIF(division, 0) AS safe_division FROM calculations;

-- IF (MySQL)
SELECT IF(age >= 18, 'Adult', 'Minor') AS age_group FROM users;  -- MySQL

-- IIF (SQL Server)
SELECT IIF(age >= 18, 'Adult', 'Minor') AS age_group FROM users;  -- SQL Server

-- GREATEST and LEAST
SELECT GREATEST(10, 20, 30) AS max_value;  -- 30
SELECT LEAST(10, 20, 30) AS min_value;     -- 10
```

### Mathematical Functions

```sql
-- Basic math
SELECT ABS(-10) AS absolute;           -- 10
SELECT ROUND(3.14159, 2) AS rounded;    -- 3.14
SELECT CEIL(3.14) AS ceiling;           -- 4
SELECT FLOOR(3.14) AS floor;            -- 3
SELECT TRUNC(3.14159, 2) AS truncated;  -- 3.14

-- Power and roots
SELECT POWER(2, 3) AS power;            -- 8
SELECT SQRT(16) AS square_root;         -- 4

-- Logarithms
SELECT LOG(10) AS natural_log;
SELECT LOG10(100) AS log_base_10;      -- 2

-- Random
SELECT RANDOM() AS random_value;        -- PostgreSQL
SELECT RAND() AS random_value;          -- MySQL, SQL Server

-- Modulo
SELECT MOD(10, 3) AS remainder;         -- 1
SELECT 10 % 3 AS remainder;             -- 1
```

---

## Joins

### Inner Join

**Returns only matching rows**:

```sql
SELECT u.name, o.order_id, o.amount
FROM users u
INNER JOIN orders o ON u.id = o.user_id;

-- Multiple joins
SELECT u.name, o.order_id, p.product_name
FROM users u
INNER JOIN orders o ON u.id = o.user_id
INNER JOIN order_items oi ON o.id = oi.order_id
INNER JOIN products p ON oi.product_id = p.id;
```

### Left Join

**Returns all rows from left table, matching rows from right**:

```sql
SELECT u.name, o.order_id, o.amount
FROM users u
LEFT JOIN orders o ON u.id = o.user_id;

-- Find users with no orders
SELECT u.name
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE o.id IS NULL;
```

### Right Join

**Returns all rows from right table, matching rows from left**:

```sql
SELECT u.name, o.order_id, o.amount
FROM users u
RIGHT JOIN orders o ON u.id = o.user_id;
```

### Full Outer Join

**Returns all rows from both tables**:

```sql
SELECT u.name, o.order_id, o.amount
FROM users u
FULL OUTER JOIN orders o ON u.id = o.user_id;
```

### Self Join

**Join a table with itself**:

```sql
-- Find employees and their managers
SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### Cross Join

**Cartesian product of two tables**:

```sql
SELECT u.name, p.product_name
FROM users u
CROSS JOIN products p;
```

---

## Window Functions

### What are Window Functions?

**Window functions** perform calculations across a set of table rows related to the current row, without collapsing rows into a single output row.

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

### Ranking Functions

```sql
-- ROW_NUMBER: Sequential numbering (no ties)
SELECT 
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rank
FROM employees;

-- RANK: Ranking with gaps (ties get same rank, next rank skips)
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;

-- DENSE_RANK: Ranking without gaps (ties get same rank, next rank doesn't skip)
SELECT 
    name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;

-- PERCENT_RANK: Relative rank (0 to 1)
SELECT 
    name,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary DESC) AS pct_rank
FROM employees;

-- NTILE: Divide into buckets
SELECT 
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
```

### Aggregate Window Functions

```sql
-- Running totals
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) AS running_total
FROM daily_sales;

-- Moving averages
SELECT 
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;

-- Partitioned aggregates
SELECT 
    department,
    employee,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    MAX(salary) OVER (PARTITION BY department) AS dept_max
FROM employees;

-- Window frame specifications
SELECT 
    date,
    sales,
    SUM(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_sum,
    AVG(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) AS moving_avg_5d
FROM daily_sales;
```

### Value Functions

```sql
-- LAG: Previous row value
SELECT 
    date,
    sales,
    LAG(sales, 1) OVER (ORDER BY date) AS prev_sales,
    sales - LAG(sales, 1) OVER (ORDER BY date) AS change
FROM daily_sales;

-- LEAD: Next row value
SELECT 
    date,
    sales,
    LEAD(sales, 1) OVER (ORDER BY date) AS next_sales
FROM daily_sales;

-- FIRST_VALUE: First value in window
SELECT 
    department,
    employee,
    salary,
    FIRST_VALUE(salary) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS top_salary
FROM employees;

-- LAST_VALUE: Last value in window
SELECT 
    date,
    sales,
    LAST_VALUE(sales) OVER (
        ORDER BY date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_sales
FROM daily_sales;

-- NTH_VALUE: Nth value in window
SELECT 
    department,
    employee,
    salary,
    NTH_VALUE(salary, 2) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS second_highest
FROM employees;
```

### Common Window Function Patterns

**Top N per Group**:

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
        ) AS rn
    FROM employees
)
SELECT *
FROM ranked
WHERE rn <= 3;
```

**Running Totals and Percentages**:

```sql
SELECT 
    date,
    sales,
    SUM(sales) OVER (ORDER BY date) AS running_total,
    SUM(sales) OVER () AS total_sales,
    sales * 100.0 / SUM(sales) OVER () AS pct_of_total
FROM daily_sales;
```

**Period-over-Period Comparison**:

```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS sales
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT 
    month,
    sales,
    LAG(sales, 1) OVER (ORDER BY month) AS prev_month,
    sales - LAG(sales, 1) OVER (ORDER BY month) AS change,
    (sales - LAG(sales, 1) OVER (ORDER BY month)) * 100.0 / 
        LAG(sales, 1) OVER (ORDER BY month) AS pct_change
FROM monthly_sales;
```

---

## Common Table Expressions (CTEs)

### Basic CTE

**Improve readability and enable recursion**:

```sql
WITH high_salary_employees AS (
    SELECT * FROM employees WHERE salary > 100000
)
SELECT * FROM high_salary_employees;
```

### Multiple CTEs

```sql
WITH 
    dept_stats AS (
        SELECT 
            department,
            AVG(salary) AS avg_salary,
            COUNT(*) AS emp_count
        FROM employees
        GROUP BY department
    ),
    high_avg_depts AS (
        SELECT department
        FROM dept_stats
        WHERE avg_salary > 80000
    )
SELECT e.*
FROM employees e
INNER JOIN high_avg_depts h ON e.department = h.department;
```

### Recursive CTE

**For hierarchical data**:

```sql
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level managers
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees reporting to managers
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy;
```

---

## Subqueries

### Scalar Subquery

**Returns single value**:

```sql
SELECT 
    name,
    salary,
    (SELECT AVG(salary) FROM employees) AS avg_salary
FROM employees;

-- In WHERE clause
SELECT * FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### Correlated Subquery

**References outer query**:

```sql
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

### EXISTS Subquery

**Check existence**:

```sql
-- Find customers with orders
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
);

-- Find customers without orders
SELECT *
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.customer_id
);
```

### IN Subquery

**Check membership**:

```sql
SELECT *
FROM products
WHERE category_id IN (
    SELECT category_id
    FROM categories
    WHERE active = true
);
```

### Derived Tables

**Subquery in FROM clause**:

```sql
SELECT *
FROM (
    SELECT 
        department,
        AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department
) dept_avg
WHERE avg_salary > 50000;
```

---

## Advanced Patterns

### Gaps and Islands

**Find gaps in sequences**:

```sql
WITH numbered AS (
    SELECT 
        id,
        ROW_NUMBER() OVER (ORDER BY id) AS rn
    FROM numbers
)
SELECT 
    n1.id + 1 AS gap_start,
    n2.id - 1 AS gap_end
FROM numbered n1
JOIN numbered n2 ON n1.rn = n2.rn - 1
WHERE n2.id - n1.id > 1;
```

**Find islands (consecutive groups)**:

```sql
WITH grouped AS (
    SELECT 
        value,
        value - ROW_NUMBER() OVER (ORDER BY value) AS grp
    FROM numbers
)
SELECT 
    MIN(value) AS island_start,
    MAX(value) AS island_end,
    COUNT(*) AS island_size
FROM grouped
GROUP BY grp;
```

### Pivoting

**Convert rows to columns**:

```sql
-- Using CASE
SELECT 
    employee,
    SUM(CASE WHEN department = 'Sales' THEN salary END) AS sales_salary,
    SUM(CASE WHEN department = 'Engineering' THEN salary END) AS eng_salary,
    SUM(CASE WHEN department = 'Marketing' THEN salary END) AS mkt_salary
FROM employees
GROUP BY employee;

-- Using PIVOT (PostgreSQL, SQL Server)
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

### Unpivoting

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

## Query Optimization Tips

### 1. Use Indexes

```sql
-- Create indexes on frequently queried columns
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_name_email ON users(name, email);

-- Use indexes in WHERE clauses
SELECT * FROM users WHERE email = 'john@example.com';  -- Uses index
```

### 2. Filter Early

```sql
-- Good: Filter before join
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.order_date > '2024-01-01';

-- Better: Filter in subquery first
SELECT *
FROM (
    SELECT * FROM orders WHERE order_date > '2024-01-01'
) o
JOIN customers c ON o.customer_id = c.id;
```

### 3. Avoid SELECT *

```sql
-- Bad
SELECT * FROM large_table;

-- Good
SELECT id, name, email FROM large_table;
```

### 4. Use LIMIT

```sql
-- Limit result set
SELECT * FROM orders ORDER BY order_date DESC LIMIT 100;
```

### 5. Use EXISTS Instead of IN

```sql
-- Better for large subqueries
SELECT *
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o 
    WHERE o.customer_id = c.customer_id
);
```

### 6. Avoid Functions in WHERE

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
- Use consistent naming conventions
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
    COALESCE(sales, 0) AS sales,
    NULLIF(division, 0) AS safe_division
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

## Summary

**Commonly Used Functions**:
- **String**: CONCAT, SUBSTRING, TRIM, REPLACE, UPPER, LOWER
- **Date**: EXTRACT, DATE_TRUNC, DATE arithmetic, formatting
- **Aggregate**: COUNT, SUM, AVG, MIN, MAX, GROUP BY
- **Conditional**: CASE, COALESCE, NULLIF

**Advanced Topics**:
- **Window Functions**: ROW_NUMBER, RANK, LAG, LEAD, running totals
- **CTEs**: Improve readability, enable recursion
- **Subqueries**: Scalar, correlated, EXISTS, IN
- **Joins**: INNER, LEFT, RIGHT, FULL OUTER, SELF

**Best Practices**:
- Use indexes for performance
- Filter early in queries
- Use CTEs for complex queries
- Handle NULLs properly
- Test and optimize queries

Mastering these SQL concepts enables efficient data analysis and database operations!

