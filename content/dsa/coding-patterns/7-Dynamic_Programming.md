+++
title = "Dynamic Programming"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Complete guide to Dynamic Programming pattern with templates in C++ and Python. Covers memoization, tabulation, 1D/2D DP, state transitions, and optimization techniques with LeetCode problem references."
+++

---

## Introduction

Dynamic Programming (DP) is a powerful optimization technique that solves complex problems by breaking them down into simpler subproblems and storing the results to avoid redundant calculations. It's essential for solving optimization problems efficiently.

This guide provides templates and patterns for Dynamic Programming with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Dynamic Programming

- **Overlapping subproblems**: Same subproblems are solved multiple times
- **Optimal substructure**: Optimal solution contains optimal solutions to subproblems
- **Optimization problems**: Finding minimum/maximum/count of something
- **Decision problems**: Making choices that lead to optimal solution
- **Sequence problems**: Working with arrays, strings, sequences
- **Grid problems**: 2D DP for matrix/grid traversal

### Time & Space Complexity

- **Time Complexity**: O(n) to O(n²) or O(n³) depending on problem dimensions
- **Space Complexity**: O(n) to O(n²) for DP table, can often be optimized to O(1)

---

## Pattern Variations

### Variation 1: 1D DP (Linear)

Single dimension DP array for linear problems.

**Use Cases**:
- Fibonacci sequence
- Climbing stairs
- House robber
- Coin change

### Variation 2: 2D DP (Matrix)

Two-dimensional DP table for grid/matrix problems.

**Use Cases**:
- Unique paths
- Edit distance
- Longest common subsequence
- Minimum path sum

### Variation 3: Memoization (Top-Down)

Recursive approach with memoization to cache results.

**Use Cases**:
- When subproblems are not visited in order
- When only some subproblems are needed
- More intuitive recursive solutions

### Variation 4: Tabulation (Bottom-Up)

Iterative approach building DP table from base cases.

**Use Cases**:
- When all subproblems need to be solved
- Better space optimization opportunities
- Avoiding recursion stack overflow

---

## Template 1: 1D DP - Linear Problems

**Key Points**:
- Define state: dp[i] represents solution for subproblem ending at i
- Base case: dp[0] and/or dp[1]
- Recurrence relation: dp[i] = f(dp[i-1], dp[i-2], ...)
- Return dp[n] or dp[n-1]
- **Time Complexity**: O(n) - iterate through array once
- **Space Complexity**: O(n) for DP array, can optimize to O(1) if only previous values needed

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int linearDP(vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;
    if (n == 1) return arr[0];
    
    vector<int> dp(n);
    
    // Base cases
    dp[0] = arr[0];
    dp[1] = max(arr[0], arr[1]);
    
    // Fill DP table
    for (int i = 2; i < n; i++) {
        dp[i] = max(dp[i-1], dp[i-2] + arr[i]);
    }
    
    return dp[n-1];
}

// Space-optimized version
int linearDPOptimized(vector<int>& arr) {
    int n = arr.size();
    if (n == 0) return 0;
    if (n == 1) return arr[0];
    
    int prev2 = arr[0];
    int prev1 = max(arr[0], arr[1]);
    
    for (int i = 2; i < n; i++) {
        int current = max(prev1, prev2 + arr[i]);
        prev2 = prev1;
        prev1 = current;
    }
    
    return prev1;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def linear_dp(arr):
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return arr[0]
    
    dp = [0] * n
    
    # Base cases
    dp[0] = arr[0]
    dp[1] = max(arr[0], arr[1])
    
    # Fill DP table
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + arr[i])
    
    return dp[n-1]

# Space-optimized version
def linear_dp_optimized(arr):
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return arr[0]
    
    prev2 = arr[0]
    prev1 = max(arr[0], arr[1])
    
    for i in range(2, n):
        current = max(prev1, prev2 + arr[i])
        prev2 = prev1
        prev1 = current
    
    return prev1
```

</details>

### Related Problems

- [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [House Robber](https://leetcode.com/problems/house-robber/)
- [House Robber II](https://leetcode.com/problems/house-robber-ii/)
- [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)
- [Decode Ways](https://leetcode.com/problems/decode-ways/)

---

## Template 2: 2D DP - Grid/Matrix Problems

**Key Points**:
- Define state: dp[i][j] represents solution for subproblem at position (i, j)
- Base case: dp[0][0] or first row/column
- Recurrence relation: dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)
- Return dp[m-1][n-1] or last cell
- **Time Complexity**: O(m×n) - iterate through entire grid
- **Space Complexity**: O(m×n) for DP table, can optimize to O(min(m,n)) or O(1)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int gridDP(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();
    
    vector<vector<int>> dp(m, vector<int>(n));
    
    // Base case: starting position
    dp[0][0] = grid[0][0];
    
    // Fill first row
    for (int j = 1; j < n; j++) {
        dp[0][j] = dp[0][j-1] + grid[0][j];
    }
    
    // Fill first column
    for (int i = 1; i < m; i++) {
        dp[i][0] = dp[i-1][0] + grid[i][0];
    }
    
    // Fill rest of the grid
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
        }
    }
    
    return dp[m-1][n-1];
}

// Space-optimized version (if only previous row needed)
int gridDPOptimized(vector<vector<int>>& grid) {
    int m = grid.size();
    int n = grid[0].size();
    
    vector<int> prev(n);
    prev[0] = grid[0][0];
    
    for (int j = 1; j < n; j++) {
        prev[j] = prev[j-1] + grid[0][j];
    }
    
    for (int i = 1; i < m; i++) {
        vector<int> curr(n);
        curr[0] = prev[0] + grid[i][0];
        
        for (int j = 1; j < n; j++) {
            curr[j] = min(prev[j], curr[j-1]) + grid[i][j];
        }
        
        prev = curr;
    }
    
    return prev[n-1];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def grid_dp(grid):
    m = len(grid)
    n = len(grid[0])
    
    dp = [[0] * n for _ in range(m)]
    
    # Base case: starting position
    dp[0][0] = grid[0][0]
    
    # Fill first row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    
    # Fill first column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    
    # Fill rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    
    return dp[m-1][n-1]

# Space-optimized version
def grid_dp_optimized(grid):
    m = len(grid)
    n = len(grid[0])
    
    prev = [0] * n
    prev[0] = grid[0][0]
    
    for j in range(1, n):
        prev[j] = prev[j-1] + grid[0][j]
    
    for i in range(1, m):
        curr = [0] * n
        curr[0] = prev[0] + grid[i][0]
        
        for j in range(1, n):
            curr[j] = min(prev[j], curr[j-1]) + grid[i][j]
        
        prev = curr
    
    return prev[n-1]
```

</details>

### Related Problems

- [Unique Paths](https://leetcode.com/problems/unique-paths/)
- [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)
- [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
- [Maximum Path Sum](https://leetcode.com/problems/maximum-path-sum/)
- [Dungeon Game](https://leetcode.com/problems/dungeon-game/)

---

## Template 3: Memoization (Top-Down DP)

**Key Points**:
- Use recursion with memoization
- Check memo before computing
- Store result in memo after computing
- More intuitive for some problems
- **Time Complexity**: O(n) to O(n²) depending on subproblems
- **Space Complexity**: O(n) for memo + O(n) for recursion stack = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int memoizationDP(vector<int>& arr, int index, vector<int>& memo) {
    // Base case
    if (index < 0) return 0;
    if (index == 0) return arr[0];
    
    // Check memo
    if (memo[index] != -1) {
        return memo[index];
    }
    
    // Compute and store in memo
    memo[index] = max(
        memoizationDP(arr, index - 1, memo),
        memoizationDP(arr, index - 2, memo) + arr[index]
    );
    
    return memo[index];
}

int solveWithMemoization(vector<int>& arr) {
    int n = arr.size();
    vector<int> memo(n, -1);
    return memoizationDP(arr, n - 1, memo);
}

// Using unordered_map for non-sequential indices
unordered_map<string, int> memo;

int dpWithMap(string s, int i, int j) {
    string key = to_string(i) + "," + to_string(j);
    
    if (memo.find(key) != memo.end()) {
        return memo[key];
    }
    
    // Base case
    if (i > j) return 0;
    if (i == j) return 1;
    
    // Compute
    if (s[i] == s[j]) {
        memo[key] = 2 + dpWithMap(s, i + 1, j - 1);
    } else {
        memo[key] = max(dpWithMap(s, i + 1, j), dpWithMap(s, i, j - 1));
    }
    
    return memo[key];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def memoization_dp(arr, index, memo):
    # Base case
    if index < 0:
        return 0
    if index == 0:
        return arr[0]
    
    # Check memo
    if memo[index] != -1:
        return memo[index]
    
    # Compute and store in memo
    memo[index] = max(
        memoization_dp(arr, index - 1, memo),
        memoization_dp(arr, index - 2, memo) + arr[index]
    )
    
    return memo[index]

def solve_with_memoization(arr):
    n = len(arr)
    memo = [-1] * n
    return memoization_dp(arr, n - 1, memo)

# Using dictionary for non-sequential indices
memo = {}

def dp_with_dict(s, i, j):
    key = (i, j)
    
    if key in memo:
        return memo[key]
    
    # Base case
    if i > j:
        return 0
    if i == j:
        return 1
    
    # Compute
    if s[i] == s[j]:
        memo[key] = 2 + dp_with_dict(s, i + 1, j - 1)
    else:
        memo[key] = max(dp_with_dict(s, i + 1, j), dp_with_dict(s, i, j - 1))
    
    return memo[key]
```

</details>

### Related Problems

- [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
- [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
- [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [Coin Change](https://leetcode.com/problems/coin-change/)

---

## Template 4: String DP (LCS, Edit Distance)

**Key Points**:
- 2D DP table: dp[i][j] for strings s1[0..i] and s2[0..j]
- Base case: empty strings (dp[0][j] and dp[i][0])
- Recurrence based on character matching
- **Time Complexity**: O(m×n) where m and n are string lengths
- **Space Complexity**: O(m×n) for DP table, can optimize to O(min(m,n))

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Longest Common Subsequence
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length();
    int n = text2.length();
    
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    
    return dp[m][n];
}

// Edit Distance
int editDistance(string word1, string word2) {
    int m = word1.length();
    int n = word2.length();
    
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    
    // Base cases
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i; // Delete all characters
    }
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j; // Insert all characters
    }
    
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1]; // No operation needed
            } else {
                dp[i][j] = 1 + min({
                    dp[i-1][j],     // Delete
                    dp[i][j-1],     // Insert
                    dp[i-1][j-1]    // Replace
                });
            }
        }
    }
    
    return dp[m][n];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Longest Common Subsequence
def longest_common_subsequence(text1, text2):
    m = len(text1)
    n = len(text2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Edit Distance
def edit_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Delete
                    dp[i][j-1],     # Insert
                    dp[i-1][j-1]    # Replace
                )
    
    return dp[m][n]
```

</details>

### Related Problems

- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [Edit Distance](https://leetcode.com/problems/edit-distance/)
- [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [Interleaving String](https://leetcode.com/problems/interleaving-string/)
- [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)

---

## Template 5: Knapsack Problems

**Key Points**:
- 2D DP: dp[i][w] = max value using first i items with weight w
- Base case: dp[0][w] = 0 for all w
- Recurrence: include item or skip item
- **Time Complexity**: O(n×W) where n is items and W is capacity
- **Space Complexity**: O(n×W) for DP table, can optimize to O(W)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// 0/1 Knapsack
int knapsack01(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(capacity + 1, 0));
    
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= capacity; w++) {
            if (weights[i-1] <= w) {
                // Can include item: max of including or excluding
                dp[i][w] = max(
                    dp[i-1][w],  // Exclude item
                    dp[i-1][w - weights[i-1]] + values[i-1]  // Include item
                );
            } else {
                // Cannot include item
                dp[i][w] = dp[i-1][w];
            }
        }
    }
    
    return dp[n][capacity];
}

// Space-optimized version
int knapsack01Optimized(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<int> dp(capacity + 1, 0);
    
    for (int i = 0; i < n; i++) {
        // Iterate backwards to avoid using updated values
        for (int w = capacity; w >= weights[i]; w--) {
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
        }
    }
    
    return dp[capacity];
}

// Unbounded Knapsack (can use items multiple times)
int unboundedKnapsack(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<int> dp(capacity + 1, 0);
    
    for (int w = 1; w <= capacity; w++) {
        for (int i = 0; i < n; i++) {
            if (weights[i] <= w) {
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i]);
            }
        }
    }
    
    return dp[capacity];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# 0/1 Knapsack
def knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                # Can include item: max of including or excluding
                dp[i][w] = max(
                    dp[i-1][w],  # Exclude item
                    dp[i-1][w - weights[i-1]] + values[i-1]  # Include item
                )
            else:
                # Cannot include item
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]

# Space-optimized version
def knapsack_01_optimized(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Iterate backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

# Unbounded Knapsack (can use items multiple times)
def unbounded_knapsack(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(n):
            if weights[i] <= w:
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

</details>

### Related Problems

- [Coin Change](https://leetcode.com/problems/coin-change/)
- [Coin Change 2](https://leetcode.com/problems/coin-change-2/)
- [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
- [Target Sum](https://leetcode.com/problems/target-sum/)
- [Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/)

---

## Key Takeaways

1. **Identify DP Pattern**: Look for overlapping subproblems and optimal substructure
2. **State Definition**: Clearly define what dp[i] or dp[i][j] represents
3. **Base Cases**: Handle edge cases (empty, single element, etc.)
4. **Recurrence Relation**: Derive how to compute current state from previous states
5. **Space Optimization**: Often can reduce space from O(n²) to O(n) or O(1)
6. **Memoization vs Tabulation**: Memoization is top-down, tabulation is bottom-up
7. **Direction Matters**: In 2D DP, fill in correct order (row by row or column by column)

---

## Common Mistakes

1. **Wrong state definition**: Not clearly defining what dp[i] represents
2. **Missing base cases**: Forgetting to initialize base cases
3. **Wrong recurrence**: Incorrectly deriving the recurrence relation
4. **Index errors**: Off-by-one errors in array indexing
5. **Not optimizing space**: Using O(n²) when O(n) is possible
6. **Wrong iteration order**: In space-optimized 2D DP, wrong direction can use updated values

---

## Practice Problems by Difficulty

### Easy
- [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/)
- [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
- [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

### Medium
- [House Robber](https://leetcode.com/problems/house-robber/)
- [House Robber II](https://leetcode.com/problems/house-robber-ii/)
- [Unique Paths](https://leetcode.com/problems/unique-paths/)
- [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/)
- [Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/)
- [Coin Change](https://leetcode.com/problems/coin-change/)
- [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [Edit Distance](https://leetcode.com/problems/edit-distance/)
- [Decode Ways](https://leetcode.com/problems/decode-ways/)
- [Word Break](https://leetcode.com/problems/word-break/)
- [Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

### Hard
- [Dungeon Game](https://leetcode.com/problems/dungeon-game/)
- [Longest Palindromic Subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)
- [Edit Distance](https://leetcode.com/problems/edit-distance/)
- [Interleaving String](https://leetcode.com/problems/interleaving-string/)
- [Distinct Subsequences](https://leetcode.com/problems/distinct-subsequences/)
- [Burst Balloons](https://leetcode.com/problems/burst-balloons/)
- [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)

---

## References

* [LeetCode Dynamic Programming Tag](https://leetcode.com/tag/dynamic-programming/)
* [Dynamic Programming (Wikipedia)](https://en.wikipedia.org/wiki/Dynamic_programming)

