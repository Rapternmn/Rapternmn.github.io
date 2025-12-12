+++
title = "Prefix Sum"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 22
description = "Complete guide to Prefix Sum pattern with templates in C++ and Python. Covers 1D prefix sum, 2D prefix sum, difference array, and subarray sum problems with LeetCode problem references."
+++

---

## Introduction

Prefix Sum (also called Cumulative Sum) is a technique that precomputes cumulative sums of array elements to answer range sum queries in O(1) time. It's essential for optimizing problems involving multiple range queries and subarray sums.

This guide provides templates and patterns for the Prefix Sum technique with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Prefix Sum

- **Range Sum Queries**: Multiple queries asking for sum of subarray
- **Subarray Sum Problems**: Finding subarrays with target sum
- **2D Range Queries**: Sum of rectangular regions in matrix
- **Optimization**: Converting O(n) range queries to O(1)
- **Frequency Counting**: Counting subarrays with certain properties

### Time & Space Complexity

- **Preprocessing Time**: O(n) - build prefix array once
- **Query Time**: O(1) - answer each query in constant time
- **Space Complexity**: O(n) - store prefix array
- **Overall**: O(n + q) where q is number of queries (vs O(nq) naive)

---

## Pattern Variations

### Variation 1: 1D Prefix Sum

Precompute cumulative sums for 1D array.

**Use Cases**:
- Range sum queries
- Subarray sum problems
- Finding subarrays with target sum

### Variation 2: 2D Prefix Sum

Precompute cumulative sums for 2D matrix.

**Use Cases**:
- Sum of rectangular regions
- Matrix range queries
- Image processing

### Variation 3: Difference Array

Use difference array for range updates.

**Use Cases**:
- Range addition/subtraction
- Multiple range updates
- Optimize range update queries

---

## Template 1: 1D Prefix Sum

**Key Points**:
- Build prefix array: `prefix[i] = sum of arr[0..i-1]`
- Query sum from i to j: `prefix[j+1] - prefix[i]`
- Handle edge cases (empty array, single element)
- **Time Complexity**: O(n) preprocessing, O(1) per query
- **Space Complexity**: O(n) for prefix array

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class PrefixSum {
private:
    vector<int> prefix;
    
public:
    PrefixSum(vector<int>& arr) {
        int n = arr.size();
        prefix.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + arr[i];
        }
    }
    
    // Query sum from index i to j (inclusive)
    int query(int i, int j) {
        return prefix[j + 1] - prefix[i];
    }
    
    // Get prefix sum up to index i
    int getPrefix(int i) {
        return prefix[i + 1];
    }
};

// Simple function version
vector<int> buildPrefixSum(vector<int>& arr) {
    int n = arr.size();
    vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + arr[i];
    }
    return prefix;
}

// Query sum from i to j
int rangeSum(vector<int>& prefix, int i, int j) {
    return prefix[j + 1] - prefix[i];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class PrefixSum:
    def __init__(self, arr):
        n = len(arr)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + arr[i]
    
    def query(self, i, j):
        """Query sum from index i to j (inclusive)"""
        return self.prefix[j + 1] - self.prefix[i]
    
    def get_prefix(self, i):
        """Get prefix sum up to index i"""
        return self.prefix[i + 1]

# Simple function version
def build_prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

# Query sum from i to j
def range_sum(prefix, i, j):
    return prefix[j + 1] - prefix[i]
```

</details>

### Related Problems

- [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
- [Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)
- [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)
- [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

---

## Template 2: Subarray Sum Equals K

**Key Points**:
- Use prefix sum with hash map
- Track frequency of prefix sums
- Look for `prefix_sum - k` in map
- **Time Complexity**: O(n) - single pass
- **Space Complexity**: O(n) - hash map

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> sumMap;
    sumMap[0] = 1;  // Empty subarray has sum 0
    int prefixSum = 0;
    int count = 0;
    
    for (int num : nums) {
        prefixSum += num;
        if (sumMap.find(prefixSum - k) != sumMap.end()) {
            count += sumMap[prefixSum - k];
        }
        sumMap[prefixSum]++;
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def subarraySum(nums, k):
    sum_map = {0: 1}  # Empty subarray has sum 0
    prefix_sum = 0
    count = 0
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_map:
            count += sum_map[prefix_sum - k]
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count
```

</details>

### Related Problems

- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
- [Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)
- [Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/)

---

## Template 3: 2D Prefix Sum

**Key Points**:
- Build 2D prefix sum matrix
- `prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)`
- Query sum of rectangle: inclusion-exclusion principle
- **Time Complexity**: O(mn) preprocessing, O(1) per query
- **Space Complexity**: O(mn) for prefix matrix

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class NumMatrix {
private:
    vector<vector<int>> prefix;
    
public:
    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        prefix.resize(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                prefix[i + 1][j + 1] = matrix[i][j] 
                    + prefix[i][j + 1] 
                    + prefix[i + 1][j] 
                    - prefix[i][j];
            }
        }
    }
    
    // Sum of rectangle from (row1, col1) to (row2, col2) inclusive
    int sumRegion(int row1, int col1, int row2, int col2) {
        return prefix[row2 + 1][col2 + 1] 
            - prefix[row1][col2 + 1] 
            - prefix[row2 + 1][col1] 
            + prefix[row1][col1];
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class NumMatrix:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (matrix[i][j] 
                    + self.prefix[i][j + 1] 
                    + self.prefix[i + 1][j] 
                    - self.prefix[i][j])
    
    def sum_region(self, row1, col1, row2, col2):
        """Sum of rectangle from (row1, col1) to (row2, col2) inclusive"""
        return (self.prefix[row2 + 1][col2 + 1] 
            - self.prefix[row1][col2 + 1] 
            - self.prefix[row2 + 1][col1] 
            + self.prefix[row1][col1])
```

</details>

### Related Problems

- [Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)
- [Max Sum of Rectangle No Larger Than K](https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/)
- [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

---

## Template 4: Difference Array (Range Updates)

**Key Points**:
- Use difference array for efficient range updates
- Update range [l, r] by adding value: `diff[l] += val, diff[r+1] -= val`
- Reconstruct array by taking prefix sum of difference array
- **Time Complexity**: O(1) per update, O(n) to reconstruct
- **Space Complexity**: O(n) for difference array

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class DifferenceArray {
private:
    vector<int> diff;
    int n;
    
public:
    DifferenceArray(int size) {
        n = size;
        diff.resize(n + 1, 0);
    }
    
    // Add value to range [l, r] (inclusive)
    void updateRange(int l, int r, int val) {
        diff[l] += val;
        if (r + 1 < n) {
            diff[r + 1] -= val;
        }
    }
    
    // Reconstruct array from difference array
    vector<int> getArray() {
        vector<int> arr(n);
        arr[0] = diff[0];
        for (int i = 1; i < n; i++) {
            arr[i] = arr[i - 1] + diff[i];
        }
        return arr;
    }
    
    // Get value at index i
    int getValue(int i) {
        int val = 0;
        for (int j = 0; j <= i; j++) {
            val += diff[j];
        }
        return val;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class DifferenceArray:
    def __init__(self, size):
        self.n = size
        self.diff = [0] * (size + 1)
    
    def update_range(self, l, r, val):
        """Add value to range [l, r] (inclusive)"""
        self.diff[l] += val
        if r + 1 < self.n:
            self.diff[r + 1] -= val
    
    def get_array(self):
        """Reconstruct array from difference array"""
        arr = [0] * self.n
        arr[0] = self.diff[0]
        for i in range(1, self.n):
            arr[i] = arr[i - 1] + self.diff[i]
        return arr
    
    def get_value(self, i):
        """Get value at index i"""
        return sum(self.diff[:i + 1])
```

</details>

### Related Problems

- [Range Addition](https://leetcode.com/problems/range-addition/)
- [Corporate Flight Bookings](https://leetcode.com/problems/corporate-flight-bookings/)
- [Car Pooling](https://leetcode.com/problems/car-pooling/)

---

## Top Problems

### Problem 1: Range Sum Query - Immutable

**Problem**: Design data structure to answer range sum queries.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class NumArray {
private:
    vector<int> prefix;
    
public:
    NumArray(vector<int>& nums) {
        int n = nums.size();
        prefix.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }
    }
    
    int sumRange(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
class NumArray:
    def __init__(self, nums):
        n = len(nums)
        self.prefix = [0] * (n + 1)
        for i in range(n):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
    
    def sumRange(self, left, right):
        return self.prefix[right + 1] - self.prefix[left]
```

</details>

**Time**: O(n) preprocessing, O(1) per query | **Space**: O(n)

**Related**: [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)

---

### Problem 2: Subarray Sum Equals K

**Problem**: Find total number of subarrays with sum equal to k.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int subarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> sumMap;
    sumMap[0] = 1;
    int prefixSum = 0;
    int count = 0;
    
    for (int num : nums) {
        prefixSum += num;
        if (sumMap.find(prefixSum - k) != sumMap.end()) {
            count += sumMap[prefixSum - k];
        }
        sumMap[prefixSum]++;
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def subarraySum(nums, k):
    sum_map = {0: 1}
    prefix_sum = 0
    count = 0
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_map:
            count += sum_map[prefix_sum - k]
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count
```

</details>

**Time**: O(n) | **Space**: O(n)

**Related**: [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

---

### Problem 3: Find Pivot Index

**Problem**: Find pivot index where sum of left equals sum of right.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int pivotIndex(vector<int>& nums) {
    int totalSum = 0;
    for (int num : nums) {
        totalSum += num;
    }
    
    int leftSum = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (leftSum == totalSum - leftSum - nums[i]) {
            return i;
        }
        leftSum += nums[i];
    }
    
    return -1;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def pivotIndex(nums):
    total_sum = sum(nums)
    left_sum = 0
    
    for i in range(len(nums)):
        if left_sum == total_sum - left_sum - nums[i]:
            return i
        left_sum += nums[i]
    
    return -1
```

</details>

**Time**: O(n) | **Space**: O(1)

**Related**: [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)

---

### Problem 4: Range Sum Query 2D - Immutable

**Problem**: Design 2D data structure to answer rectangular region sum queries.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class NumMatrix {
private:
    vector<vector<int>> prefix;
    
public:
    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        prefix.resize(m + 1, vector<int>(n + 1, 0));
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                prefix[i + 1][j + 1] = matrix[i][j] 
                    + prefix[i][j + 1] 
                    + prefix[i + 1][j] 
                    - prefix[i][j];
            }
        }
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return prefix[row2 + 1][col2 + 1] 
            - prefix[row1][col2 + 1] 
            - prefix[row2 + 1][col1] 
            + prefix[row1][col1];
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
class NumMatrix:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m):
            for j in range(n):
                self.prefix[i + 1][j + 1] = (matrix[i][j] 
                    + self.prefix[i][j + 1] 
                    + self.prefix[i + 1][j] 
                    - self.prefix[i][j])
    
    def sumRegion(self, row1, col1, row2, col2):
        return (self.prefix[row2 + 1][col2 + 1] 
            - self.prefix[row1][col2 + 1] 
            - self.prefix[row2 + 1][col1] 
            + self.prefix[row1][col1])
```

</details>

**Time**: O(mn) preprocessing, O(1) per query | **Space**: O(mn)

**Related**: [Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)

---

### Problem 5: Continuous Subarray Sum

**Problem**: Check if array has subarray with sum multiple of k.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool checkSubarraySum(vector<int>& nums, int k) {
    unordered_map<int, int> remainderMap;
    remainderMap[0] = -1;  // Handle case where prefix sum itself is multiple of k
    int prefixSum = 0;
    
    for (int i = 0; i < nums.size(); i++) {
        prefixSum += nums[i];
        int remainder = prefixSum % k;
        
        if (remainderMap.find(remainder) != remainderMap.end()) {
            if (i - remainderMap[remainder] >= 2) {
                return true;
            }
        } else {
            remainderMap[remainder] = i;
        }
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def checkSubarraySum(nums, k):
    remainder_map = {0: -1}
    prefix_sum = 0
    
    for i in range(len(nums)):
        prefix_sum += nums[i]
        remainder = prefix_sum % k
        
        if remainder in remainder_map:
            if i - remainder_map[remainder] >= 2:
                return True
        else:
            remainder_map[remainder] = i
    
    return False
```

</details>

**Time**: O(n) | **Space**: O(min(n, k))

**Related**: [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)

---

### Problem 6: Maximum Size Subarray Sum Equals k

**Problem**: Find maximum length subarray with sum equal to k.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int maxSubArrayLen(vector<int>& nums, int k) {
    unordered_map<int, int> sumMap;
    sumMap[0] = -1;
    int prefixSum = 0;
    int maxLen = 0;
    
    for (int i = 0; i < nums.size(); i++) {
        prefixSum += nums[i];
        if (sumMap.find(prefixSum - k) != sumMap.end()) {
            maxLen = max(maxLen, i - sumMap[prefixSum - k]);
        }
        if (sumMap.find(prefixSum) == sumMap.end()) {
            sumMap[prefixSum] = i;
        }
    }
    
    return maxLen;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def maxSubArrayLen(nums, k):
    sum_map = {0: -1}
    prefix_sum = 0
    max_len = 0
    
    for i in range(len(nums)):
        prefix_sum += nums[i]
        if prefix_sum - k in sum_map:
            max_len = max(max_len, i - sum_map[prefix_sum - k])
        if prefix_sum not in sum_map:
            sum_map[prefix_sum] = i
    
    return max_len
```

</details>

**Time**: O(n) | **Space**: O(n)

**Related**: [Maximum Size Subarray Sum Equals k](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

---

### Problem 7: Range Addition

**Problem**: Apply range updates efficiently using difference array.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> getModifiedArray(int length, vector<vector<int>>& updates) {
    vector<int> diff(length + 1, 0);
    
    for (auto& update : updates) {
        int start = update[0];
        int end = update[1];
        int val = update[2];
        diff[start] += val;
        if (end + 1 < length) {
            diff[end + 1] -= val;
        }
    }
    
    vector<int> result(length);
    result[0] = diff[0];
    for (int i = 1; i < length; i++) {
        result[i] = result[i - 1] + diff[i];
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def getModifiedArray(length, updates):
    diff = [0] * (length + 1)
    
    for start, end, val in updates:
        diff[start] += val
        if end + 1 < length:
            diff[end + 1] -= val
    
    result = [0] * length
    result[0] = diff[0]
    for i in range(1, length):
        result[i] = result[i - 1] + diff[i]
    
    return result
```

</details>

**Time**: O(n + m) where m is number of updates | **Space**: O(n)

**Related**: [Range Addition](https://leetcode.com/problems/range-addition/)

---

### Problem 8: Number of Submatrices That Sum to Target

**Problem**: Count submatrices with sum equal to target.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int numSubmatrixSumTarget(vector<vector<int>>& matrix, int target) {
    int m = matrix.size();
    int n = matrix[0].size();
    int count = 0;
    
    // Build 2D prefix sum
    vector<vector<int>> prefix(m + 1, vector<int>(n + 1, 0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            prefix[i + 1][j + 1] = matrix[i][j] 
                + prefix[i][j + 1] 
                + prefix[i + 1][j] 
                - prefix[i][j];
        }
    }
    
    // For each pair of rows, use 1D subarray sum technique
    for (int r1 = 0; r1 < m; r1++) {
        for (int r2 = r1; r2 < m; r2++) {
            unordered_map<int, int> sumMap;
            sumMap[0] = 1;
            
            for (int c = 0; c < n; c++) {
                int colSum = prefix[r2 + 1][c + 1] - prefix[r1][c + 1];
                if (sumMap.find(colSum - target) != sumMap.end()) {
                    count += sumMap[colSum - target];
                }
                sumMap[colSum]++;
            }
        }
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def numSubmatrixSumTarget(matrix, target):
    m, n = len(matrix), len(matrix[0])
    count = 0
    
    # Build 2D prefix sum
    prefix = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            prefix[i + 1][j + 1] = (matrix[i][j] 
                + prefix[i][j + 1] 
                + prefix[i + 1][j] 
                - prefix[i][j])
    
    # For each pair of rows, use 1D subarray sum technique
    for r1 in range(m):
        for r2 in range(r1, m):
            sum_map = {0: 1}
            for c in range(n):
                col_sum = prefix[r2 + 1][c + 1] - prefix[r1][c + 1]
                if col_sum - target in sum_map:
                    count += sum_map[col_sum - target]
                sum_map[col_sum] = sum_map.get(col_sum, 0) + 1
    
    return count
```

</details>

**Time**: O(m²n) | **Space**: O(mn)

**Related**: [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

---

## Advanced Patterns

### 1. Circular Array Prefix Sum

**Pattern**: Handle circular arrays with prefix sum.

```cpp
int circularSubarraySum(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> extended = nums;
    extended.insert(extended.end(), nums.begin(), nums.end());
    
    vector<int> prefix(2 * n + 1, 0);
    for (int i = 0; i < 2 * n; i++) {
        prefix[i + 1] = prefix[i] + extended[i];
    }
    
    // Use sliding window or hash map technique
    // ...
}
```

---

### 2. Prefix Sum with Modulo

**Pattern**: Use prefix sum with modulo for remainder problems.

```cpp
int subarrayDivisibleByK(vector<int>& nums, int k) {
    unordered_map<int, int> remainderMap;
    remainderMap[0] = 1;
    int prefixSum = 0;
    int count = 0;
    
    for (int num : nums) {
        prefixSum = (prefixSum + num) % k;
        if (prefixSum < 0) prefixSum += k;  // Handle negative
        count += remainderMap[prefixSum];
        remainderMap[prefixSum]++;
    }
    
    return count;
}
```

---

## Key Takeaways

- Prefix sum converts O(n) range queries to O(1)
- Preprocessing: O(n) time, O(n) space
- Query formula: `prefix[j+1] - prefix[i]` for sum from i to j
- Combine with hash map for subarray sum problems
- 2D prefix sum uses inclusion-exclusion principle
- Difference array optimizes range updates
- Practice range query and subarray sum problems
- Time complexity: O(n) preprocessing + O(1) per query

---

## Practice Problems

**Easy**:
- [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/)
- [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/)
- [Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/)

**Medium**:
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/)
- [Range Sum Query 2D - Immutable](https://leetcode.com/problems/range-sum-query-2d-immutable/)

**Hard**:
- [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)
- [Max Sum of Rectangle No Larger Than K](https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/)

