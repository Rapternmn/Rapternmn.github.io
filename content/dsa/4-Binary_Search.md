+++
title = "Binary Search"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Complete guide to Binary Search pattern with templates in C++ and Python. Covers standard binary search, search in rotated arrays, search in 2D matrices, and search space optimization with LeetCode problem references."
+++

---

## Introduction

Binary Search is a powerful divide-and-conquer algorithm that efficiently searches for an element in a sorted array by repeatedly dividing the search space in half. It reduces time complexity from O(n) to O(log n), making it essential for optimization problems.

This guide provides templates and patterns for Binary Search with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Binary Search

- **Sorted arrays**: Searching in sorted arrays or sorted data structures
- **Search space reduction**: Problems where we can eliminate half the search space
- **Optimization problems**: Finding minimum/maximum value that satisfies a condition
- **Rotated arrays**: Searching in rotated sorted arrays
- **2D matrices**: Searching in sorted 2D matrices
- **Answer space search**: When answer lies in a range and we can check validity

### Time & Space Complexity

- **Time Complexity**: O(log n) - eliminates half the search space each iteration
- **Space Complexity**: O(1) for iterative, O(log n) for recursive

---

## Pattern Variations

### Variation 1: Standard Binary Search

Search for exact target in sorted array.

**Use Cases**:
- Search in sorted array
- Find element in sorted list
- Basic binary search problems

### Variation 2: Search for Boundary/Insertion Point

Find first/last occurrence or insertion position.

**Use Cases**:
- Find first/last position of element
- Search insert position
- Lower/upper bound problems

### Variation 3: Search in Rotated Array

Search in array that has been rotated.

**Use Cases**:
- Search in rotated sorted array
- Find minimum in rotated array
- Search in rotated array with duplicates

### Variation 4: Binary Search on Answer Space

Search for answer in a range using validity function.

**Use Cases**:
- Koko eating bananas
- Split array largest sum
- Capacity to ship packages
- Minimize maximum value

---

## Template 1: Standard Binary Search

**Key Points**:
- Maintain left and right boundaries
- Calculate mid without overflow: `mid = left + (right - left) / 2`
- Compare target with mid element
- Narrow search space based on comparison
- **Time Complexity**: O(log n) - eliminates half the search space each iteration
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int binarySearch(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2; // Avoid overflow
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1; // Target not found
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found
```

</details>

<details>
<summary><strong>📋 C++ STL Functions</strong></summary>

```cpp
#include <bits/stdc++.h>
using namespace std;

// Check if element exists
bool binarySearchSTL(vector<int>& arr, int target) {
    return binary_search(arr.begin(), arr.end(), target);
}

// Find insertion position (lower bound)
int lowerBoundSTL(vector<int>& arr, int target) {
    auto it = lower_bound(arr.begin(), arr.end(), target);
    return it - arr.begin(); // Index where target should be inserted
}

// Find upper bound (first element greater than target)
int upperBoundSTL(vector<int>& arr, int target) {
    auto it = upper_bound(arr.begin(), arr.end(), target);
    return it - arr.begin();
}

// Check if found and get index
int binarySearchWithIndex(vector<int>& arr, int target) {
    auto it = lower_bound(arr.begin(), arr.end(), target);
    if (it != arr.end() && *it == target) {
        return it - arr.begin();
    }
    return -1; // Not found
}

// Get both bounds at once
pair<int, int> equalRangeSTL(vector<int>& arr, int target) {
    auto range = equal_range(arr.begin(), arr.end(), target);
    int first = range.first - arr.begin();
    int last = range.second - arr.begin() - 1; // Last position of target
    return {first, last};
}
```

</details>

<details>
<summary><strong>🐍 Python bisect Module</strong></summary>

```python
import bisect

# Check if element exists (using bisect_left)
def binary_search_bisect(arr, target):
    index = bisect.bisect_left(arr, target)
    return index < len(arr) and arr[index] == target

# Find insertion position (lower bound)
index = bisect.bisect_left(arr, target)

# Find upper bound (insertion point for element greater than target)
index = bisect.bisect_right(arr, target)

# Get both bounds
left = bisect.bisect_left(arr, target)
right = bisect.bisect_right(arr, target)
# Range is [left, right) - elements equal to target are at [left, right)
```
</details>

### Related Problems

- [Binary Search](https://leetcode.com/problems/binary-search/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/)
- [Sqrt(x)](https://leetcode.com/problems/sqrtx/)
- [Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/)

---

## Template 2: Find First/Last Position (Lower/Upper Bound)

**Key Points**:
- For first position: when found, don't return, continue searching left
- For last position: when found, don't return, continue searching right
- Use separate templates for lower_bound and upper_bound
- **Time Complexity**: O(log n) - binary search eliminates half each iteration
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Find first position (lower bound)
int findFirstPosition(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            right = mid - 1; // Continue searching left
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// Find last position (upper bound - 1)
int findLastPosition(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            result = mid;
            left = mid + 1; // Continue searching right
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Find first position (lower bound)
def find_first_position(arr, target):
    left = 0
    right = len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Find last position (upper bound - 1)
def find_last_position(arr, target):
    left = 0
    right = len(arr) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

</details>

<details>
<summary><strong>📋 C++ STL Functions</strong></summary>

```cpp
#include <bits/stdc++.h>
using namespace std;

// Find first position (lower_bound)
int findFirstPositionSTL(vector<int>& arr, int target) {
    auto it = lower_bound(arr.begin(), arr.end(), target);
    if (it != arr.end() && *it == target) {
        return it - arr.begin();
    }
    return -1;
}

// Find last position (upper_bound - 1)
int findLastPositionSTL(vector<int>& arr, int target) {
    auto it = upper_bound(arr.begin(), arr.end(), target);
    if (it != arr.begin()) {
        it--;
        if (*it == target) {
            return it - arr.begin();
        }
    }
    return -1;
}

// Using equal_range (most efficient)
pair<int, int> findFirstLastSTL(vector<int>& arr, int target) {
    auto range = equal_range(arr.begin(), arr.end(), target);
    if (range.first != arr.end() && *range.first == target) {
        int firstPos = range.first - arr.begin();
        int lastPos = range.second - arr.begin() - 1;
        return {firstPos, lastPos};
    }
    return {-1, -1};
}

// Complete solution using STL
vector<int> searchRangeSTL(vector<int>& arr, int target) {
    auto range = equal_range(arr.begin(), arr.end(), target);
    if (range.first == range.second) {
        return {-1, -1}; // Not found
    }
    return {
        (int)(range.first - arr.begin()),
        (int)(range.second - arr.begin() - 1)
    };
}
```

</details>

<details>
<summary><strong>🐍 Python bisect Module</strong></summary>

```python
import bisect

# Find first position
def find_first_position_bisect(arr, target):
    left = bisect.bisect_left(arr, target)
    if left < len(arr) and arr[left] == target:
        return left
    return -1

# Find last position
def find_last_position_bisect(arr, target):
    right = bisect.bisect_right(arr, target)
    if right > 0 and arr[right - 1] == target:
        return right - 1
    return -1

# Complete solution using bisect
def search_range_bisect(arr, target):
    left = bisect.bisect_left(arr, target)
    right = bisect.bisect_right(arr, target)
    
    if left == right:
        return [-1, -1]  # Not found
    
    return [left, right - 1]
```
</details>

### Related Problems

- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search for a Range](https://leetcode.com/problems/search-for-a-range/)
- [Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)

---

## Template 3: Search in Rotated Sorted Array

**Key Points**:
- Check which half is sorted (left or right)
- Determine if target is in sorted half
- If in sorted half, search there; otherwise search other half
- Handle duplicates by skipping them
- **Time Complexity**: O(log n) average, O(n) worst case with duplicates
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int searchRotatedArray(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        // Left half is sorted
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (arr[mid] < target && target <= arr[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}

// Handle duplicates
bool searchRotatedArrayWithDuplicates(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return true;
        }
        
        // Skip duplicates
        if (arr[left] == arr[mid] && arr[mid] == arr[right]) {
            left++;
            right--;
            continue;
        }
        
        // Left half is sorted
        if (arr[left] <= arr[mid]) {
            if (arr[left] <= target && target < arr[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (arr[mid] < target && target <= arr[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def search_rotated_array(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Handle duplicates
def search_rotated_array_with_duplicates(arr, target):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return True
        
        # Skip duplicates
        if arr[left] == arr[mid] == arr[right]:
            left += 1
            right -= 1
            continue
        
        # Left half is sorted
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return False
```

</details>

### Related Problems

- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

---

## Template 4: Binary Search on Answer Space

**Key Points**:
- Answer lies in a range [left, right]
- Define a validity function to check if answer is valid
- Binary search on the answer space
- Find minimum/maximum valid answer
- **Time Complexity**: O(log(max - min) × f(n)) where f(n) is validity check complexity
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Example: Find minimum capacity to ship packages within D days
bool canShip(vector<int>& weights, int capacity, int days) {
    int currentDays = 1;
    int currentWeight = 0;
    
    for (int weight : weights) {
        if (currentWeight + weight > capacity) {
            currentDays++;
            currentWeight = weight;
            if (currentDays > days) return false;
        } else {
            currentWeight += weight;
        }
    }
    
    return true;
}

int shipWithinDays(vector<int>& weights, int days) {
    int left = *max_element(weights.begin(), weights.end());
    int right = accumulate(weights.begin(), weights.end(), 0);
    int result = right;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (canShip(weights, mid, days)) {
            result = mid;
            right = mid - 1; // Try smaller capacity
        } else {
            left = mid + 1; // Need larger capacity
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Example: Find minimum capacity to ship packages within D days
def can_ship(weights, capacity, days):
    current_days = 1
    current_weight = 0
    
    for weight in weights:
        if current_weight + weight > capacity:
            current_days += 1
            current_weight = weight
            if current_days > days:
                return False
        else:
            current_weight += weight
    
    return True

def ship_within_days(weights, days):
    left = max(weights)
    right = sum(weights)
    result = right
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if can_ship(weights, mid, days):
            result = mid
            right = mid - 1  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return result
```

</details>

### Related Problems

- [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)
- [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
- [Minimize Maximum Distance to Gas Station](https://leetcode.com/problems/minimize-max-distance-to-gas-station/)
- [Divide Chocolate](https://leetcode.com/problems/divide-chocolate/)

---

## Template 5: Search in 2D Matrix

**Key Points**:
- Treat 2D matrix as 1D array using index conversion
- Convert 1D index to 2D: `row = index / cols`, `col = index % cols`
- Apply standard binary search with index conversion
- **Time Complexity**: O(log(m×n)) where m is rows and n is cols
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    int left = 0;
    int right = rows * cols - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int row = mid / cols;
        int col = mid % cols;
        int midValue = matrix[row][col];
        
        if (midValue == target) {
            return true;
        } else if (midValue < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    
    rows = len(matrix)
    cols = len(matrix[0])
    left = 0
    right = rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        row = mid // cols
        col = mid % cols
        mid_value = matrix[row][col]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

</details>

### Related Problems

- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

---

## Key Takeaways

1. **Overflow Prevention**: Always use `mid = left + (right - left) / 2` instead of `(left + right) / 2`
2. **Boundary Conditions**: Carefully handle `left <= right` vs `left < right`
3. **First/Last Position**: When found, continue searching in the direction needed
4. **Rotated Array**: Check which half is sorted, then determine search direction
5. **Answer Space**: Binary search on the answer when validity can be checked
6. **2D Matrix**: Convert 2D index to 1D for binary search
7. **Duplicates**: Skip duplicates when they prevent determining sorted half

---

## Common Mistakes

1. **Integer overflow**: Using `(left + right) / 2` instead of `left + (right - left) / 2`
2. **Off-by-one errors**: Wrong boundary conditions (`<=` vs `<`)
3. **Infinite loops**: Not updating left/right correctly
4. **Wrong search direction**: In rotated arrays, not correctly identifying sorted half
5. **Not handling duplicates**: In rotated arrays with duplicates
6. **Index conversion errors**: In 2D matrix problems

---

## Practice Problems by Difficulty

### Easy
- [Binary Search](https://leetcode.com/problems/binary-search/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/)
- [Sqrt(x)](https://leetcode.com/problems/sqrtx/)
- [Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/)
- [First Bad Version](https://leetcode.com/problems/first-bad-version/)

### Medium
- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)
- [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

### Hard
- [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [Find Minimum in Rotated Sorted Array II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

---

## References

* [LeetCode Binary Search Tag](https://leetcode.com/tag/binary-search/)
* [Algorithm Patterns: Binary Search](https://leetcode.com/discuss/study-guide/786126/Python-or-Binary-Search-or-Explained)

