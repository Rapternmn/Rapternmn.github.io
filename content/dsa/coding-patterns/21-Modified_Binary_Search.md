+++
title = "Modified Binary Search"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 21
description = "Complete guide to Modified Binary Search pattern with templates in C++ and Python. Covers rotated array search, search in 2D matrix, find peak element, and variations of binary search with LeetCode problem references."
+++

---

## Introduction

Modified Binary Search extends the standard binary search to handle variations like rotated arrays, 2D matrices, finding peaks, and searching in answer space. It adapts the binary search logic to different problem constraints.

This guide provides templates and patterns for Modified Binary Search with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Modified Binary Search

- **Rotated array**: Search in rotated sorted array
- **2D matrix**: Search in sorted 2D matrix
- **Peak finding**: Find peak element in array
- **Answer space**: Binary search on answer space
- **Boundary search**: Find first/last occurrence

### Time & Space Complexity

- **Time Complexity**: O(log n) - binary search complexity
- **Space Complexity**: O(1) - iterative, or O(log n) for recursive

---

## Pattern Variations

### Variation 1: Rotated Array Search

Search in array that's been rotated.

**Use Cases**:
- Search in Rotated Sorted Array
- Find Minimum in Rotated Sorted Array

### Variation 2: 2D Matrix Search

Search in sorted 2D matrix.

**Use Cases**:
- Search a 2D Matrix
- Search a 2D Matrix II

### Variation 3: Answer Space Binary Search

Binary search on answer space instead of array.

**Use Cases**:
- Koko Eating Bananas
- Split Array Largest Sum

---

## Template 1: Search in Rotated Sorted Array

**Key Points**:
- One half is always sorted
- Check which half is sorted
- If target in sorted half, search there; else search other half
- **Time Complexity**: O(log n) - binary search
- **Space Complexity**: O(1) - iterative

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int searchRotated(vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            return mid;
        }
        
        // Left half is sorted
        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        // Right half is sorted
        else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}

// Find Minimum in Rotated Sorted Array
int findMin(vector<int>& nums) {
    int left = 0;
    int right = nums.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        // Right half is unsorted, minimum is there
        if (nums[mid] > nums[right]) {
            left = mid + 1;
        }
        // Left half is unsorted or mid is minimum
        else {
            right = mid;
        }
    }
    
    return nums[left];
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def search_rotated(nums, target):
    left = 0
    right = len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

# Find Minimum in Rotated Sorted Array
def find_min(nums):
    left = 0
    right = len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # Right half is unsorted, minimum is there
        if nums[mid] > nums[right]:
            left = mid + 1
        # Left half is unsorted or mid is minimum
        else:
            right = mid
    
    return nums[left]
```

</details>

### Related Problems

- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

---

## Template 2: Search in 2D Matrix

**Key Points**:
- Treat 2D matrix as 1D array
- Convert 1D index to 2D coordinates
- Or use two binary searches (row, then column)
- **Time Complexity**: O(log(m×n)) or O(log m + log n)
- **Space Complexity**: O(1)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Search in sorted 2D matrix (treat as 1D)
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int m = matrix.size();
    int n = matrix[0].size();
    int left = 0;
    int right = m * n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int row = mid / n;
        int col = mid % n;
        int val = matrix[row][col];
        
        if (val == target) {
            return true;
        } else if (val < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return false;
}

// Search in 2D Matrix II (each row and column sorted)
bool searchMatrixII(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int m = matrix.size();
    int n = matrix[0].size();
    int row = 0;
    int col = n - 1;
    
    while (row < m && col >= 0) {
        if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] > target) {
            col--; // Move left
        } else {
            row++; // Move down
        }
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Search in sorted 2D matrix (treat as 1D)
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    
    m = len(matrix)
    n = len(matrix[0])
    left = 0
    right = m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        row = mid // n
        col = mid % n
        val = matrix[row][col]
        
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# Search in 2D Matrix II (each row and column sorted)
def search_matrix_ii(matrix, target):
    if not matrix or not matrix[0]:
        return False
    
    m = len(matrix)
    n = len(matrix[0])
    row = 0
    col = n - 1
    
    while row < m and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return False
```

</details>

### Related Problems

- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)

---

## Template 3: Find Peak Element

**Key Points**:
- Peak: element greater than neighbors
- Compare mid with mid+1
- If mid < mid+1, peak is in right half
- If mid > mid+1, peak is in left half
- **Time Complexity**: O(log n) - binary search
- **Space Complexity**: O(1)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int findPeakElement(vector<int>& nums) {
    int left = 0;
    int right = nums.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        // If mid < mid+1, peak is in right half
        if (nums[mid] < nums[mid + 1]) {
            left = mid + 1;
        }
        // If mid > mid+1, peak is in left half (including mid)
        else {
            right = mid;
        }
    }
    
    return left;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def find_peak_element(nums):
    left = 0
    right = len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        # If mid < mid+1, peak is in right half
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        # If mid > mid+1, peak is in left half (including mid)
        else:
            right = mid
    
    return left
```

</details>

### Related Problems

- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [Peak Index in a Mountain Array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

---

## Template 4: Binary Search on Answer Space

**Key Points**:
- Binary search on possible answers, not array
- Define feasible function to check if answer is valid
- Find minimum/maximum valid answer
- **Time Complexity**: O(n log(max_answer)) typically
- **Space Complexity**: O(1)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Koko Eating Bananas
int minEatingSpeed(vector<int>& piles, int h) {
    int left = 1;
    int right = *max_element(piles.begin(), piles.end());
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (canFinish(piles, h, mid)) {
            right = mid; // Try smaller speed
        } else {
            left = mid + 1; // Need larger speed
        }
    }
    
    return left;
}

bool canFinish(vector<int>& piles, int h, int speed) {
    int hours = 0;
    for (int pile : piles) {
        hours += (pile + speed - 1) / speed; // Ceiling division
    }
    return hours <= h;
}

// Split Array Largest Sum
int splitArray(vector<int>& nums, int k) {
    int left = *max_element(nums.begin(), nums.end());
    int right = accumulate(nums.begin(), nums.end(), 0);
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (canSplit(nums, k, mid)) {
            right = mid; // Try smaller sum
        } else {
            left = mid + 1; // Need larger sum
        }
    }
    
    return left;
}

bool canSplit(vector<int>& nums, int k, int maxSum) {
    int count = 1;
    int currentSum = 0;
    
    for (int num : nums) {
        if (currentSum + num > maxSum) {
            count++;
            currentSum = num;
        } else {
            currentSum += num;
        }
    }
    
    return count <= k;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Koko Eating Bananas
def min_eating_speed(piles, h):
    left = 1
    right = max(piles)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_finish(piles, h, mid):
            right = mid  # Try smaller speed
        else:
            left = mid + 1  # Need larger speed
    
    return left

def can_finish(piles, h, speed):
    hours = 0
    for pile in piles:
        hours += (pile + speed - 1) // speed  # Ceiling division
    return hours <= h

# Split Array Largest Sum
def split_array(nums, k):
    left = max(nums)
    right = sum(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if can_split(nums, k, mid):
            right = mid  # Try smaller sum
        else:
            left = mid + 1  # Need larger sum
    
    return left

def can_split(nums, k, max_sum):
    count = 1
    current_sum = 0
    
    for num in nums:
        if current_sum + num > max_sum:
            count += 1
            current_sum = num
        else:
            current_sum += num
    
    return count <= k
```

</details>

### Related Problems

- [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
- [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

---

## Template 5: Find First/Last Position

**Key Points**:
- Use binary search to find boundaries
- Find left boundary: keep moving right when found
- Find right boundary: keep moving left when found
- **Time Complexity**: O(log n) - binary search
- **Space Complexity**: O(1)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> searchRange(vector<int>& nums, int target) {
    int first = findFirst(nums, target);
    if (first == -1) {
        return {-1, -1};
    }
    int last = findLast(nums, target);
    return {first, last};
}

int findFirst(vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            right = mid - 1; // Continue searching left
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

int findLast(vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            left = mid + 1; // Continue searching right
        } else if (nums[mid] < target) {
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
def search_range(nums, target):
    first = find_first(nums, target)
    if first == -1:
        return [-1, -1]
    last = find_last(nums, target)
    return [first, last]

def find_first(nums, target):
    left = 0
    right = len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def find_last(nums, target):
    left = 0
    right = len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

</details>

### Related Problems

- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search for a Range](https://leetcode.com/problems/search-for-a-range/)

---

## Key Takeaways

1. **Rotated Array**: One half is always sorted, check which half contains target
2. **2D Matrix**: Treat as 1D or use two-pointer approach
3. **Peak Finding**: Compare mid with mid+1 to determine search direction
4. **Answer Space**: Binary search on possible answers, not array
5. **Boundary Search**: Continue searching in one direction after finding target
6. **Edge Cases**: Handle empty array, single element, all same elements

---

## Common Mistakes

1. **Wrong comparison**: Comparing wrong elements in rotated array
2. **Index calculation**: Wrong 1D to 2D conversion in matrix search
3. **Boundary conditions**: Off-by-one errors in left/right updates
4. **Infinite loop**: Not updating left/right correctly
5. **Answer space**: Wrong range for binary search on answer space

---

## Practice Problems by Difficulty

### Medium
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Search in Rotated Sorted Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Search a 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/)
- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)

### Hard
- [Capacity To Ship Packages Within D Days](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

---

## References

* [LeetCode Binary Search Tag](https://leetcode.com/tag/binary-search/)
* [Binary Search (Wikipedia)](https://en.wikipedia.org/wiki/Binary_search_algorithm)

