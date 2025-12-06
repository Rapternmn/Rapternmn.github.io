+++
title = "Two Pointers Pattern"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Complete guide to Two Pointers pattern with templates in C++ and Python. Covers opposite ends, same direction, and fast-slow pointer techniques with LeetCode problem references."
+++

---

## Introduction

The Two Pointers technique is a powerful approach for solving array and string problems efficiently. It uses two pointers that traverse the data structure, reducing time complexity from O(n²) to O(n) in many cases.

This guide provides templates and patterns for the Two Pointers technique with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Two Pointers

- **Sorted arrays/strings**: When data is sorted, two pointers can efficiently find pairs
- **Palindrome problems**: Check if string/array is palindrome
- **Pair/triplet finding**: Find pairs or triplets that meet certain conditions
- **Removing duplicates**: Remove duplicates from sorted arrays
- **Partitioning**: Partition array based on conditions
- **Merging**: Merge two sorted arrays

### Time & Space Complexity

- **Time Complexity**: O(n) - single pass through array
- **Space Complexity**: O(1) - only using two pointer variables

---

## Pattern Variations

### Variation 1: Opposite Ends (Most Common)

Two pointers start from opposite ends and move towards each other.

**Use Cases**:
- Sorted array problems
- Palindrome checking
- Finding pairs with target sum
- Container problems

### Variation 2: Same Direction (Sliding Window-like)

Two pointers start from the same end and move in the same direction at different speeds.

**Use Cases**:
- Removing duplicates
- Partitioning arrays
- In-place array modifications

### Variation 3: Fast & Slow Pointers

One pointer moves faster than the other (covered in separate pattern file).

---

## Template 1: Opposite Ends Pattern

**Key Points**:
- Start with `left = 0` and `right = n-1`
- Move pointers based on condition
- Terminate when `left >= right`

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
void twoPointersOppositeEnds(vector<int>& arr) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left < right) {
        // Process elements at left and right
        if (condition) {
            left++;
        } else {
            right--;
        }
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def two_pointers_opposite_ends(arr):
    left = 0
    right = len(arr) - 1
    
    while left < right:
        # Process elements at left and right
        if condition:
            left += 1
        else:
            right -= 1
```

</details>

### Related Problems

- [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [3Sum](https://leetcode.com/problems/3sum/)
- [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
- [4Sum](https://leetcode.com/problems/4sum/)
- [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
- [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
- [Reverse String](https://leetcode.com/problems/reverse-string/)
- [Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/)

---

## Template 2: Same Direction Pattern (Slow-Fast)

**Key Points**:
- `slow` pointer: position to write next valid element
- `fast` pointer: scans through entire array
- Use for in-place modifications

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int twoPointersSameDirection(vector<int>& arr) {
    int slow = 0;
    for (int fast = 0; fast < arr.size(); fast++) {
        if (condition) {
            arr[slow] = arr[fast];
            slow++;
        }
    }
    return slow; // Result length
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def two_pointers_same_direction(arr):
    slow = 0
    for fast in range(len(arr)):
        if condition:
            arr[slow] = arr[fast]
            slow += 1
    return slow  # Result length
```

</details>

### Related Problems

- [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)
- [Remove Element](https://leetcode.com/problems/remove-element/)
- [Move Zeroes](https://leetcode.com/problems/move-zeroes/)
- [Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
- [Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/)

---

## Template 3: Two Pointers for Triplets (3Sum Pattern)

**Key Points**:
- Sort array first
- Fix one element, use two pointers for remaining
- Important to skip duplicates

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> threeSum(vector<int>& arr, int target) {
    sort(arr.begin(), arr.end());
    int n = arr.size();
    vector<vector<int>> result;
    
    for (int i = 0; i < n - 2; i++) {
        // Skip duplicates for first element
        if (i > 0 && arr[i] == arr[i-1]) continue;
        
        int left = i + 1;
        int right = n - 1;
        int currentTarget = target - arr[i];
        
        while (left < right) {
            int sum = arr[left] + arr[right];
            
            if (sum == currentTarget) {
                // Found triplet: arr[i], arr[left], arr[right]
                result.push_back({arr[i], arr[left], arr[right]});
                
                // Skip duplicates
                while (left < right && arr[left] == arr[left + 1]) left++;
                while (left < right && arr[right] == arr[right - 1]) right--;
                
                left++;
                right--;
            } else if (sum < currentTarget) {
                left++;
            } else {
                right--;
            }
        }
    }
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def three_sum(arr, target):
    arr.sort()
    n = len(arr)
    result = []
    
    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and arr[i] == arr[i-1]:
            continue
        
        left = i + 1
        right = n - 1
        current_target = target - arr[i]
        
        while left < right:
            current_sum = arr[left] + arr[right]
            
            if current_sum == current_target:
                # Found triplet: arr[i], arr[left], arr[right]
                result.append([arr[i], arr[left], arr[right]])
                
                # Skip duplicates
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < current_target:
                left += 1
            else:
                right -= 1
    
    return result
```

</details>

### Related Problems

- [3Sum](https://leetcode.com/problems/3sum/)
- [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
- [4Sum](https://leetcode.com/problems/4sum/)
- [3Sum Smaller](https://leetcode.com/problems/3sum-smaller/)

---

## Template 4: Partitioning Pattern

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
void partition(vector<int>& arr, int pivot) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        if (arr[left] < pivot) {
            left++;
        } else if (arr[right] >= pivot) {
            right--;
        } else {
            swap(arr[left], arr[right]);
            left++;
            right--;
        }
    }
}
```

**Alternative: Three Regions (Dutch National Flag)**

```cpp
void partitionThreeRegions(vector<int>& arr, int pivot) {
    int low = 0, mid = 0, high = arr.size() - 1;
    
    while (mid <= high) {
        if (arr[mid] < pivot) {
            swap(arr[low++], arr[mid++]);
        } else if (arr[mid] == pivot) {
            mid++;
        } else {
            swap(arr[mid], arr[high--]);
        }
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def partition(arr, pivot):
    left = 0
    right = len(arr) - 1
    
    while left <= right:
        if arr[left] < pivot:
            left += 1
        elif arr[right] >= pivot:
            right -= 1
        else:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
```

**Alternative: Three Regions (Dutch National Flag)**

```python
def partition_three_regions(arr, pivot):
    low, mid, high = 0, 0, len(arr) - 1
    
    while mid <= high:
        if arr[mid] < pivot:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == pivot:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
```

</details>

### Related Problems

- [Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/)
- [Sort Colors](https://leetcode.com/problems/sort-colors/)
- [Move Zeroes](https://leetcode.com/problems/move-zeroes/)
- [Partition Labels](https://leetcode.com/problems/partition-labels/)

---

## Template 5: Merging Two Sorted Arrays

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> merge(vector<int>& nums1, vector<int>& nums2) {
    vector<int> result;
    int i = 0, j = 0;
    
    while (i < nums1.size() && j < nums2.size()) {
        if (nums1[i] <= nums2[j]) {
            result.push_back(nums1[i++]);
        } else {
            result.push_back(nums2[j++]);
        }
    }
    
    // Add remaining elements
    while (i < nums1.size()) {
        result.push_back(nums1[i++]);
    }
    while (j < nums2.size()) {
        result.push_back(nums2[j++]);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def merge(nums1, nums2):
    result = []
    i, j = 0, 0
    
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    
    # Add remaining elements
    while i < len(nums1):
        result.append(nums1[i])
        i += 1
    while j < len(nums2):
        result.append(nums2[j])
        j += 1
    
    return result
```

</details>

### Related Problems

- [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)
- [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
- [Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

---

## Key Takeaways

1. **Opposite Ends**: Use when array is sorted - start from both ends
2. **Same Direction**: Use for removing duplicates, partitioning - slow/fast pointers
3. **Greedy Movement**: Always move the pointer that limits the solution
4. **Skip Duplicates**: Important in problems like 3Sum to avoid duplicate results
5. **In-place Operations**: Two pointers enable O(1) space solutions

---

## Common Mistakes

1. **Not handling duplicates**: Forgetting to skip duplicates in 3Sum-like problems
2. **Wrong pointer movement**: Moving wrong pointer (should move the limiting one)
3. **Boundary conditions**: Not checking `left < right` properly
4. **Index errors**: Off-by-one errors, especially with 1-indexed problems

---

## Practice Problems by Difficulty

### Easy
- [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/)
- [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [Remove Element](https://leetcode.com/problems/remove-element/)
- [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/)
- [Reverse String](https://leetcode.com/problems/reverse-string/)
- [Reverse Vowels of a String](https://leetcode.com/problems/reverse-vowels-of-a-string/)
- [Move Zeroes](https://leetcode.com/problems/move-zeroes/)

### Medium
- [3Sum](https://leetcode.com/problems/3sum/)
- [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
- [4Sum](https://leetcode.com/problems/4sum/)
- [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Remove Duplicates from Sorted Array II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/)
- [Sort Colors](https://leetcode.com/problems/sort-colors/)
- [Partition Array According to Given Pivot](https://leetcode.com/problems/partition-array-according-to-given-pivot/)
- [Partition Labels](https://leetcode.com/problems/partition-labels/)

### Hard
- [Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)
- [4Sum II](https://leetcode.com/problems/4sum-ii/)

---

## References

* [LeetCode Two Pointers Tag](https://leetcode.com/tag/two-pointers/)
* [Algorithm Patterns: Two Pointers](https://leetcode.com/discuss/study-guide/1688903/Solved-all-two-pointers-problems-in-100-days)