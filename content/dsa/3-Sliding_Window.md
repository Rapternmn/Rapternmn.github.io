+++
title = "Sliding Window"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Complete guide to Sliding Window pattern with templates in C++ and Python. Covers fixed-size and variable-size windows, substring problems, and array optimization techniques with LeetCode problem references."
+++

---

## Introduction

The Sliding Window technique is an efficient approach for solving problems involving subarrays or substrings. It maintains a window of elements and slides it across the data structure, avoiding redundant calculations and reducing time complexity from O(n²) or O(n³) to O(n).

This guide provides templates and patterns for the Sliding Window technique with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Sliding Window

- **Subarray/substring problems**: Finding subarrays or substrings that meet certain conditions
- **Fixed-size windows**: Problems requiring a window of specific size
- **Variable-size windows**: Problems where window size can vary (longest/shortest subarray)
- **Optimization problems**: Finding maximum/minimum sum, product, or count in a window
- **String problems**: Finding substrings with specific properties (anagrams, unique characters)
- **Array problems**: Finding contiguous subarrays with specific sum/product

### Time & Space Complexity

- **Time Complexity**: O(n) - single pass through array/string
- **Space Complexity**: O(k) where k is the window size, or O(1) for fixed-size windows

---

## Pattern Variations

### Variation 1: Fixed-Size Window

Window size is predetermined and remains constant throughout.

**Use Cases**:
- Maximum/minimum sum of subarray of size k
- Average of subarrays of size k
- Count occurrences of anagrams
- Permutation in string

### Variation 2: Variable-Size Window (Expand/Shrink)

Window size can grow or shrink based on conditions.

**Use Cases**:
- Longest substring with k distinct characters
- Minimum window substring
- Longest substring without repeating characters
- Subarray with given sum

### Variation 3: Two Pointers (Fast & Slow)

Similar to two pointers but specifically for maintaining a window.

**Use Cases**:
- Longest palindromic substring
- Trapping rain water
- Container with most water

---

## Template 1: Fixed-Size Sliding Window

**Key Points**:
- Window size is fixed (k)
- Slide window one position at a time
- Update result based on window contents
- Remove leftmost element, add rightmost element

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int fixedSizeSlidingWindow(vector<int>& arr, int k) {
    int n = arr.size();
    if (n < k) return 0; // or handle edge case
    
    // Calculate initial window sum
    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }
    
    int maxSum = windowSum; // or minSum, depending on problem
    
    // Slide the window
    for (int i = k; i < n; i++) {
        // Remove leftmost element, add rightmost element
        windowSum = windowSum - arr[i - k] + arr[i];
        maxSum = max(maxSum, windowSum); // or min, depending on problem
    }
    
    return maxSum;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def fixed_size_sliding_window(arr, k):
    n = len(arr)
    if n < k:
        return 0  # or handle edge case
    
    # Calculate initial window sum
    window_sum = sum(arr[:k])
    max_sum = window_sum  # or min_sum, depending on problem
    
    # Slide the window
    for i in range(k, n):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)  # or min, depending on problem
    
    return max_sum
```

</details>

### Related Problems

- [Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [Maximum Sum Subarray of Size K](https://leetcode.com/problems/maximum-sum-subarray-of-size-k/)
- [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [Permutation in String](https://leetcode.com/problems/permutation-in-string/)
- [Maximum of All Subarrays of Size K](https://leetcode.com/problems/sliding-window-maximum/)

---

## Template 2: Variable-Size Window (Expand/Shrink)

**Key Points**:
- Start with window size 0 or 1
- Expand window by moving right pointer
- Shrink window by moving left pointer when condition is met
- Track result during expansion/shrinking

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int variableSizeSlidingWindow(vector<int>& arr, int target) {
    int left = 0;
    int windowSum = 0;
    int minLength = INT_MAX; // or maxLength, depending on problem
    
    for (int right = 0; right < arr.size(); right++) {
        // Expand window
        windowSum += arr[right];
        
        // Shrink window while condition is met
        while (windowSum >= target) { // or other condition
            minLength = min(minLength, right - left + 1);
            windowSum -= arr[left];
            left++;
        }
    }
    
    return minLength == INT_MAX ? 0 : minLength;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def variable_size_sliding_window(arr, target):
    left = 0
    window_sum = 0
    min_length = float('inf')  # or max_length, depending on problem
    
    for right in range(len(arr)):
        # Expand window
        window_sum += arr[right]
        
        # Shrink window while condition is met
        while window_sum >= target:  # or other condition
            min_length = min(min_length, right - left + 1)
            window_sum -= arr[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0
```

</details>

### Related Problems

- [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)

---

## Template 3: Sliding Window with Hash Map (Character/String Problems)

**Key Points**:
- Use hash map to track character frequencies
- Expand window and update map
- Shrink window when condition is violated
- Track result during valid windows

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int longestSubstringWithKDistinct(string s, int k) {
    unordered_map<char, int> charCount;
    int left = 0;
    int maxLength = 0;
    
    for (int right = 0; right < s.length(); right++) {
        // Expand window
        charCount[s[right]]++;
        
        // Shrink window if more than k distinct characters
        while (charCount.size() > k) {
            charCount[s[left]]--;
            if (charCount[s[left]] == 0) {
                charCount.erase(s[left]);
            }
            left++;
        }
        
        // Update result
        maxLength = max(maxLength, right - left + 1);
    }
    
    return maxLength;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def longest_substring_with_k_distinct(s, k):
    char_count = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Expand window
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window if more than k distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        # Update result
        max_length = max(max_length, right - left + 1)
    
    return max_length
```

</details>

### Related Problems

- [Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

---

## Template 4: Sliding Window Maximum/Minimum

**Key Points**:
- Use deque to maintain window maximum/minimum efficiently
- Remove elements outside window from front
- Remove smaller elements from back (for max) or larger elements (for min)
- Front of deque always has the answer

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> slidingWindowMaximum(vector<int>& nums, int k) {
    deque<int> dq; // stores indices
    vector<int> result;
    
    for (int i = 0; i < nums.size(); i++) {
        // Remove indices outside current window
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        
        // Remove indices with smaller values (for maximum)
        // For minimum, change < to >
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }
        
        dq.push_back(i);
        
        // Add to result when window size is reached
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque

def sliding_window_maximum(nums, k):
    dq = deque()  # stores indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values (for maximum)
        # For minimum, change < to >
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add to result when window size is reached
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

</details>

### Related Problems

- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)
- [Maximum of All Subarrays of Size K](https://www.geeksforgeeks.org/sliding-window-maximum-maximum-of-all-subarrays-of-size-k/)

---

## Template 5: Count Subarrays with Condition

**Key Points**:
- Use sliding window to count valid subarrays
- At each position, count subarrays ending at that position
- Expand and shrink window based on condition
- Accumulate count during valid windows

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int countSubarraysWithCondition(vector<int>& arr, int k) {
    int left = 0;
    int count = 0;
    int windowSum = 0;
    
    for (int right = 0; right < arr.size(); right++) {
        windowSum += arr[right];
        
        // Shrink window while condition is violated
        while (windowSum > k) { // or other condition
            windowSum -= arr[left];
            left++;
        }
        
        // Count subarrays ending at 'right'
        // All subarrays from left to right are valid
        count += (right - left + 1);
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def count_subarrays_with_condition(arr, k):
    left = 0
    count = 0
    window_sum = 0
    
    for right in range(len(arr)):
        window_sum += arr[right]
        
        # Shrink window while condition is violated
        while window_sum > k:  # or other condition
            window_sum -= arr[left]
            left += 1
        
        # Count subarrays ending at 'right'
        # All subarrays from left to right are valid
        count += (right - left + 1)
    
    return count
```

</details>

### Related Problems

- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Number of Substrings Containing All Three Characters](https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/)
- [Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/)
- [Binary Subarray With Sum](https://leetcode.com/problems/binary-subarray-with-sum/)

---

## Key Takeaways

1. **Fixed-Size Window**: Calculate initial window, then slide by removing leftmost and adding rightmost element
2. **Variable-Size Window**: Expand with right pointer, shrink with left pointer based on conditions
3. **Hash Map for Characters**: Use map to track frequencies when dealing with string/character problems
4. **Deque for Max/Min**: Use deque to efficiently maintain window maximum/minimum
5. **Count Subarrays**: At each position, count valid subarrays ending at that position
6. **Time Optimization**: Sliding window reduces O(n²) or O(n³) to O(n) by avoiding redundant calculations

---

## Common Mistakes

1. **Not handling edge cases**: Empty arrays, window size larger than array size
2. **Off-by-one errors**: Window boundaries, array indices
3. **Not updating result correctly**: Forgetting to update result during valid windows
4. **Incorrect shrinking condition**: Shrinking too much or too little
5. **Hash map cleanup**: Not removing characters with count 0 from map
6. **Deque maintenance**: Not removing elements outside window or not maintaining order correctly

---

## Practice Problems by Difficulty

### Easy
- [Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- [Contains Duplicate II](https://leetcode.com/problems/contains-duplicate-ii/)

### Medium
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/)
- [Longest Substring with At Most K Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)
- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/)
- [Permutation in String](https://leetcode.com/problems/permutation-in-string/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- [Fruit Into Baskets](https://leetcode.com/problems/fruit-into-baskets/)

### Hard
- [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)
- [Substring with Concatenation of All Words](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

---

## References

* [LeetCode Sliding Window Tag](https://leetcode.com/tag/sliding-window/)
* [Algorithm Patterns: Sliding Window](https://leetcode.com/discuss/study-guide/1773891/Solved-all-sliding-window-problems-in-100-days)

