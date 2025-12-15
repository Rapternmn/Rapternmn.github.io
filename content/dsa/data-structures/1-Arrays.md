+++
title = "Arrays"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 1
description = "Arrays data structure: operations, common patterns, and top interview problems with solutions. Covers dynamic arrays, two pointers, sliding window, and more."
+++

---

## Introduction

Arrays are the most fundamental data structure - a collection of elements stored in contiguous memory locations. Understanding arrays is essential for solving most coding problems.

---

## Array Fundamentals

### What is an Array?

**Array**: A collection of elements of the same type, stored in contiguous memory locations, accessible by index.

**Key Characteristics**:
- **Indexed Access**: O(1) access by index
- **Contiguous Memory**: Elements stored sequentially
- **Fixed/Dynamic Size**: Static or resizable
- **Zero-Based Indexing**: First element at index 0

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(1) |
| Search (unsorted) | O(n) |
| Search (sorted) | O(log n) with binary search |
| Insert at end | O(1) amortized |
| Insert at position | O(n) |
| Delete | O(n) |
| Update | O(1) |

---

## Common Patterns

### 1. Two Pointers

**Pattern**: Use two pointers moving from different ends or at different speeds.

**When to Use**:
- Sorted arrays
- Finding pairs/triplets
- Removing duplicates
- Palindrome checks

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> twoPointers(vector<int>& arr) {
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
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def two_pointers(arr):
    left = 0
    right = len(arr) - 1
    
    while left < right:
        # Process elements at left and right
        if condition:
            left += 1
        else:
            right -= 1
    
    return result
```

</details>

**Related Pattern**: See [Two Pointers Pattern]({{< ref "../coding-patterns/2-Two_Pointers.md" >}})

---

### 2. Sliding Window

**Pattern**: Maintain a window of elements and slide it to find optimal subarrays.

**When to Use**:
- Subarray/substring problems
- Fixed or variable window size
- Finding min/max/longest/shortest

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int slidingWindow(vector<int>& arr, int k) {
    int left = 0;
    int windowSum = 0;
    int result = 0;
    
    for (int right = 0; right < arr.size(); right++) {
        // Expand window
        windowSum += arr[right];
        
        // Shrink window if needed
        while (windowSum > target) {
            windowSum -= arr[left];
            left++;
        }
        
        // Update result
        result = max(result, right - left + 1);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def sliding_window(arr, k):
    left = 0
    window_sum = 0
    result = 0
    
    for right in range(len(arr)):
        # Expand window
        window_sum += arr[right]
        
        # Shrink window if needed
        while window_sum > target:
            window_sum -= arr[left]
            left += 1
        
        # Update result
        result = max(result, right - left + 1)
    
    return result
```

</details>

**Related Pattern**: See [Sliding Window Pattern]({{< ref "../coding-patterns/3-Sliding_Window.md" >}})

---

### 3. Binary Search

**Pattern**: Divide and conquer on sorted arrays.

**When to Use**:
- Sorted arrays
- Finding target element
- Finding boundaries
- Search space reduction

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int binarySearch(vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

</details>

**Related Pattern**: See [Binary Search Pattern]({{< ref "../coding-patterns/4-Binary_Search.md" >}})

---

### 4. Prefix Sum

**Pattern**: Precompute cumulative sums for range queries.

**When to Use**:
- Range sum queries
- Subarray sum problems
- Multiple queries on same array

**Related Pattern**: See [Prefix Sum Pattern]({{< ref "../coding-patterns/22-Prefix_Sum.md" >}})

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> prefixSum(vector<int>& arr) {
    int n = arr.size();
    vector<int> prefix(n + 1, 0);
    
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + arr[i];
    }
    
    // Query sum from i to j: prefix[j + 1] - prefix[i]
    return prefix;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def prefix_sum(arr):
    n = len(arr)
    prefix = [0] * (n + 1)
    
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]
    
    # Query sum from i to j: prefix[j + 1] - prefix[i]
    return prefix
```

</details>

---

### 5. Monotonic Stack

**Pattern**: Maintain stack with monotonic property (increasing/decreasing).

**When to Use**:
- Next greater/smaller element
- Largest rectangle in histogram
- Trapping rain water

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> monotonicStack(vector<int>& arr) {
    stack<int> st;
    vector<int> result(arr.size(), -1);
    
    for (int i = 0; i < arr.size(); i++) {
        // Maintain monotonic property
        while (!st.empty() && arr[st.top()] < arr[i]) {
            int idx = st.top();
            st.pop();
            result[idx] = arr[i];
        }
        st.push(i);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def monotonic_stack(arr):
    stack = []
    result = []
    
    for i in range(len(arr)):
        # Maintain monotonic property
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]
        stack.append(i)
    
    return result
```

</details>

**Related Pattern**: See [Monotonic Stack Pattern]({{< ref "../coding-patterns/20-Monotonic_Stack.md" >}})

---

## Top Problems

### Problem 1: Two Sum

**Problem**: Find two numbers that add up to target.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> hashMap;
    
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (hashMap.find(complement) != hashMap.end()) {
            return {hashMap[complement], i};
        }
        hashMap[nums[i]] = i;
    }
    
    return {};
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def twoSum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []
```

</details>

**Time**: O(n) | **Space**: O(n)

**Variations**:
- [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) (Use two pointers)
- [3Sum](https://leetcode.com/problems/3sum/) (Two pointers + sorting)

---

### Problem 2: Best Time to Buy and Sell Stock

**Problem**: Find maximum profit from buying and selling stock.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX;
    int maxProfit = 0;
    
    for (int price : prices) {
        minPrice = min(minPrice, price);
        maxProfit = max(maxProfit, price - minPrice);
    }
    
    return maxProfit;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def maxProfit(prices):
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    
    return max_profit
```

**Time**: O(n) | **Space**: O(1)

**Variations**:
- [Best Time to Buy and Sell Stock II](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) (Multiple transactions)
- [Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) (At most 2 transactions)

---

### Problem 3: Maximum Subarray (Kadane's Algorithm)

**Problem**: Find contiguous subarray with largest sum.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];
    
    for (int i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }
    
    return maxSum;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def maxSubArray(nums):
    max_sum = current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Maximum Subarray](https://leetcode.com/problems/maximum-subarray/)

---

### Problem 4: Container With Most Water

**Problem**: Find two lines that together form container with most water.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int maxArea(vector<int>& height) {
    int left = 0;
    int right = height.size() - 1;
    int maxArea = 0;
    
    while (left < right) {
        int width = right - left;
        int area = min(height[left], height[right]) * width;
        maxArea = max(maxArea, area);
        
        if (height[left] < height[right]) {
            left++;
        } else {
            right--;
        }
    }
    
    return maxArea;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_area = 0
    
    while left < right:
        width = right - left
        area = min(height[left], height[right]) * width
        max_area = max(max_area, area)
        
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_area
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)

**Pattern**: Two Pointers

---

### Problem 5: Trapping Rain Water

**Problem**: Calculate trapped rainwater between bars.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int trap(vector<int>& height) {
    if (height.empty()) return 0;
    
    int left = 0;
    int right = height.size() - 1;
    int leftMax = 0;
    int rightMax = 0;
    int water = 0;
    
    while (left < right) {
        if (height[left] < height[right]) {
            if (height[left] >= leftMax) {
                leftMax = height[left];
            } else {
                water += leftMax - height[left];
            }
            left++;
        } else {
            if (height[right] >= rightMax) {
                rightMax = height[right];
            } else {
                water += rightMax - height[right];
            }
            right--;
        }
    }
    
    return water;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def trap(height):
    if not height:
        return 0
    
    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    
    return water
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

**Pattern**: Two Pointers

---

### Problem 6: Product of Array Except Self

**Problem**: Return array where each element is product of all other elements.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, 1);
    
    // Left products
    for (int i = 1; i < n; i++) {
        result[i] = result[i - 1] * nums[i - 1];
    }
    
    // Right products
    int right = 1;
    for (int i = n - 1; i >= 0; i--) {
        result[i] *= right;
        right *= nums[i];
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def productExceptSelf(nums):
    n = len(nums)
    result = [1] * n
    
    # Left products
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Right products
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right *= nums[i]
    
    return result
```

**Time**: O(n) | **Space**: O(1) excluding output

**Related**: [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

---

### Problem 7: Merge Intervals

**Problem**: Merge overlapping intervals.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> merged;
    merged.push_back(intervals[0]);
    
    for (int i = 1; i < intervals.size(); i++) {
        vector<int>& last = merged.back();
        if (intervals[i][0] <= last[1]) {
            last[1] = max(last[1], intervals[i][1]);
        } else {
            merged.push_back(intervals[i]);
        }
    }
    
    return merged;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def merge(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged
```

**Time**: O(n log n) | **Space**: O(n)

**Related**: [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

**Pattern**: See [Merge Intervals Pattern]({{< ref "../coding-patterns/9-Merge_Intervals.md" >}})

---

### Problem 8: Rotate Array

**Problem**: Rotate array to the right by k steps.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
void rotate(vector<int>& nums, int k) {
    int n = nums.size();
    k = k % n;
    
    auto reverse = [&](int start, int end) {
        while (start < end) {
            swap(nums[start], nums[end]);
            start++;
            end--;
        }
    };
    
    reverse(0, n - 1);
    reverse(0, k - 1);
    reverse(k, n - 1);
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def rotate(nums, k):
    n = len(nums)
    k = k % n
    
    def reverse(start, end):
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
    
    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Rotate Array](https://leetcode.com/problems/rotate-array/)

---

### Problem 9: Find All Duplicates in Array

**Problem**: Find all duplicates in array (1 ≤ nums[i] ≤ n).

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> findDuplicates(vector<int>& nums) {
    vector<int> result;
    for (int num : nums) {
        int idx = abs(num) - 1;
        if (nums[idx] < 0) {
            result.push_back(abs(num));
        } else {
            nums[idx] = -nums[idx];
        }
    }
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def findDuplicates(nums):
    result = []
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            result.append(abs(num))
        else:
            nums[idx] = -nums[idx]
    return result
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/)

---

### Problem 10: Longest Consecutive Sequence

**Problem**: Find length of longest consecutive sequence.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int longest = 0;
    
    for (int num : numSet) {
        if (numSet.find(num - 1) == numSet.end()) {
            int currentNum = num;
            int currentStreak = 1;
            
            while (numSet.find(currentNum + 1) != numSet.end()) {
                currentNum++;
                currentStreak++;
            }
            
            longest = max(longest, currentStreak);
        }
    }
    
    return longest;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1
            
            longest = max(longest, current_streak)
    
    return longest
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

---

## Advanced Patterns

### 1. Dutch National Flag

**Pattern**: Sort array with three distinct values.

**Problem**: [Sort Colors](https://leetcode.com/problems/sort-colors/)

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
void sortColors(vector<int>& nums) {
    int left = 0;
    int current = 0;
    int right = nums.size() - 1;
    
    while (current <= right) {
        if (nums[current] == 0) {
            swap(nums[left], nums[current]);
            left++;
            current++;
        } else if (nums[current] == 2) {
            swap(nums[current], nums[right]);
            right--;
        } else {
            current++;
        }
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def sortColors(nums):
    left = current = 0
    right = len(nums) - 1
    
    while current <= right:
        if nums[current] == 0:
            nums[left], nums[current] = nums[current], nums[left]
            left += 1
            current += 1
        elif nums[current] == 2:
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
        else:
            current += 1
```

---

### 2. Next Permutation

**Pattern**: Find lexicographically next permutation.

**Problem**: [Next Permutation](https://leetcode.com/problems/next-permutation/)

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
void nextPermutation(vector<int>& nums) {
    int i = nums.size() - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }
    
    if (i >= 0) {
        int j = nums.size() - 1;
        while (nums[j] <= nums[i]) {
            j--;
        }
        swap(nums[i], nums[j]);
    }
    
    reverse(nums.begin() + i + 1, nums.end());
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def nextPermutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    
    nums[i + 1:] = reversed(nums[i + 1:])
```

---

## Key Takeaways

- Arrays are fundamental - master basic operations first
- Two pointers and sliding window are most common patterns
- Binary search works on sorted arrays
- Prefix sum optimizes range queries
- Monotonic stack solves next greater/smaller problems
- Practice pattern recognition for faster problem solving
- Time/space complexity analysis is crucial

---

## Practice Problems

**Easy**:
- [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
- [Plus One](https://leetcode.com/problems/plus-one/)
- [Move Zeroes](https://leetcode.com/problems/move-zeroes/)

**Medium**:
- [3Sum](https://leetcode.com/problems/3sum/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)
- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)

**Hard**:
- [First Missing Positive](https://leetcode.com/problems/first-missing-positive/)
- [Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)
- [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/)

