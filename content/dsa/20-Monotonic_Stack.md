+++
title = "Monotonic Stack"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 20
description = "Complete guide to Monotonic Stack pattern with templates in C++ and Python. Covers next greater element, next smaller element, largest rectangle, and stack-based problems with LeetCode problem references."
+++

---

## Introduction

Monotonic Stack is a stack that maintains elements in monotonic (increasing or decreasing) order. It's used to solve problems involving finding next/previous greater/smaller elements, and problems that can be solved by maintaining a sorted order.

This guide provides templates and patterns for Monotonic Stack with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Monotonic Stack

- **Next greater element**: Find next greater element for each element
- **Next smaller element**: Find next smaller element for each element
- **Largest rectangle**: Find largest rectangle in histogram
- **Trapping rain water**: Calculate trapped rainwater
- **Stock span**: Calculate stock span problem

### Time & Space Complexity

- **Time Complexity**: O(n) - each element pushed and popped once
- **Space Complexity**: O(n) - stack size

---

## Pattern Variations

### Variation 1: Monotonic Increasing Stack

Stack maintains increasing order (bottom to top).

**Use Cases**:
- Next greater element
- Previous greater element

### Variation 2: Monotonic Decreasing Stack

Stack maintains decreasing order (bottom to top).

**Use Cases**:
- Next smaller element
- Previous smaller element

---

## Template 1: Next Greater Element

**Key Points**:
- Use decreasing stack (monotonic decreasing)
- For each element, pop all smaller elements and set their next greater
- Push current element
- **Time Complexity**: O(n) - each element pushed and popped once
- **Space Complexity**: O(n) - stack size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> nextGreaterElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st; // Store indices
    
    for (int i = 0; i < n; i++) {
        // Pop all elements smaller than current
        while (!st.empty() && nums[st.top()] < nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}

// Next Greater Element in Circular Array
vector<int> nextGreaterElementsCircular(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;
    
    // Process array twice for circular
    for (int i = 0; i < 2 * n; i++) {
        int idx = i % n;
        while (!st.empty() && nums[st.top()] < nums[idx]) {
            result[st.top()] = nums[idx];
            st.pop();
        }
        if (i < n) {
            st.push(idx);
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop all elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            result[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    
    return result

# Next Greater Element in Circular Array
def next_greater_elements_circular(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    # Process array twice for circular
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack[-1]] = nums[idx]
            stack.pop()
        if i < n:
            stack.append(idx)
    
    return result
```

</details>

### Related Problems

- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

---

## Template 2: Next Smaller Element

**Key Points**:
- Use increasing stack (monotonic increasing)
- For each element, pop all larger elements and set their next smaller
- Push current element
- **Time Complexity**: O(n) - each element pushed and popped once
- **Space Complexity**: O(n) - stack size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> nextSmallerElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st; // Store indices
    
    for (int i = 0; i < n; i++) {
        // Pop all elements larger than current
        while (!st.empty() && nums[st.top()] > nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }
    
    return result;
}

// Previous Smaller Element
vector<int> previousSmallerElement(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;
    
    for (int i = n - 1; i >= 0; i--) {
        // Pop all elements larger than current
        while (!st.empty() && nums[st.top()] > nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
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
def next_smaller_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop all elements larger than current
        while stack and nums[stack[-1]] > nums[i]:
            result[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    
    return result

# Previous Smaller Element
def previous_smaller_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    
    for i in range(n - 1, -1, -1):
        # Pop all elements larger than current
        while stack and nums[stack[-1]] > nums[i]:
            result[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)
    
    return result
```

</details>

### Related Problems

- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

---

## Template 3: Largest Rectangle in Histogram

**Key Points**:
- Use increasing stack
- For each bar, calculate area with it as height
- Width = current index - previous smaller index - 1
- **Time Complexity**: O(n) - each bar pushed and popped once
- **Space Complexity**: O(n) - stack size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    int n = heights.size();
    
    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];
        
        // Pop all bars taller than current
        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        st.push(i);
    }
    
    return maxArea;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    n = len(heights)
    
    for i in range(n + 1):
        h = 0 if i == n else heights[i]
        
        # Pop all bars taller than current
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area
```

</details>

### Related Problems

- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

---

## Template 4: Trapping Rain Water

**Key Points**:
- Use decreasing stack
- When current bar is taller than stack top, water can be trapped
- Calculate trapped water between stack top and current
- **Time Complexity**: O(n) - each bar processed once
- **Space Complexity**: O(n) - stack size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int trap(vector<int>& height) {
    stack<int> st;
    int water = 0;
    
    for (int i = 0; i < height.size(); i++) {
        while (!st.empty() && height[st.top()] < height[i]) {
            int bottom = st.top();
            st.pop();
            
            if (st.empty()) break;
            
            int width = i - st.top() - 1;
            int trappedHeight = min(height[i], height[st.top()]) - height[bottom];
            water += width * trappedHeight;
        }
        st.push(i);
    }
    
    return water;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def trap(height):
    stack = []
    water = 0
    
    for i in range(len(height)):
        while stack and height[stack[-1]] < height[i]:
            bottom = stack.pop()
            
            if not stack:
                break
            
            width = i - stack[-1] - 1
            trapped_height = min(height[i], height[stack[-1]]) - height[bottom]
            water += width * trapped_height
        
        stack.append(i)
    
    return water
```

</details>

### Related Problems

- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)

---

## Template 5: Remove K Digits / Monotonic Stack for Strings

**Key Points**:
- Use increasing stack
- Remove digits that are larger than next digit
- Keep k digits to remove
- **Time Complexity**: O(n) - process each digit once
- **Space Complexity**: O(n) - stack size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
string removeKdigits(string num, int k) {
    stack<char> st;
    
    for (char digit : num) {
        // Remove digits larger than current while k > 0
        while (!st.empty() && k > 0 && st.top() > digit) {
            st.pop();
            k--;
        }
        st.push(digit);
    }
    
    // Remove remaining k digits from end
    while (k > 0 && !st.empty()) {
        st.pop();
        k--;
    }
    
    // Build result string
    string result = "";
    while (!st.empty()) {
        result = st.top() + result;
        st.pop();
    }
    
    // Remove leading zeros
    int start = 0;
    while (start < result.length() && result[start] == '0') {
        start++;
    }
    
    return (start == result.length()) ? "0" : result.substr(start);
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def remove_k_digits(num, k):
    stack = []
    
    for digit in num:
        # Remove digits larger than current while k > 0
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    while k > 0 and stack:
        stack.pop()
        k -= 1
    
    # Build result string
    result = "".join(stack)
    
    # Remove leading zeros
    result = result.lstrip('0')
    
    return result if result else "0"
```

</details>

### Related Problems

- [Remove K Digits](https://leetcode.com/problems/remove-k-digits/)
- [Create Maximum Number](https://leetcode.com/problems/create-maximum-number/)

---

## Key Takeaways

1. **Decreasing Stack for Next Greater**: Use decreasing stack to find next greater element
2. **Increasing Stack for Next Smaller**: Use increasing stack to find next smaller element
3. **Store Indices**: Usually store indices in stack, not values, for easier calculation
4. **Process Twice for Circular**: Process array twice for circular array problems
5. **Width Calculation**: Width = current index - previous index - 1
6. **Leading Zeros**: Handle leading zeros in string problems

---

## Common Mistakes

1. **Wrong stack type**: Using increasing stack for next greater or vice versa
2. **Index vs Value**: Storing values instead of indices
3. **Width calculation**: Wrong formula for calculating width/area
4. **Edge cases**: Not handling empty stack, single element, or all same elements
5. **Circular arrays**: Forgetting to process twice for circular problems

---

## Practice Problems by Difficulty

### Easy
- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)

### Medium
- [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [Remove K Digits](https://leetcode.com/problems/remove-k-digits/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

### Hard
- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)

---

## References

* [LeetCode Stack Tag](https://leetcode.com/tag/stack/)
* [Monotonic Stack (GeeksforGeeks)](https://www.geeksforgeeks.org/introduction-to-monotonic-stack-data-structure-and-algorithms/)

