+++
title = "Stacks"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 4
description = "Stacks data structure: LIFO principle, operations, and top interview problems. Covers stack applications, monotonic stacks, and more."
+++

---

## Introduction

Stacks are linear data structures that follow the Last-In-First-Out (LIFO) principle. They are essential for solving problems involving nested structures, parsing, and backtracking.

---

## Stack Fundamentals

### What is a Stack?

**Stack**: A linear data structure where elements are added and removed from the same end (top).

**Key Characteristics**:
- **LIFO**: Last In, First Out
- **Top Element**: Most recently added element
- **Operations**: Push (add), Pop (remove), Peek/Top (view)
- **Dynamic Size**: Can grow/shrink

### Operations

| Operation | Description | Time Complexity |
|-----------|-------------|----------------|
| Push | Add element to top | O(1) |
| Pop | Remove top element | O(1) |
| Peek/Top | View top element | O(1) |
| IsEmpty | Check if empty | O(1) |
| Size | Get number of elements | O(1) |

---

## Implementation

### Using List (Python)

```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Using Linked List

```python
class StackNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

class Stack:
    def __init__(self):
        self.top = None
        self.size = 0
    
    def push(self, val):
        new_node = StackNode(val)
        new_node.next = self.top
        self.top = new_node
        self.size += 1
    
    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        val = self.top.val
        self.top = self.top.next
        self.size -= 1
        return val
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.top.val
    
    def is_empty(self):
        return self.top is None
```

---

## Common Patterns

### 1. Monotonic Stack

**Pattern**: Maintain stack with monotonic property (increasing/decreasing).

**When to Use**:
- Next greater/smaller element
- Largest rectangle in histogram
- Trapping rain water
- Stock span problem

**Template**:
```python
def monotonic_stack(arr):
    stack = []
    result = []
    
    for i in range(len(arr)):
        # Maintain monotonic decreasing property
        while stack and arr[stack[-1]] < arr[i]:
            idx = stack.pop()
            result[idx] = arr[i]  # Next greater element
        stack.append(i)
    
    return result
```

**Related Pattern**: See [Monotonic Stack Pattern]({{< ref "../20-Monotonic_Stack.md" >}})

---

### 2. Parentheses Matching

**Pattern**: Use stack to match opening and closing brackets.

**When to Use**:
- Valid parentheses
- Nested structures
- Expression evaluation
- HTML/XML parsing

**Template**:
```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0
```

---

### 3. Expression Evaluation

**Pattern**: Evaluate mathematical expressions using stack.

**When to Use**:
- Infix to postfix conversion
- Postfix evaluation
- Calculator problems

**Template**:
```python
def evaluate_postfix(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

---

## Top Problems

### Problem 1: Valid Parentheses

**Problem**: Check if string has valid parentheses.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isValid(string s) {
    stack<char> st;
    unordered_map<char, char> mapping = {
        {')', '('},
        {'}', '{'},
        {']', '['}
    };
    
    for (char c : s) {
        if (mapping.find(c) != mapping.end()) {
            if (st.empty() || st.top() != mapping[c]) {
                return false;
            }
            st.pop();
        } else {
            st.push(c);
        }
    }
    
    return st.empty();
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0
```

</details>

**Time**: O(n) | **Space**: O(n)

**Related**: [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

---

### Problem 2: Daily Temperatures

**Problem**: Find days until warmer temperature for each day.

**Solution** (Monotonic Stack):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> dailyTemperatures(vector<int>& temperatures) {
    stack<int> st;
    vector<int> result(temperatures.size(), 0);
    
    for (int i = 0; i < temperatures.size(); i++) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            int idx = st.top();
            st.pop();
            result[idx] = i - idx;
        }
        st.push(i);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def dailyTemperatures(temperatures):
    stack = []
    result = [0] * len(temperatures)
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    
    return result
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)

**Pattern**: Monotonic Stack

---

### Problem 3: Largest Rectangle in Histogram

**Problem**: Find largest rectangle area in histogram.

**Solution** (Monotonic Stack):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int largestRectangleArea(vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    
    for (int i = 0; i < heights.size(); i++) {
        while (!st.empty() && heights[st.top()] > heights[i]) {
            int h = heights[st.top()];
            st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, h * width);
        }
        st.push(i);
    }
    
    while (!st.empty()) {
        int h = heights[st.top()];
        st.pop();
        int width = st.empty() ? heights.size() : heights.size() - st.top() - 1;
        maxArea = max(maxArea, h * width);
    }
    
    return maxArea;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * width)
        stack.append(i)
    
    while stack:
        h = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * width)
    
    return max_area
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)

**Pattern**: Monotonic Stack

---

### Problem 4: Evaluate Reverse Polish Notation

**Problem**: Evaluate postfix expression.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int evalRPN(vector<string>& tokens) {
    stack<int> st;
    unordered_set<string> operators = {"+", "-", "*", "/"};
    
    for (string token : tokens) {
        if (operators.find(token) != operators.end()) {
            int b = st.top();
            st.pop();
            int a = st.top();
            st.pop();
            
            if (token == "+") {
                st.push(a + b);
            } else if (token == "-") {
                st.push(a - b);
            } else if (token == "*") {
                st.push(a * b);
            } else {
                st.push(a / b);
            }
        } else {
            st.push(stoi(token));
        }
    }
    
    return st.top();
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def evalRPN(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

</details>

**Time**: O(n) | **Space**: O(n)

**Related**: [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)

---

### Problem 5: Min Stack

**Problem**: Design stack with O(1) getMin operation.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class MinStack {
private:
    stack<int> st;
    stack<int> minSt;
    
public:
    void push(int val) {
        st.push(val);
        if (minSt.empty() || val <= minSt.top()) {
            minSt.push(val);
        }
    }
    
    void pop() {
        if (st.top() == minSt.top()) {
            minSt.pop();
        }
        st.pop();
    }
    
    int top() {
        return st.top();
    }
    
    int getMin() {
        return minSt.top();
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.min_stack[-1]
```

</details>

**Time**: O(1) for all operations | **Space**: O(n)

**Related**: [Min Stack](https://leetcode.com/problems/min-stack/)

---

### Problem 6: Next Greater Element

**Problem**: Find next greater element for each element.

**Solution** (Monotonic Stack):
```python
def nextGreaterElement(nums1, nums2):
    stack = []
    next_greater = {}
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    return [next_greater.get(num, -1) for num in nums1]
```

</details>

**Time**: O(n + m) | **Space**: O(n)

**Related**: [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)

**Pattern**: Monotonic Stack

---

### Problem 7: Decode String

**Problem**: Decode string with nested brackets and numbers.

**Solution**:
```python
def decodeString(s):
    stack = []
    current_string = ""
    current_num = 0
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            current_string += char
    
    return current_string
```

</details>

**Time**: O(n) | **Space**: O(n)

**Related**: [Decode String](https://leetcode.com/problems/decode-string/)

---

### Problem 8: Basic Calculator

**Problem**: Evaluate basic arithmetic expression with +, -, (, ).

**Solution**:
```python
def calculate(s):
    stack = []
    num = 0
    sign = 1
    result = 0
    
    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result
    
    return result + sign * num
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Basic Calculator](https://leetcode.com/problems/basic-calculator/)

---

### Problem 9: Remove K Digits

**Problem**: Remove k digits to form smallest number.

**Solution** (Monotonic Stack):
```python
def removeKdigits(num, k):
    stack = []
    
    for digit in num:
        while k > 0 and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    while k > 0:
        stack.pop()
        k -= 1
    
    # Remove leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Remove K Digits](https://leetcode.com/problems/remove-k-digits/)

**Pattern**: Monotonic Stack

---

### Problem 10: Trapping Rain Water

**Problem**: Calculate trapped rainwater between bars.

**Solution** (Stack):
```python
def trap(height):
    stack = []
    water = 0
    
    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = stack.pop()
            if not stack:
                break
            width = i - stack[-1] - 1
            trapped_height = min(height[stack[-1]], h) - height[bottom]
            water += width * trapped_height
        stack.append(i)
    
    return water
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)

---

## Advanced Patterns

### 1. Stock Span Problem

**Pattern**: Find span (days until price was higher) for each day.

```python
def stock_span(prices):
    stack = []
    span = []
    
    for i, price in enumerate(prices):
        while stack and prices[stack[-1]] <= price:
            stack.pop()
        
        span.append(i - stack[-1] if stack else i + 1)
        stack.append(i)
    
    return span
```

---

### 2. Maximum Area in Binary Matrix

**Pattern**: Find maximum rectangle of 1s in binary matrix.

```python
def maximalRectangle(matrix):
    if not matrix:
        return 0
    
    def largest_rectangle(heights):
        stack = []
        max_area = 0
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)
        while stack:
            height = heights[stack.pop()]
            width = len(heights) if not stack else len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        return max_area
    
    heights = [0] * len(matrix[0])
    max_area = 0
    
    for row in matrix:
        for j, cell in enumerate(row):
            heights[j] = heights[j] + 1 if cell == '1' else 0
        max_area = max(max_area, largest_rectangle(heights))
    
    return max_area
```

---

## Key Takeaways

- Stacks follow LIFO principle - last element added is first removed
- Monotonic stacks maintain sorted property - very useful pattern
- Perfect for nested structures: parentheses, brackets, expressions
- Expression evaluation: infix to postfix, postfix evaluation
- Backtracking: undo operations, path finding
- Time complexity: O(1) for basic operations
- Space complexity: O(n) for storing elements
- Practice monotonic stack problems - very common in interviews

---

## Practice Problems

**Easy**:
- [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- [Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/)
- [Baseball Game](https://leetcode.com/problems/baseball-game/)

**Medium**:
- [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/)
- [Asteroid Collision](https://leetcode.com/problems/asteroid-collision/)
- [Online Stock Span](https://leetcode.com/problems/online-stock-span/)

**Hard**:
- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [Maximal Rectangle](https://leetcode.com/problems/maximal-rectangle/)
- [Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)

