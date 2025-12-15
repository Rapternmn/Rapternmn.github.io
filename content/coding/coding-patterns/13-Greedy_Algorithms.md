+++
title = "Greedy Algorithms"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 13
description = "Complete guide to Greedy Algorithms pattern with templates in C++ and Python. Covers activity selection, interval scheduling, coin change, minimum spanning tree concepts, and optimization problems with LeetCode problem references."
+++

---

## Introduction

Greedy Algorithms make locally optimal choices at each step with the hope of finding a global optimum. They're efficient and intuitive but require careful proof that the greedy choice leads to an optimal solution.

This guide provides templates and patterns for Greedy Algorithms with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Greedy Algorithms

- **Optimization problems**: Finding minimum/maximum of something
- **Activity selection**: Choosing maximum non-overlapping activities
- **Interval scheduling**: Scheduling with constraints
- **Coin change**: Making change with minimum coins (when greedy works)
- **MST problems**: Minimum spanning tree, shortest path concepts

### Time & Space Complexity

- **Time Complexity**: Often O(n log n) due to sorting, then O(n) processing
- **Space Complexity**: O(1) to O(n) depending on problem

---

## Pattern Variations

### Variation 1: Activity Selection

Select maximum non-overlapping activities.

**Use Cases**:
- Non-overlapping Intervals
- Meeting Rooms
- Activity Selection

### Variation 2: Interval Scheduling

Schedule intervals optimally.

**Use Cases**:
- Meeting Rooms II
- Minimum Arrows to Burst Balloons
- Remove Overlapping Intervals

### Variation 3: Greedy Choice Property

Make greedy choice based on specific criteria.

**Use Cases**:
- Jump Game
- Gas Station
- Assign Cookies

---

## Template 1: Activity Selection / Non-overlapping Intervals

**Key Points**:
- Sort by end time (greedy: keep intervals ending earliest)
- Track last end time
- Select intervals that don't overlap with last selected
- **Time Complexity**: O(n log n) - sorting + O(n) processing = O(n log n)
- **Space Complexity**: O(1) - only tracking variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int maxNonOverlappingIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    
    // Sort by end time (greedy: keep intervals ending earliest)
    sort(intervals.begin(), intervals.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[1] < b[1];
         });
    
    int count = 1; // First interval is always selected
    int lastEnd = intervals[0][1];
    
    for (int i = 1; i < intervals.size(); i++) {
        // If current interval doesn't overlap with last selected
        if (intervals[i][0] >= lastEnd) {
            count++;
            lastEnd = intervals[i][1];
        }
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def max_non_overlapping_intervals(intervals):
    if not intervals:
        return 0
    
    # Sort by end time (greedy: keep intervals ending earliest)
    intervals.sort(key=lambda x: x[1])
    
    count = 1  # First interval is always selected
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        # If current interval doesn't overlap with last selected
        if intervals[i][0] >= last_end:
            count += 1
            last_end = intervals[i][1]
    
    return count
```

</details>

### Related Problems

- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

---

## Template 2: Jump Game / Reachability

**Key Points**:
- Track maximum reachable position
- At each step, update maximum reach
- If current position > maximum reach, cannot proceed
- **Time Complexity**: O(n) - single pass through array
- **Space Complexity**: O(1) - only tracking maximum reach

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool canJump(vector<int>& nums) {
    int maxReach = 0;
    
    for (int i = 0; i < nums.size(); i++) {
        // If current position is beyond maximum reach
        if (i > maxReach) {
            return false;
        }
        
        // Update maximum reach
        maxReach = max(maxReach, i + nums[i]);
        
        // Early exit: already reached end
        if (maxReach >= nums.size() - 1) {
            return true;
        }
    }
    
    return true;
}

// Minimum jumps to reach end
int jump(vector<int>& nums) {
    int jumps = 0;
    int currentEnd = 0;
    int farthest = 0;
    
    for (int i = 0; i < nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        
        // Need to make a jump
        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;
        }
    }
    
    return jumps;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def can_jump(nums):
    max_reach = 0
    
    for i in range(len(nums)):
        # If current position is beyond maximum reach
        if i > max_reach:
            return False
        
        # Update maximum reach
        max_reach = max(max_reach, i + nums[i])
        
        # Early exit: already reached end
        if max_reach >= len(nums) - 1:
            return True
    
    return True

# Minimum jumps to reach end
def jump(nums):
    jumps = 0
    current_end = 0
    farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        
        # Need to make a jump
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

</details>

### Related Problems

- [Jump Game](https://leetcode.com/problems/jump-game/)
- [Jump Game II](https://leetcode.com/problems/jump-game-ii/)
- [Jump Game III](https://leetcode.com/problems/jump-game-iii/)

---

## Template 3: Gas Station / Circular Problems

**Key Points**:
- Track total gas and total cost
- If total gas >= total cost, solution exists
- Find starting point where gas never goes negative
- **Time Complexity**: O(n) - single pass through array
- **Space Complexity**: O(1) - only tracking variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int totalGas = 0;
    int totalCost = 0;
    int currentGas = 0;
    int start = 0;
    
    for (int i = 0; i < gas.size(); i++) {
        totalGas += gas[i];
        totalCost += cost[i];
        currentGas += gas[i] - cost[i];
        
        // If current gas goes negative, cannot start from previous start
        if (currentGas < 0) {
            start = i + 1;
            currentGas = 0;
        }
    }
    
    // If total gas < total cost, no solution
    if (totalGas < totalCost) {
        return -1;
    }
    
    return start;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def can_complete_circuit(gas, cost):
    total_gas = 0
    total_cost = 0
    current_gas = 0
    start = 0
    
    for i in range(len(gas)):
        total_gas += gas[i]
        total_cost += cost[i]
        current_gas += gas[i] - cost[i]
        
        # If current gas goes negative, cannot start from previous start
        if current_gas < 0:
            start = i + 1
            current_gas = 0
    
    # If total gas < total cost, no solution
    if total_gas < total_cost:
        return -1
    
    return start
```

</details>

### Related Problems

- [Gas Station](https://leetcode.com/problems/gas-station/)

---

## Template 4: Assign Cookies / Two Pointers Greedy

**Key Points**:
- Sort both arrays
- Use two pointers to match
- Greedy: assign smallest cookie that satisfies child
- **Time Complexity**: O(n log n + m log m) - sorting both arrays
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int findContentChildren(vector<int>& g, vector<int>& s) {
    // Sort both arrays
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    
    int i = 0; // Pointer for children
    int j = 0; // Pointer for cookies
    int count = 0;
    
    while (i < g.size() && j < s.size()) {
        // If cookie can satisfy child
        if (s[j] >= g[i]) {
            count++;
            i++; // Move to next child
        }
        j++; // Move to next cookie (always)
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def find_content_children(g, s):
    # Sort both arrays
    g.sort()
    s.sort()
    
    i = 0  # Pointer for children
    j = 0  # Pointer for cookies
    count = 0
    
    while i < len(g) and j < len(s):
        # If cookie can satisfy child
        if s[j] >= g[i]:
            count += 1
            i += 1  # Move to next child
        j += 1  # Move to next cookie (always)
    
    return count
```

</details>

### Related Problems

- [Assign Cookies](https://leetcode.com/problems/assign-cookies/)
- [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)

---

## Template 5: Partition Labels / Greedy Partitioning

**Key Points**:
- Track last occurrence of each character
- Use greedy: extend partition as far as possible
- When current index equals maximum last occurrence, partition
- **Time Complexity**: O(n) - two passes through string
- **Space Complexity**: O(1) - only 26 characters for last occurrence map

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> partitionLabels(string s) {
    // Find last occurrence of each character
    vector<int> lastOccurrence(26, -1);
    for (int i = 0; i < s.length(); i++) {
        lastOccurrence[s[i] - 'a'] = i;
    }
    
    vector<int> result;
    int start = 0;
    int end = 0;
    
    for (int i = 0; i < s.length(); i++) {
        // Extend partition to include last occurrence of current character
        end = max(end, lastOccurrence[s[i] - 'a']);
        
        // If reached end of partition
        if (i == end) {
            result.push_back(end - start + 1);
            start = i + 1;
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def partition_labels(s):
    # Find last occurrence of each character
    last_occurrence = {}
    for i, char in enumerate(s):
        last_occurrence[char] = i
    
    result = []
    start = 0
    end = 0
    
    for i in range(len(s)):
        # Extend partition to include last occurrence of current character
        end = max(end, last_occurrence[s[i]])
        
        # If reached end of partition
        if i == end:
            result.append(end - start + 1)
            start = i + 1
    
    return result
```

</details>

### Related Problems

- [Partition Labels](https://leetcode.com/problems/partition-labels/)
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

---

## Key Takeaways

1. **Greedy Choice**: Make locally optimal choice at each step
2. **Sorting**: Often need to sort data first (by end time, value, etc.)
3. **Proof Required**: Need to prove greedy choice leads to optimal solution
4. **No Backtracking**: Once choice is made, don't reconsider
5. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
6. **When Greedy Fails**: Some problems require DP (e.g., coin change with arbitrary denominations)

---

## Common Mistakes

1. **Wrong sort order**: Sorting by wrong key (start vs end time)
2. **Not proving optimality**: Assuming greedy works without proof
3. **Edge cases**: Not handling empty arrays, single element
4. **Index errors**: Off-by-one errors in loops
5. **Using greedy when DP needed**: Coin change with arbitrary denominations

---

## Practice Problems by Difficulty

### Easy
- [Assign Cookies](https://leetcode.com/problems/assign-cookies/)
- [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)
- [Lemonade Change](https://leetcode.com/problems/lemonade-change/)

### Medium
- [Jump Game](https://leetcode.com/problems/jump-game/)
- [Jump Game II](https://leetcode.com/problems/jump-game-ii/)
- [Gas Station](https://leetcode.com/problems/gas-station/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Partition Labels](https://leetcode.com/problems/partition-labels/)
- [Queue Reconstruction by Height](https://leetcode.com/problems/queue-reconstruction-by-height/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

### Hard
- [Candy](https://leetcode.com/problems/candy/)
- [Trapping Rain Water II](https://leetcode.com/problems/trapping-rain-water-ii/)

---

## References

* [LeetCode Greedy Tag](https://leetcode.com/tag/greedy/)
* [Greedy Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Greedy_algorithm)

