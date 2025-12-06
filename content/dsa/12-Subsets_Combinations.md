+++
title = "Subsets/Combinations"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 12
description = "Complete guide to Subsets and Combinations pattern with templates in C++ and Python. Covers generating all subsets, combinations of k elements, combination sums, and iterative/recursive approaches with LeetCode problem references."
+++

---

## Introduction

Subsets and Combinations are fundamental combinatorial problems that involve generating all possible selections from a set. These patterns are essential for solving problems involving power sets, combinations, and constraint-based selections.

This guide provides templates and patterns for Subsets and Combinations with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Subsets/Combinations

- **Generate all subsets**: Power set, all possible subsets
- **Generate combinations**: All ways to choose k elements from n
- **Combination sums**: Find combinations that sum to target
- **Constraint problems**: Subsets/combinations with constraints
- **Decision trees**: Making choices to form subsets/combinations

### Time & Space Complexity

- **Subsets**: O(2^n × n) time - 2^n subsets, each takes O(n) to build
- **Combinations**: O(C(n,k) × k) time - C(n,k) combinations
- **Space Complexity**: O(n) for recursion stack + O(n) for path = O(n)

---

## Pattern Variations

### Variation 1: All Subsets

Generate all possible subsets (power set).

**Use Cases**:
- Subsets
- Subsets with duplicates
- Subset sum

### Variation 2: Combinations of K

Generate all ways to choose k elements from n.

**Use Cases**:
- Combinations
- Combination Sum
- Letter Combinations

### Variation 3: Iterative Approach

Generate subsets/combinations iteratively without recursion.

**Use Cases**:
- When recursion depth is a concern
- Space-optimized solutions

---

## Template 1: Generate All Subsets (Recursive)

**Key Points**:
- At each element: include it or exclude it
- Add current subset to result at each step
- Use start index to avoid duplicates
- **Time Complexity**: O(2^n × n) - 2^n subsets, each takes O(n) to build
- **Space Complexity**: O(n) for path + O(n) for recursion stack = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> path;
    
    backtrackSubsets(nums, 0, path, result);
    return result;
}

void backtrackSubsets(vector<int>& nums, int start, vector<int>& path,
                     vector<vector<int>>& result) {
    // Add current subset to result
    result.push_back(path);
    
    // Try each element from start to end
    for (int i = start; i < nums.size(); i++) {
        // Choose: include nums[i]
        path.push_back(nums[i]);
        
        // Explore: generate subsets with nums[i] included
        backtrackSubsets(nums, i + 1, path, result);
        
        // Unchoose: exclude nums[i] (backtrack)
        path.pop_back();
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def subsets(nums):
    result = []
    path = []
    
    backtrack_subsets(nums, 0, path, result)
    return result

def backtrack_subsets(nums, start, path, result):
    # Add current subset to result
    result.append(path[:])  # Copy of path
    
    # Try each element from start to end
    for i in range(start, len(nums)):
        # Choose: include nums[i]
        path.append(nums[i])
        
        # Explore: generate subsets with nums[i] included
        backtrack_subsets(nums, i + 1, path, result)
        
        # Unchoose: exclude nums[i] (backtrack)
        path.pop()
```

</details>

### Related Problems

- [Subsets](https://leetcode.com/problems/subsets/)
- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Subset Sum](https://www.geeksforgeeks.org/subset-sum-problem-dp-25/)

---

## Template 2: Generate All Subsets (Iterative)

**Key Points**:
- Start with empty subset
- For each element, add it to all existing subsets
- Build subsets incrementally
- **Time Complexity**: O(2^n × n) - 2^n subsets
- **Space Complexity**: O(2^n × n) - storing all subsets

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> subsetsIterative(vector<int>& nums) {
    vector<vector<int>> result;
    result.push_back({}); // Start with empty subset
    
    for (int num : nums) {
        int size = result.size();
        // Add current number to all existing subsets
        for (int i = 0; i < size; i++) {
            vector<int> newSubset = result[i];
            newSubset.push_back(num);
            result.push_back(newSubset);
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def subsets_iterative(nums):
    result = [[]]  # Start with empty subset
    
    for num in nums:
        # Add current number to all existing subsets
        result += [subset + [num] for subset in result]
    
    return result
```

</details>

### Related Problems

- [Subsets](https://leetcode.com/problems/subsets/)

---

## Template 3: Combinations of K Elements

**Key Points**:
- Generate all ways to choose k elements from n
- Use start index to avoid duplicates and ensure order
- Base case: path size equals k
- **Time Complexity**: O(C(n,k) × k) - C(n,k) combinations, each takes O(k) to build
- **Space Complexity**: O(k) for path + O(k) for recursion stack = O(k)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> combine(int n, int k) {
    vector<vector<int>> result;
    vector<int> path;
    
    backtrackCombine(n, k, 1, path, result);
    return result;
}

void backtrackCombine(int n, int k, int start, vector<int>& path,
                       vector<vector<int>>& result) {
    // Base case: path has k elements
    if (path.size() == k) {
        result.push_back(path);
        return;
    }
    
    // Try each element from start to n
    for (int i = start; i <= n; i++) {
        // Choose: include i
        path.push_back(i);
        
        // Explore: generate combinations with i included
        backtrackCombine(n, k, i + 1, path, result);
        
        // Unchoose: exclude i (backtrack)
        path.pop_back();
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def combine(n, k):
    result = []
    path = []
    
    backtrack_combine(n, k, 1, path, result)
    return result

def backtrack_combine(n, k, start, path, result):
    # Base case: path has k elements
    if len(path) == k:
        result.append(path[:])  # Copy of path
        return
    
    # Try each element from start to n
    for i in range(start, n + 1):
        # Choose: include i
        path.append(i)
        
        # Explore: generate combinations with i included
        backtrack_combine(n, k, i + 1, path, result)
        
        # Unchoose: exclude i (backtrack)
        path.pop()
```

</details>

### Related Problems

- [Combinations](https://leetcode.com/problems/combinations/)
- [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)

---

## Template 4: Combination Sum

**Key Points**:
- Can reuse same element multiple times (start from i) or not (start from i+1)
- Track remaining target sum
- Base case: target reached or exceeded
- **Time Complexity**: O(2^target) worst case - exponential
- **Space Complexity**: O(target) for path + O(target) for recursion stack = O(target)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Combination Sum (can reuse elements)
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> result;
    vector<int> path;
    
    backtrackCombinationSum(candidates, target, 0, path, result);
    return result;
}

void backtrackCombinationSum(vector<int>& candidates, int target, int start,
                             vector<int>& path, vector<vector<int>>& result) {
    // Base case: target reached
    if (target == 0) {
        result.push_back(path);
        return;
    }
    
    // Base case: target exceeded
    if (target < 0) {
        return;
    }
    
    for (int i = start; i < candidates.size(); i++) {
        path.push_back(candidates[i]);
        // Can reuse same element, so start from i (not i+1)
        backtrackCombinationSum(candidates, target - candidates[i], i, path, result);
        path.pop_back(); // Backtrack
    }
}

// Combination Sum II (cannot reuse elements, with duplicates)
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    sort(candidates.begin(), candidates.end()); // Sort to handle duplicates
    vector<vector<int>> result;
    vector<int> path;
    
    backtrackCombinationSum2(candidates, target, 0, path, result);
    return result;
}

void backtrackCombinationSum2(vector<int>& candidates, int target, int start,
                              vector<int>& path, vector<vector<int>>& result) {
    if (target == 0) {
        result.push_back(path);
        return;
    }
    
    if (target < 0) {
        return;
    }
    
    for (int i = start; i < candidates.size(); i++) {
        // Skip duplicates: if same as previous and not first in this level
        if (i > start && candidates[i] == candidates[i-1]) {
            continue;
        }
        
        path.push_back(candidates[i]);
        // Cannot reuse, so start from i+1
        backtrackCombinationSum2(candidates, target - candidates[i], i + 1, path, result);
        path.pop_back();
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Combination Sum (can reuse elements)
def combination_sum(candidates, target):
    result = []
    path = []
    
    backtrack_combination_sum(candidates, target, 0, path, result)
    return result

def backtrack_combination_sum(candidates, target, start, path, result):
    # Base case: target reached
    if target == 0:
        result.append(path[:])
        return
    
    # Base case: target exceeded
    if target < 0:
        return
    
    for i in range(start, len(candidates)):
        path.append(candidates[i])
        # Can reuse same element, so start from i (not i+1)
        backtrack_combination_sum(candidates, target - candidates[i], i, path, result)
        path.pop()  # Backtrack

# Combination Sum II (cannot reuse elements, with duplicates)
def combination_sum2(candidates, target):
    candidates.sort()  # Sort to handle duplicates
    result = []
    path = []
    
    backtrack_combination_sum2(candidates, target, 0, path, result)
    return result

def backtrack_combination_sum2(candidates, target, start, path, result):
    if target == 0:
        result.append(path[:])
        return
    
    if target < 0:
        return
    
    for i in range(start, len(candidates)):
        # Skip duplicates: if same as previous and not first in this level
        if i > start and candidates[i] == candidates[i-1]:
            continue
        
        path.append(candidates[i])
        # Cannot reuse, so start from i+1
        backtrack_combination_sum2(candidates, target - candidates[i], i + 1, path, result)
        path.pop()
```

</details>

### Related Problems

- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
- [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
- [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)

---

## Template 5: Subsets with Duplicates

**Key Points**:
- Sort array first to group duplicates
- Skip duplicates when not first in current level
- Add subset at each step
- **Time Complexity**: O(2^n × n) - 2^n subsets
- **Space Complexity**: O(n) for path + O(n) for recursion stack = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    sort(nums.begin(), nums.end()); // Sort to handle duplicates
    vector<vector<int>> result;
    vector<int> path;
    
    backtrackSubsetsWithDup(nums, 0, path, result);
    return result;
}

void backtrackSubsetsWithDup(vector<int>& nums, int start, vector<int>& path,
                            vector<vector<int>>& result) {
    result.push_back(path);
    
    for (int i = start; i < nums.size(); i++) {
        // Skip duplicates: if same as previous and not first in this level
        if (i > start && nums[i] == nums[i-1]) {
            continue;
        }
        
        path.push_back(nums[i]);
        backtrackSubsetsWithDup(nums, i + 1, path, result);
        path.pop_back();
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def subsets_with_dup(nums):
    nums.sort()  # Sort to handle duplicates
    result = []
    path = []
    
    backtrack_subsets_with_dup(nums, 0, path, result)
    return result

def backtrack_subsets_with_dup(nums, start, path, result):
    result.append(path[:])
    
    for i in range(start, len(nums)):
        # Skip duplicates: if same as previous and not first in this level
        if i > start and nums[i] == nums[i-1]:
            continue
        
        path.append(nums[i])
        backtrack_subsets_with_dup(nums, i + 1, path, result)
        path.pop()
```

</details>

### Related Problems

- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)

---

## Key Takeaways

1. **Choose-Explore-Unchoose**: Standard backtracking pattern
2. **Start Index**: Use start index to avoid duplicates and ensure order
3. **Path Copying**: Always copy path when adding to result (not reference)
4. **Base Cases**: Handle target reached, target exceeded, path size reached
5. **Duplicate Handling**: Sort first, then skip duplicates when not first in level
6. **Reuse Elements**: Start from i to allow reuse, start from i+1 to prevent reuse
7. **Iterative Approach**: Build subsets incrementally for space efficiency

---

## Common Mistakes

1. **Not backtracking**: Forgetting to remove from path after recursion
2. **Reference vs Copy**: Adding path reference instead of copy to result
3. **Wrong start index**: Using wrong start index leading to duplicates
4. **Not sorting**: Forgetting to sort when handling duplicates
5. **Duplicate skip logic**: Wrong condition for skipping duplicates
6. **Base case errors**: Not handling all termination conditions correctly

---

## Practice Problems by Difficulty

### Easy
- [Subsets](https://leetcode.com/problems/subsets/)

### Medium
- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Combinations](https://leetcode.com/problems/combinations/)
- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
- [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
- [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

### Hard
- [Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)

---

## References

* [LeetCode Backtracking Tag](https://leetcode.com/tag/backtracking/)
* [Combination (Wikipedia)](https://en.wikipedia.org/wiki/Combination)

