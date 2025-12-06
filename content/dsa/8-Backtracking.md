+++
title = "Backtracking"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Complete guide to Backtracking pattern with templates in C++ and Python. Covers permutations, combinations, subsets, N-Queens, and constraint satisfaction problems with LeetCode problem references."
+++

---

## Introduction

Backtracking is a systematic method for solving problems by trying partial solutions and abandoning them ("backtracking") if they cannot lead to a valid solution. It's particularly useful for constraint satisfaction problems and generating all possible solutions.

This guide provides templates and patterns for Backtracking with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Backtracking

- **Generate all solutions**: Permutations, combinations, subsets
- **Constraint satisfaction**: N-Queens, Sudoku solver
- **Decision problems**: Making choices that must satisfy constraints
- **Search problems**: Finding all paths, all valid configurations
- **Optimization with constraints**: Finding best solution under constraints

### Time & Space Complexity

- **Time Complexity**: Often exponential O(2^n) or O(n!) depending on problem
- **Space Complexity**: O(n) for recursion stack + O(n) for current path = O(n)

---

## Pattern Variations

### Variation 1: Permutations

Generate all arrangements of elements.

**Use Cases**:
- Permutations
- Permutations with duplicates
- Next permutation

### Variation 2: Combinations

Generate all selections of k elements from n.

**Use Cases**:
- Combinations
- Combination sum
- Letter combinations

### Variation 3: Subsets

Generate all possible subsets.

**Use Cases**:
- Subsets
- Subsets with duplicates
- Subset sum

### Variation 4: Constraint Satisfaction

Solve problems with specific constraints.

**Use Cases**:
- N-Queens
- Sudoku solver
- Word search

---

## Template 1: Permutations

**Key Points**:
- Use visited array or swap technique
- Add element to path, recurse, remove from path (backtrack)
- Base case: path size equals input size
- **Time Complexity**: O(n! × n) - n! permutations, each takes O(n) to build
- **Space Complexity**: O(n) for recursion stack + O(n) for path = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> result;
    vector<int> path;
    vector<bool> visited(nums.size(), false);
    
    backtrackPermute(nums, path, visited, result);
    return result;
}

void backtrackPermute(vector<int>& nums, vector<int>& path, 
                      vector<bool>& visited, vector<vector<int>>& result) {
    // Base case: path is complete
    if (path.size() == nums.size()) {
        result.push_back(path);
        return;
    }
    
    // Try each unvisited element
    for (int i = 0; i < nums.size(); i++) {
        if (!visited[i]) {
            // Choose
            path.push_back(nums[i]);
            visited[i] = true;
            
            // Explore
            backtrackPermute(nums, path, visited, result);
            
            // Unchoose (backtrack)
            path.pop_back();
            visited[i] = false;
        }
    }
}

// Alternative: Using swap (no extra space for visited)
vector<vector<int>> permuteSwap(vector<int>& nums) {
    vector<vector<int>> result;
    backtrackPermuteSwap(nums, 0, result);
    return result;
}

void backtrackPermuteSwap(vector<int>& nums, int start, vector<vector<int>>& result) {
    if (start == nums.size()) {
        result.push_back(nums);
        return;
    }
    
    for (int i = start; i < nums.size(); i++) {
        swap(nums[start], nums[i]);
        backtrackPermuteSwap(nums, start + 1, result);
        swap(nums[start], nums[i]); // Backtrack
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def permute(nums):
    result = []
    path = []
    visited = [False] * len(nums)
    
    backtrack_permute(nums, path, visited, result)
    return result

def backtrack_permute(nums, path, visited, result):
    # Base case: path is complete
    if len(path) == len(nums):
        result.append(path[:])  # Copy of path
        return
    
    # Try each unvisited element
    for i in range(len(nums)):
        if not visited[i]:
            # Choose
            path.append(nums[i])
            visited[i] = True
            
            # Explore
            backtrack_permute(nums, path, visited, result)
            
            # Unchoose (backtrack)
            path.pop()
            visited[i] = False

# Alternative: Using swap (no extra space for visited)
def permute_swap(nums):
    result = []
    backtrack_permute_swap(nums, 0, result)
    return result

def backtrack_permute_swap(nums, start, result):
    if start == len(nums):
        result.append(nums[:])  # Copy of nums
        return
    
    for i in range(start, len(nums)):
        nums[start], nums[i] = nums[i], nums[start]
        backtrack_permute_swap(nums, start + 1, result)
        nums[start], nums[i] = nums[i], nums[start]  # Backtrack
```

</details>

### Related Problems

- [Permutations](https://leetcode.com/problems/permutations/)
- [Permutations II](https://leetcode.com/problems/permutations-ii/)
- [Next Permutation](https://leetcode.com/problems/next-permutation/)
- [Permutation Sequence](https://leetcode.com/problems/permutation-sequence/)

---

## Template 2: Combinations

**Key Points**:
- Start index to avoid duplicates and ensure order
- Add element to path, recurse with next index, remove (backtrack)
- Base case: path size equals k or all elements processed
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
        // Choose
        path.push_back(i);
        
        // Explore (start from i+1 to avoid duplicates)
        backtrackCombine(n, k, i + 1, path, result);
        
        // Unchoose (backtrack)
        path.pop_back();
    }
}

// Combination Sum (can use same element multiple times)
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
        # Choose
        path.append(i)
        
        # Explore (start from i+1 to avoid duplicates)
        backtrack_combine(n, k, i + 1, path, result)
        
        # Unchoose (backtrack)
        path.pop()

# Combination Sum (can use same element multiple times)
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
```

</details>

### Related Problems

- [Combinations](https://leetcode.com/problems/combinations/)
- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
- [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/)
- [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

---

## Template 3: Subsets

**Key Points**:
- Add current subset to result at each step
- For each element: include it or exclude it
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

// Subsets with duplicates (need to skip duplicates)
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

# Subsets with duplicates (need to skip duplicates)
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

- [Subsets](https://leetcode.com/problems/subsets/)
- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Subset Sum](https://www.geeksforgeeks.org/subset-sum-problem-dp-25/)

---

## Template 4: N-Queens / Constraint Satisfaction

**Key Points**:
- Place element, check constraints, recurse, remove (backtrack)
- Use helper functions to check validity
- Base case: all elements placed successfully
- **Time Complexity**: O(n!) for N-Queens - exponential for constraint problems
- **Space Complexity**: O(n) for board representation + O(n) for recursion stack = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> result;
    vector<string> board(n, string(n, '.'));
    
    backtrackNQueens(board, 0, result);
    return result;
}

void backtrackNQueens(vector<string>& board, int row, vector<vector<string>>& result) {
    // Base case: all queens placed
    if (row == board.size()) {
        result.push_back(board);
        return;
    }
    
    // Try placing queen in each column of current row
    for (int col = 0; col < board.size(); col++) {
        if (isValidQueen(board, row, col)) {
            // Choose: place queen
            board[row][col] = 'Q';
            
            // Explore: place next queen
            backtrackNQueens(board, row + 1, result);
            
            // Unchoose: remove queen (backtrack)
            board[row][col] = '.';
        }
    }
}

bool isValidQueen(vector<string>& board, int row, int col) {
    int n = board.size();
    
    // Check column
    for (int i = 0; i < row; i++) {
        if (board[i][col] == 'Q') return false;
    }
    
    // Check diagonal (top-left to bottom-right)
    for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j] == 'Q') return false;
    }
    
    // Check diagonal (top-right to bottom-left)
    for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (board[i][j] == 'Q') return false;
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    backtrack_n_queens(board, 0, result)
    return result

def backtrack_n_queens(board, row, result):
    # Base case: all queens placed
    if row == len(board):
        result.append([''.join(row) for row in board])
        return
    
    # Try placing queen in each column of current row
    for col in range(len(board)):
        if is_valid_queen(board, row, col):
            # Choose: place queen
            board[row][col] = 'Q'
            
            # Explore: place next queen
            backtrack_n_queens(board, row + 1, result)
            
            # Unchoose: remove queen (backtrack)
            board[row][col] = '.'

def is_valid_queen(board, row, col):
    n = len(board)
    
    # Check column
    for i in range(row):
        if board[i][col] == 'Q':
            return False
    
    # Check diagonal (top-left to bottom-right)
    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j -= 1
    
    # Check diagonal (top-right to bottom-left)
    i, j = row - 1, col + 1
    while i >= 0 and j < n:
        if board[i][j] == 'Q':
            return False
        i -= 1
        j += 1
    
    return True
```

</details>

### Related Problems

- [N-Queens](https://leetcode.com/problems/n-queens/)
- [N-Queens II](https://leetcode.com/problems/n-queens-ii/)
- [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)
- [Word Search](https://leetcode.com/problems/word-search/)
- [Word Search II](https://leetcode.com/problems/word-search-ii/)

---

## Template 5: Word Search / Path Finding

**Key Points**:
- Mark cell as visited, explore neighbors, unmark (backtrack)
- Check bounds and validity before exploring
- Base case: word found or path invalid
- **Time Complexity**: O(m×n×4^L) where L is word length - exponential backtracking
- **Space Complexity**: O(L) for recursion stack + O(m×n) for visited = O(m×n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool exist(vector<vector<char>>& board, string word) {
    int m = board.size();
    int n = board[0].size();
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (backtrackWordSearch(board, word, i, j, 0)) {
                return true;
            }
        }
    }
    
    return false;
}

bool backtrackWordSearch(vector<vector<char>>& board, string& word,
                        int row, int col, int index) {
    // Base case: word found
    if (index == word.length()) {
        return true;
    }
    
    // Check bounds and validity
    if (row < 0 || row >= board.size() || 
        col < 0 || col >= board[0].size() ||
        board[row][col] != word[index]) {
        return false;
    }
    
    // Mark as visited
    char temp = board[row][col];
    board[row][col] = '#';
    
    // Explore neighbors
    bool found = backtrackWordSearch(board, word, row + 1, col, index + 1) ||
                 backtrackWordSearch(board, word, row - 1, col, index + 1) ||
                 backtrackWordSearch(board, word, row, col + 1, index + 1) ||
                 backtrackWordSearch(board, word, row, col - 1, index + 1);
    
    // Unmark (backtrack)
    board[row][col] = temp;
    
    return found;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def exist(board, word):
    m = len(board)
    n = len(board[0])
    
    for i in range(m):
        for j in range(n):
            if backtrack_word_search(board, word, i, j, 0):
                return True
    
    return False

def backtrack_word_search(board, word, row, col, index):
    # Base case: word found
    if index == len(word):
        return True
    
    # Check bounds and validity
    if (row < 0 or row >= len(board) or 
        col < 0 or col >= len(board[0]) or
        board[row][col] != word[index]):
        return False
    
    # Mark as visited
    temp = board[row][col]
    board[row][col] = '#'
    
    # Explore neighbors
    found = (backtrack_word_search(board, word, row + 1, col, index + 1) or
             backtrack_word_search(board, word, row - 1, col, index + 1) or
             backtrack_word_search(board, word, row, col + 1, index + 1) or
             backtrack_word_search(board, word, row, col - 1, index + 1))
    
    # Unmark (backtrack)
    board[row][col] = temp
    
    return found
```

</details>

### Related Problems

- [Word Search](https://leetcode.com/problems/word-search/)
- [Word Search II](https://leetcode.com/problems/word-search-ii/)
- [Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)

---

## Key Takeaways

1. **Three Steps**: Choose, Explore, Unchoose (backtrack)
2. **Base Case**: Define when to stop recursion and add to result
3. **Pruning**: Skip invalid choices early to reduce search space
4. **State Management**: Mark/unmark visited, add/remove from path
5. **Avoid Duplicates**: Sort and skip duplicates when needed
6. **Start Index**: Use start index in combinations/subsets to avoid duplicates
7. **Path Copying**: Always copy path when adding to result (not reference)

---

## Common Mistakes

1. **Not backtracking**: Forgetting to undo changes (remove from path, unmark visited)
2. **Reference vs Copy**: Adding path reference instead of copy to result
3. **Wrong base case**: Not handling all termination conditions
4. **Index errors**: Off-by-one errors in loops
5. **Not pruning**: Not skipping invalid choices early
6. **Duplicate handling**: Not sorting or not skipping duplicates correctly

---

## Practice Problems by Difficulty

### Easy
- [Subsets](https://leetcode.com/problems/subsets/)
- [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

### Medium
- [Permutations](https://leetcode.com/problems/permutations/)
- [Permutations II](https://leetcode.com/problems/permutations-ii/)
- [Combinations](https://leetcode.com/problems/combinations/)
- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/)
- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Word Search](https://leetcode.com/problems/word-search/)
- [Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/)
- [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

### Hard
- [N-Queens](https://leetcode.com/problems/n-queens/)
- [N-Queens II](https://leetcode.com/problems/n-queens-ii/)
- [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)
- [Word Search II](https://leetcode.com/problems/word-search-ii/)
- [Remove Invalid Parentheses](https://leetcode.com/problems/remove-invalid-parentheses/)

---

## References

* [LeetCode Backtracking Tag](https://leetcode.com/tag/backtracking/)
* [Backtracking (Wikipedia)](https://en.wikipedia.org/wiki/Backtracking)

