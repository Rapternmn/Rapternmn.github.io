+++
title = "DSA Coding Patterns"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to essential coding patterns for Data Structures and Algorithms. Covers 20+ patterns including two pointers, sliding window, binary search, dynamic programming, and more with LeetCode examples."
+++

---

## Introduction

Mastering coding patterns is crucial for solving Data Structures and Algorithms problems efficiently. This guide covers the most important patterns you'll encounter in technical interviews, particularly LeetCode-style problems.

Each pattern includes:
- **Concept**: What the pattern is and when to use it
- **When to Use**: Situations where this pattern applies
- **Examples**: Classic LeetCode problems
- **Time/Space Complexity**: Performance analysis

---

## Tier 1: Must-Know Patterns

### 1. Two Pointers

**Concept**: Use two pointers moving from different ends or at different speeds to solve problems efficiently.

**When to Use**:
- Sorted arrays or strings
- Need to find pairs or triplets
- Palindrome problems
- Removing duplicates

**Classic Problems**:
- [Two Sum II - Input Array Is Sorted](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)
- [3Sum](https://leetcode.com/problems/3sum/) / [3Sum Closest](https://leetcode.com/problems/3sum-closest/)
- [Container With Most Water](https://leetcode.com/problems/container-with-most-water/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)
- [Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

---

### 2. Sliding Window

**Concept**: Maintain a window of elements and slide it to find optimal subarrays/substrings.

**When to Use**:
- Subarray/substring problems
- Need to find min/max/longest/shortest
- Fixed or variable window size
- String problems with constraints

**Classic Problems**:
- [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [Maximum Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/)
- [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)
- [Permutation in String](https://leetcode.com/problems/permutation-in-string/)
- [Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/)

**Time Complexity**: O(n)  
**Space Complexity**: O(k) where k is window size

---

### 3. Binary Search

**Concept**: Divide search space in half repeatedly to find target in O(log n) time.

**When to Use**:
- Sorted array or search space
- Need O(log n) solution
- Finding boundaries or ranges
- Search in rotated arrays

**Classic Problems**:
- [Binary Search](https://leetcode.com/problems/binary-search/)
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [Search Insert Position](https://leetcode.com/problems/search-insert-position/)
- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)

**Time Complexity**: O(log n)  
**Space Complexity**: O(1)

---

### 4. Fast & Slow Pointers (Floyd's Cycle Detection)

**Concept**: Use two pointers moving at different speeds to detect cycles or find middle elements.

**When to Use**:
- Linked list problems
- Cycle detection
- Finding middle element
- Palindrome in linked list
- Intersection of two lists

**Classic Problems**:
- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
- [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
- [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

---

### 5. Tree BFS/DFS

**Concept**: Traverse trees using breadth-first (level-order) or depth-first (pre/in/post-order) approaches.

**When to Use**:
- Tree traversal problems
- Level-order problems
- Path finding
- Tree construction
- Tree validation

**Classic Problems**:
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [Same Tree](https://leetcode.com/problems/same-tree/)
- [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [Path Sum](https://leetcode.com/problems/path-sum/)
- [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

**Time Complexity**: O(n)  
**Space Complexity**: O(h) for DFS, O(w) for BFS (h = height, w = max width)

---

### 6. Dynamic Programming

**Concept**: Break problem into subproblems, solve each once, and store results to avoid recomputation.

**When to Use**:
- Optimization problems
- Overlapping subproblems
- Optimal substructure
- Counting problems
- Decision problems

**Classic Problems**:
- [Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
- [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [Coin Change](https://leetcode.com/problems/coin-change/)
- [Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [Edit Distance](https://leetcode.com/problems/edit-distance/)
- [House Robber](https://leetcode.com/problems/house-robber/)
- [Unique Paths](https://leetcode.com/problems/unique-paths/)
- [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

**Time Complexity**: O(n) to O(n²) or O(n³) depending on problem  
**Space Complexity**: O(n) to O(n²) depending on dimensions

---

### 7. Backtracking

**Concept**: Try choices, recursively explore, and backtrack if solution is invalid.

**When to Use**:
- Generate all solutions
- Constraint satisfaction
- Permutations/combinations
- N-Queens, Sudoku
- Path finding with constraints

**Classic Problems**:
- [N-Queens](https://leetcode.com/problems/n-queens/)
- [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)
- [Word Search](https://leetcode.com/problems/word-search/)
- [Permutations](https://leetcode.com/problems/permutations/)
- [Combinations](https://leetcode.com/problems/combinations/)
- [Subsets](https://leetcode.com/problems/subsets/)
- [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
- [Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

**Time Complexity**: O(2^n) to O(n!) depending on problem  
**Space Complexity**: O(n) for recursion stack

---

## Tier 2: Very Common Patterns

### 8. Merge Intervals

**Concept**: Combine overlapping or mergeable intervals efficiently.

**When to Use**:
- Interval problems
- Scheduling problems
- Overlapping ranges
- Meeting room problems

**Classic Problems**:
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)
- [Insert Interval](https://leetcode.com/problems/insert-interval/)
- [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
- [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Employee Free Time](https://leetcode.com/problems/employee-free-time/)

**Time Complexity**: O(n log n) due to sorting  
**Space Complexity**: O(n)

---

### 9. Top K Elements

**Concept**: Find K largest, smallest, or most frequent elements efficiently using heaps.

**When to Use**:
- Find K largest/smallest
- K most frequent
- K closest points
- Need priority queue

**Classic Problems**:
- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
- [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
- [Reorganize String](https://leetcode.com/problems/reorganize-string/)

**Time Complexity**: O(n log k)  
**Space Complexity**: O(k)

---

### 10. In-place Reversal of Linked List

**Concept**: Reverse linked list nodes without using extra space.

**When to Use**:
- Reverse linked list
- Reverse in groups
- Reverse between positions
- Palindrome check

**Classic Problems**:
- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
- [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

**Time Complexity**: O(n)  
**Space Complexity**: O(1)

---

### 11. Subsets/Combinations

**Concept**: Generate all possible subsets, combinations, or permutations.

**When to Use**:
- Generate all subsets
- Combinations
- Permutations
- Power set problems

**Classic Problems**:
- [Subsets](https://leetcode.com/problems/subsets/)
- [Subsets II](https://leetcode.com/problems/subsets-ii/)
- [Combinations](https://leetcode.com/problems/combinations/)
- [Combination Sum](https://leetcode.com/problems/combination-sum/)
- [Permutations](https://leetcode.com/problems/permutations/)
- [Permutations II](https://leetcode.com/problems/permutations-ii/)

**Time Complexity**: O(2^n) for subsets, O(n!) for permutations  
**Space Complexity**: O(n) for recursion stack

---

### 12. Greedy Algorithms

**Concept**: Make locally optimal choices at each step to find global optimum.

**When to Use**:
- Optimization problems
- Activity selection
- Scheduling
- When greedy choice leads to optimal solution

**Classic Problems**:
- [Jump Game](https://leetcode.com/problems/jump-game/)
- [Jump Game II](https://leetcode.com/problems/jump-game-ii/)
- [Gas Station](https://leetcode.com/problems/gas-station/)
- [Assign Cookies](https://leetcode.com/problems/assign-cookies/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

**Time Complexity**: O(n log n) typically due to sorting  
**Space Complexity**: O(1) to O(n)

---

### 13. Graph BFS/DFS

**Concept**: Traverse graph structures using breadth-first or depth-first search.

**When to Use**:
- Graph traversal
- Path finding
- Connected components
- Cycle detection
- Shortest path (BFS)

**Classic Problems**:
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Word Ladder](https://leetcode.com/problems/word-ladder/)
- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- [All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

**Time Complexity**: O(V + E) where V = vertices, E = edges  
**Space Complexity**: O(V) for visited set

---

## Tier 3: Important Patterns

### 14. Two Heaps

**Concept**: Use min-heap and max-heap together to maintain median or balance.

**When to Use**:
- Find median from data stream
- Sliding window median
- Need to track min and max simultaneously

**Classic Problems**:
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)
- [IPO](https://leetcode.com/problems/ipo/)

**Time Complexity**: O(log n) for add, O(1) for find median  
**Space Complexity**: O(n)

---

### 15. K-way Merge

**Concept**: Merge K sorted lists or arrays efficiently using heap.

**When to Use**:
- Merge K sorted lists
- Kth smallest in sorted matrix
- Multiple sorted streams

**Classic Problems**:
- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [Smallest Range Covering Elements from K Lists](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/)

**Time Complexity**: O(n log k) where n = total elements, k = number of lists  
**Space Complexity**: O(k)

---

### 16. Topological Sort

**Concept**: Order nodes in directed acyclic graph (DAG) such that all dependencies come before dependents.

**When to Use**:
- Course scheduling
- Build order problems
- Dependency resolution
- Directed graph with dependencies

**Classic Problems**:
- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
- [Sequence Reconstruction](https://leetcode.com/problems/sequence-reconstruction/)

**Time Complexity**: O(V + E)  
**Space Complexity**: O(V + E)

---

### 17. Union-Find (Disjoint Set)

**Concept**: Efficiently track and merge disjoint sets, find connected components.

**When to Use**:
- Connected components
- Cycle detection in undirected graph
- Network connectivity
- Friend circles

**Classic Problems**:
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Friend Circles](https://leetcode.com/problems/friend-circles/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)
- [Accounts Merge](https://leetcode.com/problems/accounts-merge/)
- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

**Time Complexity**: O(α(n)) per operation (almost constant)  
**Space Complexity**: O(n)

---

### 18. Trie (Prefix Tree)

**Concept**: Tree data structure for storing strings with common prefixes efficiently.

**When to Use**:
- Prefix matching
- Word search
- Autocomplete
- Dictionary problems
- String prefix/suffix problems

**Classic Problems**:
- [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
- [Word Search II](https://leetcode.com/problems/word-search-ii/)
- [Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/)
- [Add and Search Word - Data structure design](https://leetcode.com/problems/add-and-search-word-data-structure-design/)
- [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)

**Time Complexity**: O(m) for insert/search where m = word length  
**Space Complexity**: O(ALPHABET_SIZE * N * M) where N = number of words, M = average length

---

### 19. Monotonic Stack

**Concept**: Stack that maintains elements in monotonic (increasing or decreasing) order.

**When to Use**:
- Next greater/smaller element
- Largest rectangle in histogram
- Daily temperatures
- Stock span problems

**Classic Problems**:
- [Next Greater Element I](https://leetcode.com/problems/next-greater-element-i/)
- [Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Remove K Digits](https://leetcode.com/problems/remove-k-digits/)
- [Online Stock Span](https://leetcode.com/problems/online-stock-span/)

**Time Complexity**: O(n)  
**Space Complexity**: O(n)

---

### 20. Modified Binary Search

**Concept**: Binary search with custom conditions or in rotated/2D arrays.

**When to Use**:
- Rotated sorted arrays
- 2D matrix search
- Search with conditions
- Finding boundaries

**Classic Problems**:
- [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [Find Minimum in Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)
- [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/)
- [Find Peak Element](https://leetcode.com/problems/find-peak-element/)
- [Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

**Time Complexity**: O(log n)  
**Space Complexity**: O(1)

---

## Pattern Selection Guide

### How to Identify Which Pattern to Use

1. **Array/String Problems**:
   - Sorted? → Binary Search or Two Pointers
   - Subarray/Substring? → Sliding Window
   - Need all combinations? → Backtracking

2. **Linked List Problems**:
   - Cycle detection? → Fast & Slow Pointers
   - Reverse? → In-place Reversal
   - Merge? → Two Pointers

3. **Tree Problems**:
   - Level-order? → BFS
   - Path/Depth? → DFS
   - Construction? → Recursive DFS

4. **Graph Problems**:
   - Shortest path? → BFS
   - All paths? → DFS
   - Dependencies? → Topological Sort
   - Connected components? → Union-Find

5. **Optimization Problems**:
   - Overlapping subproblems? → Dynamic Programming
   - Greedy choice works? → Greedy
   - All solutions needed? → Backtracking

---

## Key Takeaways

1. **Pattern Recognition**: Most problems follow one of these patterns
2. **Practice**: Solve 3-5 problems per pattern to internalize
3. **Implementation**: Each pattern will have a dedicated file with template code and examples
4. **Complexity**: Always analyze time and space complexity
5. **Edge Cases**: Consider empty inputs, single elements, duplicates

---

## Interview Tips

1. **Identify Pattern First**: Spend 1-2 minutes identifying the pattern
2. **Explain Approach**: Walk through your thought process
3. **Use Pattern Template**: Reference the pattern's template code, then customize
4. **Test Cases**: Walk through examples before coding
5. **Optimize**: Discuss optimizations (time/space trade-offs)
6. **Edge Cases**: Mention edge cases you're handling

---

## Practice Strategy

### Week 1-2: Tier 1 Patterns
- Two Pointers: 5 problems
- Sliding Window: 5 problems
- Binary Search: 5 problems
- Fast & Slow Pointers: 3 problems
- Tree BFS/DFS: 5 problems
- Dynamic Programming: 5 problems
- Backtracking: 5 problems

### Week 3-4: Tier 2 Patterns
- Merge Intervals: 3 problems
- Top K Elements: 3 problems
- In-place Reversal: 3 problems
- Subsets/Combinations: 3 problems
- Greedy: 5 problems
- Graph BFS/DFS: 5 problems

### Week 5-6: Tier 3 Patterns
- Two Heaps: 2 problems
- K-way Merge: 2 problems
- Topological Sort: 3 problems
- Union-Find: 3 problems
- Trie: 3 problems
- Monotonic Stack: 3 problems
- Modified Binary Search: 3 problems

---

## References

* LeetCode problem patterns
* Algorithm design techniques
* Data structure fundamentals
* Time and space complexity analysis
* Interview preparation strategies

