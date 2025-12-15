+++
title = "Coding Patterns"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
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

## 📖 Patterns

### Tier 1: Must-Know Patterns

- **[Two Pointers]({{< ref "2-Two_Pointers.md" >}})** - Efficiently solve problems with sorted arrays/strings
- **[Sliding Window]({{< ref "3-Sliding_Window.md" >}})** - Find optimal subarrays/substrings
- **[Binary Search]({{< ref "4-Binary_Search.md" >}})** - O(log n) search in sorted data
- **[Fast & Slow Pointers]({{< ref "5-Fast_Slow_Pointers.md" >}})** - Cycle detection and linked list problems
- **[Tree BFS/DFS]({{< ref "6-Tree_BFS_DFS.md" >}})** - Tree traversal techniques
- **[Dynamic Programming]({{< ref "7-Dynamic_Programming.md" >}})** - Optimize overlapping subproblems
- **[Backtracking]({{< ref "8-Backtracking.md" >}})** - Generate all solutions with constraints

### Tier 2: Very Common Patterns

- **[Merge Intervals]({{< ref "9-Merge_Intervals.md" >}})** - Combine overlapping intervals
- **[Top K Elements]({{< ref "10-Top_K_Elements.md" >}})** - Find K largest/smallest/most frequent
- **[In-Place Reversal of Linked List]({{< ref "11-In_Place_Reversal_Linked_List.md" >}})** - Reverse linked lists efficiently
- **[Subsets/Combinations]({{< ref "12-Subsets_Combinations.md" >}})** - Generate all subsets and combinations
- **[Greedy Algorithms]({{< ref "13-Greedy_Algorithms.md" >}})** - Make locally optimal choices
- **[Graph BFS/DFS]({{< ref "14-Graph_BFS_DFS.md" >}})** - Graph traversal and path finding

### Tier 3: Important Patterns

- **[Two Heaps]({{< ref "15-Two_Heaps.md" >}})** - Maintain median with min/max heaps
- **[K-way Merge]({{< ref "16-K_way_Merge.md" >}})** - Merge K sorted lists efficiently
- **[Topological Sort]({{< ref "17-Topological_Sort.md" >}})** - Order nodes in DAG
- **[Union-Find (Disjoint Set)]({{< ref "18-Union_Find_Disjoint_Set.md" >}})** - Track connected components
- **[Trie (Prefix Tree)]({{< ref "19-Trie.md" >}})** - Efficient string prefix operations
- **[Monotonic Stack]({{< ref "20-Monotonic_Stack.md" >}})** - Next greater/smaller element problems
- **[Modified Binary Search]({{< ref "21-Modified_Binary_Search.md" >}})** - Binary search with custom conditions
- **[Prefix Sum]({{< ref "22-Prefix_Sum.md" >}})** - Efficient range sum queries

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

- **Pattern Recognition**: Most problems follow one of these patterns
- **Practice**: Solve 3-5 problems per pattern to internalize
- **Implementation**: Each pattern has template code and examples
- **Complexity**: Always analyze time and space complexity
- **Edge Cases**: Consider empty inputs, single elements, duplicates

