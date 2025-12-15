+++
title = "K-way Merge"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 16
description = "Complete guide to K-way Merge pattern with templates in C++ and Python. Covers merging k sorted lists, k sorted arrays, finding kth smallest in sorted matrix, and heap-based merge techniques with LeetCode problem references."
+++

---

## Introduction

K-way Merge is a technique for merging k sorted sequences into one sorted sequence. It efficiently combines multiple sorted inputs using heaps or divide-and-conquer approaches.

This guide provides templates and patterns for K-way Merge with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use K-way Merge

- **Merge k sorted lists**: Combine multiple sorted linked lists
- **Merge k sorted arrays**: Combine multiple sorted arrays
- **Kth smallest in matrix**: Find kth element in sorted matrix
- **External sorting**: Merge sorted chunks
- **Streaming data**: Merge multiple sorted streams

### Time & Space Complexity

- **Heap approach**: O(n log k) where n is total elements, k is number of lists
- **Divide and conquer**: O(n log k) - merge pairs recursively
- **Space Complexity**: O(k) for heap, or O(1) for iterative merge

---

## Pattern Variations

### Variation 1: Merge K Sorted Lists

Use min-heap to merge k sorted linked lists.

**Use Cases**:
- Merge k Sorted Lists
- Merge sorted sequences

### Variation 2: Merge K Sorted Arrays

Extend to arrays or use 2D matrix.

**Use Cases**:
- Merge k sorted arrays
- Kth smallest in sorted matrix

### Variation 3: Divide and Conquer

Merge pairs recursively.

**Use Cases**:
- When heap overhead is concern
- Space-optimized solutions

---

## Template 1: Merge K Sorted Lists (Heap)

**Key Points**:
- Use min-heap to track smallest element from each list
- Push first element of each list into heap
- Pop smallest, add to result, push next from same list
- **Time Complexity**: O(n log k) where n is total nodes, k is number of lists
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

struct Compare {
    bool operator()(ListNode* a, ListNode* b) {
        return a->val > b->val; // Min-heap
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    priority_queue<ListNode*, vector<ListNode*>, Compare> minHeap;
    
    // Push first node of each list into heap
    for (ListNode* list : lists) {
        if (list != nullptr) {
            minHeap.push(list);
        }
    }
    
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (!minHeap.empty()) {
        ListNode* node = minHeap.top();
        minHeap.pop();
        
        curr->next = node;
        curr = curr->next;
        
        // Push next node from same list
        if (node->next != nullptr) {
            minHeap.push(node->next);
        }
    }
    
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def merge_k_lists(lists):
    min_heap = []
    
    # Push first node of each list into heap
    for i, lst in enumerate(lists):
        if lst is not None:
            heapq.heappush(min_heap, (lst.val, i, lst))
    
    dummy = ListNode(0)
    curr = dummy
    
    while min_heap:
        val, idx, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next
        
        # Push next node from same list
        if node.next is not None:
            heapq.heappush(min_heap, (node.next.val, idx, node.next))
    
    return dummy.next
```

</details>

### Related Problems

- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/)

---

## Template 2: Merge K Sorted Lists (Divide and Conquer)

**Key Points**:
- Merge pairs of lists recursively
- Divide lists into two halves
- Merge each half, then merge results
- **Time Complexity**: O(n log k) - log k levels, each level processes n nodes
- **Space Complexity**: O(1) - iterative merge, or O(log k) for recursion stack

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* mergeKListsDivideConquer(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;
    
    while (lists.size() > 1) {
        vector<ListNode*> merged;
        
        // Merge pairs
        for (int i = 0; i < lists.size(); i += 2) {
            ListNode* l1 = lists[i];
            ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
            merged.push_back(mergeTwoLists(l1, l2));
        }
        
        lists = merged;
    }
    
    return lists[0];
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode* dummy = new ListNode(0);
    ListNode* curr = dummy;
    
    while (l1 != nullptr && l2 != nullptr) {
        if (l1->val <= l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }
    
    curr->next = (l1 != nullptr) ? l1 : l2;
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def merge_k_lists_divide_conquer(lists):
    if not lists:
        return None
    
    while len(lists) > 1:
        merged = []
        
        # Merge pairs
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(merge_two_lists(l1, l2))
        
        lists = merged
    
    return lists[0]

def merge_two_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    
    while l1 is not None and l2 is not None:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 if l1 is not None else l2
    return dummy.next
```

</details>

### Related Problems

- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

---

## Template 3: Kth Smallest in Sorted Matrix

**Key Points**:
- Use min-heap to track smallest element from each row
- Pop k times to get kth smallest
- Push next element from same row
- **Time Complexity**: O(k log n) where n is matrix size
- **Space Complexity**: O(n) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int kthSmallest(vector<vector<int>>& matrix, int k) {
    int n = matrix.size();
    priority_queue<pair<int, pair<int, int>>, 
                   vector<pair<int, pair<int, int>>>,
                   greater<pair<int, pair<int, int>>>> minHeap;
    
    // Push first element of each row
    for (int i = 0; i < n; i++) {
        minHeap.push({matrix[i][0], {i, 0}});
    }
    
    int result = 0;
    for (int count = 0; count < k; count++) {
        auto [val, pos] = minHeap.top();
        minHeap.pop();
        result = val;
        
        int row = pos.first;
        int col = pos.second;
        
        // Push next element from same row
        if (col + 1 < n) {
            minHeap.push({matrix[row][col + 1], {row, col + 1}});
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def kth_smallest(matrix, k):
    n = len(matrix)
    min_heap = []
    
    # Push first element of each row
    for i in range(n):
        heapq.heappush(min_heap, (matrix[i][0], i, 0))
    
    result = 0
    for count in range(k):
        val, row, col = heapq.heappop(min_heap)
        result = val
        
        # Push next element from same row
        if col + 1 < n:
            heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
    
    return result
```

</details>

### Related Problems

- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

---

## Template 4: Merge K Sorted Arrays

**Key Points**:
- Similar to lists but with arrays
- Track current index for each array
- Use heap to merge efficiently
- **Time Complexity**: O(n log k) where n is total elements
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> mergeKSortedArrays(vector<vector<int>>& arrays) {
    priority_queue<pair<int, pair<int, int>>,
                   vector<pair<int, pair<int, int>>>,
                   greater<pair<int, pair<int, int>>>> minHeap;
    
    // Push first element of each array
    for (int i = 0; i < arrays.size(); i++) {
        if (!arrays[i].empty()) {
            minHeap.push({arrays[i][0], {i, 0}});
        }
    }
    
    vector<int> result;
    while (!minHeap.empty()) {
        auto [val, pos] = minHeap.top();
        minHeap.pop();
        result.push_back(val);
        
        int arrIdx = pos.first;
        int elemIdx = pos.second;
        
        // Push next element from same array
        if (elemIdx + 1 < arrays[arrIdx].size()) {
            minHeap.push({arrays[arrIdx][elemIdx + 1], {arrIdx, elemIdx + 1}});
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def merge_k_sorted_arrays(arrays):
    min_heap = []
    
    # Push first element of each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(min_heap, (arr[0], i, 0))
    
    result = []
    while min_heap:
        val, arr_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)
        
        # Push next element from same array
        if elem_idx + 1 < len(arrays[arr_idx]):
            heapq.heappush(min_heap, (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))
    
    return result
```

</details>

### Related Problems

- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

---

## Template 5: Find K Pairs with Smallest Sums

**Key Points**:
- Use min-heap to track pairs from two arrays
- Start with (arr1[0], arr2[0]) for each arr1 element
- Pop k times, push next pair from same arr1 element
- **Time Complexity**: O(k log k) - k operations, each O(log k)
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
    priority_queue<pair<int, pair<int, int>>,
                   vector<pair<int, pair<int, int>>>,
                   greater<pair<int, pair<int, int>>>> minHeap;
    
    // Push first pair from each nums1 element
    for (int i = 0; i < min((int)nums1.size(), k); i++) {
        minHeap.push({nums1[i] + nums2[0], {i, 0}});
    }
    
    vector<vector<int>> result;
    while (k-- > 0 && !minHeap.empty()) {
        auto [sum, pos] = minHeap.top();
        minHeap.pop();
        
        int i = pos.first;
        int j = pos.second;
        result.push_back({nums1[i], nums2[j]});
        
        // Push next pair from same nums1 element
        if (j + 1 < nums2.size()) {
            minHeap.push({nums1[i] + nums2[j + 1], {i, j + 1}});
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def k_smallest_pairs(nums1, nums2, k):
    min_heap = []
    
    # Push first pair from each nums1 element
    for i in range(min(len(nums1), k)):
        heapq.heappush(min_heap, (nums1[i] + nums2[0], i, 0))
    
    result = []
    while k > 0 and min_heap:
        sum_val, i, j = heapq.heappop(min_heap)
        result.append([nums1[i], nums2[j]])
        
        # Push next pair from same nums1 element
        if j + 1 < len(nums2):
            heapq.heappush(min_heap, (nums1[i] + nums2[j + 1], i, j + 1))
        k -= 1
    
    return result
```

</details>

### Related Problems

- [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

---

## Key Takeaways

1. **Min-Heap for Merging**: Use min-heap to track smallest element from each sequence
2. **Track Position**: Store (value, position) in heap to know which sequence it came from
3. **Push Next Element**: After popping, push next element from same sequence
4. **Divide and Conquer**: Alternative approach - merge pairs recursively
5. **Time Complexity**: O(n log k) where n is total elements, k is number of sequences
6. **Python Heap**: Use tuple (value, index, ...) to break ties in Python heap

---

## Common Mistakes

1. **Not tracking position**: Forgetting which sequence element came from
2. **Wrong heap type**: Using max-heap instead of min-heap
3. **Index out of bounds**: Not checking if next element exists
4. **Python tie-breaking**: Not including index in tuple for stable sorting
5. **Empty sequences**: Not handling empty lists/arrays

---

## Practice Problems by Difficulty

### Medium
- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)
- [Find K Pairs with Smallest Sums](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

---

## References

* [LeetCode Heap Tag](https://leetcode.com/tag/heap-priority-queue/)
* [Merge Sort (Wikipedia)](https://en.wikipedia.org/wiki/Merge_sort)

