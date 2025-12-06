+++
title = "Top K Elements"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Complete guide to Top K Elements pattern with templates in C++ and Python. Covers heap-based solutions, quickselect algorithm, frequency counting, and finding k largest/smallest/frequent elements with LeetCode problem references."
+++

---

## Introduction

The Top K Elements pattern is used to find the k largest, smallest, or most frequent elements in a dataset. It typically uses heaps (priority queues) or the quickselect algorithm to solve these problems efficiently.

This guide provides templates and patterns for Top K Elements with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Top K Elements

- **Find k largest/smallest**: Kth largest element, top k elements
- **Frequency problems**: Top k frequent elements, characters
- **Streaming data**: Find top k in data stream
- **Selection problems**: Kth element without full sort
- **Optimization**: When full sort is unnecessary

### Time & Space Complexity

- **Heap approach**: O(n log k) time, O(k) space
- **Quickselect**: O(n) average, O(n²) worst case, O(1) space
- **Sorting**: O(n log n) time, O(1) space

---

## Pattern Variations

### Variation 1: K Largest/Smallest Elements

Find k largest or smallest elements.

**Use Cases**:
- Kth Largest Element
- Top K Frequent Elements
- Find K Closest Points

### Variation 2: Frequency-Based

Find elements based on frequency.

**Use Cases**:
- Top K Frequent Elements
- Top K Frequent Words
- Sort Characters by Frequency

### Variation 3: Quickselect

Find kth element without full sort.

**Use Cases**:
- Kth Largest Element
- Find Median
- Quick selection

---

## Template 1: K Largest Elements (Min-Heap)

**Key Points**:
- Use min-heap of size k
- Add elements, remove smallest when heap size > k
- Heap top is kth largest
- **Time Complexity**: O(n log k) - n elements, heap operations are O(log k)
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> findKLargest(vector<int>& nums, int k) {
    // Min-heap to keep k largest elements
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num);
        
        // Keep only k largest elements
        if (minHeap.size() > k) {
            minHeap.pop(); // Remove smallest
        }
    }
    
    // Extract k largest elements
    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
    }
    
    reverse(result.begin(), result.end()); // Largest to smallest
    return result;
}

// Kth Largest Element
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) {
            minHeap.pop();
        }
    }
    
    return minHeap.top(); // Kth largest
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def find_k_largest(nums, k):
    # Min-heap to keep k largest elements
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        
        # Keep only k largest elements
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Remove smallest
    
    # Extract k largest elements
    result = []
    while min_heap:
        result.append(heapq.heappop(min_heap))
    
    return result[::-1]  # Largest to smallest

# Kth Largest Element
def find_kth_largest(nums, k):
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return min_heap[0]  # Kth largest
```

</details>

### Related Problems

- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Find K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

---

## Template 2: K Smallest Elements (Max-Heap)

**Key Points**:
- Use max-heap of size k
- Add elements, remove largest when heap size > k
- Heap top is kth smallest
- **Time Complexity**: O(n log k) - n elements, heap operations are O(log k)
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> findKSmallest(vector<int>& nums, int k) {
    // Max-heap to keep k smallest elements
    priority_queue<int> maxHeap;
    
    for (int num : nums) {
        maxHeap.push(num);
        
        // Keep only k smallest elements
        if (maxHeap.size() > k) {
            maxHeap.pop(); // Remove largest
        }
    }
    
    // Extract k smallest elements
    vector<int> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top());
        maxHeap.pop();
    }
    
    reverse(result.begin(), result.end()); // Smallest to largest
    return result;
}

// Kth Smallest Element
int findKthSmallest(vector<int>& nums, int k) {
    priority_queue<int> maxHeap;
    
    for (int num : nums) {
        maxHeap.push(num);
        if (maxHeap.size() > k) {
            maxHeap.pop();
        }
    }
    
    return maxHeap.top(); // Kth smallest
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def find_k_smallest(nums, k):
    # Max-heap to keep k smallest elements (use negative values)
    max_heap = []
    
    for num in nums:
        heapq.heappush(max_heap, -num)  # Negate for max-heap
        
        # Keep only k smallest elements
        if len(max_heap) > k:
            heapq.heappop(max_heap)  # Remove largest
    
    # Extract k smallest elements
    result = [-x for x in max_heap]
    result.sort()  # Smallest to largest
    return result

# Kth Smallest Element
def find_kth_smallest(nums, k):
    max_heap = []
    
    for num in nums:
        heapq.heappush(max_heap, -num)  # Negate for max-heap
        if len(max_heap) > k:
            heapq.heappop(max_heap)
    
    return -max_heap[0]  # Kth smallest
```

</details>

### Related Problems

- [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [Find K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

---

## Template 3: Top K Frequent Elements

**Key Points**:
- Count frequencies using hash map
- Use min-heap to keep k most frequent
- Compare by frequency, not value
- **Time Complexity**: O(n) for counting + O(n log k) for heap = O(n log k)
- **Space Complexity**: O(n) for frequency map + O(k) for heap = O(n)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    // Count frequencies
    unordered_map<int, int> freq;
    for (int num : nums) {
        freq[num]++;
    }
    
    // Min-heap of size k (compare by frequency)
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> minHeap;
    
    for (auto& [num, count] : freq) {
        minHeap.push({count, num});
        
        if (minHeap.size() > k) {
            minHeap.pop(); // Remove least frequent
        }
    }
    
    // Extract top k frequent
    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }
    
    reverse(result.begin(), result.end());
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    # Count frequencies
    freq = Counter(nums)
    
    # Min-heap of size k (compare by frequency)
    min_heap = []
    
    for num, count in freq.items():
        heapq.heappush(min_heap, (count, num))
        
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Remove least frequent
    
    # Extract top k frequent
    result = [num for count, num in min_heap]
    return result[::-1]
```

</details>

### Related Problems

- [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
- [Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)

---

## Template 4: Quickselect Algorithm

**Key Points**:
- Partition-based selection algorithm
- Similar to quicksort but only recurses on needed partition
- Average O(n), worst case O(n²)
- **Time Complexity**: O(n) average, O(n²) worst case
- **Space Complexity**: O(1) - in-place partitioning

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int findKthLargestQuickselect(vector<int>& nums, int k) {
    int n = nums.size();
    int targetIndex = n - k; // Kth largest is at index n-k
    
    return quickselect(nums, 0, n - 1, targetIndex);
}

int quickselect(vector<int>& nums, int left, int right, int k) {
    if (left == right) {
        return nums[left];
    }
    
    int pivotIndex = partition(nums, left, right);
    
    if (pivotIndex == k) {
        return nums[pivotIndex];
    } else if (pivotIndex < k) {
        return quickselect(nums, pivotIndex + 1, right, k);
    } else {
        return quickselect(nums, left, pivotIndex - 1, k);
    }
}

int partition(vector<int>& nums, int left, int right) {
    int pivot = nums[right];
    int i = left;
    
    for (int j = left; j < right; j++) {
        if (nums[j] <= pivot) {
            swap(nums[i], nums[j]);
            i++;
        }
    }
    
    swap(nums[i], nums[right]);
    return i;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def find_kth_largest_quickselect(nums, k):
    n = len(nums)
    target_index = n - k  # Kth largest is at index n-k
    
    return quickselect(nums, 0, n - 1, target_index)

def quickselect(nums, left, right, k):
    if left == right:
        return nums[left]
    
    pivot_index = partition(nums, left, right)
    
    if pivot_index == k:
        return nums[pivot_index]
    elif pivot_index < k:
        return quickselect(nums, pivot_index + 1, right, k)
    else:
        return quickselect(nums, left, pivot_index - 1, k)

def partition(nums, left, right):
    pivot = nums[right]
    i = left
    
    for j in range(left, right):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    
    nums[i], nums[right] = nums[right], nums[i]
    return i
```

</details>

### Related Problems

- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

---

## Template 5: Top K in Data Stream

**Key Points**:
- Maintain heap as data streams in
- Add new elements, maintain heap size k
- Handle updates and removals
- **Time Complexity**: O(log k) per operation
- **Space Complexity**: O(k) - heap size

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class KthLargest {
private:
    priority_queue<int, vector<int>, greater<int>> minHeap;
    int k;
    
public:
    KthLargest(int k, vector<int>& nums) {
        this->k = k;
        for (int num : nums) {
            add(num);
        }
    }
    
    int add(int val) {
        minHeap.push(val);
        
        if (minHeap.size() > k) {
            minHeap.pop();
        }
        
        return minHeap.top(); // Kth largest
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.min_heap = []
        for num in nums:
            self.add(num)
    
    def add(self, val):
        heapq.heappush(self.min_heap, val)
        
        if len(self.min_heap) > self.k:
            heapq.heappop(self.min_heap)
        
        return self.min_heap[0]  # Kth largest
```

</details>

### Related Problems

- [Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

---

## Key Takeaways

1. **Min-Heap for K Largest**: Keep k largest, remove smallest when size > k
2. **Max-Heap for K Smallest**: Keep k smallest, remove largest when size > k
3. **Frequency Counting**: Use hash map, then heap by frequency
4. **Quickselect**: O(n) average for finding kth element
5. **Heap Size**: Always maintain heap size k for O(log k) operations
6. **Comparison**: For custom objects, define comparator for heap

---

## Common Mistakes

1. **Wrong heap type**: Using max-heap for k largest or min-heap for k smallest
2. **Heap size**: Not maintaining heap size k, leading to O(n log n)
3. **Frequency comparison**: Comparing by value instead of frequency
4. **Quickselect pivot**: Wrong pivot selection leading to worst case
5. **Index errors**: Off-by-one errors in kth element (0-indexed vs 1-indexed)

---

## Practice Problems by Difficulty

### Easy
- [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Kth Largest Element in a Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/)

### Medium
- [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
- [Top K Frequent Words](https://leetcode.com/problems/top-k-frequent-words/)
- [Find K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)
- [Sort Characters By Frequency](https://leetcode.com/problems/sort-characters-by-frequency/)
- [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

### Hard
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

---

## References

* [LeetCode Heap Tag](https://leetcode.com/tag/heap-priority-queue/)
* [Quickselect Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Quickselect)

