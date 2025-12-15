+++
title = "Two Heaps"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 15
description = "Complete guide to Two Heaps pattern with templates in C++ and Python. Covers maintaining two heaps (min and max) for finding medians, sliding window medians, and balancing problems with LeetCode problem references."
+++

---

## Introduction

The Two Heaps pattern uses two priority queues (heaps) to maintain a balanced partition of data. Typically, one heap stores smaller elements (max-heap) and another stores larger elements (min-heap), allowing efficient access to median or kth element.

This guide provides templates and patterns for Two Heaps with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Two Heaps

- **Find median**: Maintain two heaps for median finding
- **Sliding window median**: Median in sliding window
- **Balancing problems**: Keep two halves balanced
- **Range problems**: Find elements in specific range
- **Data stream problems**: Process streaming data with median/quintile queries

### Time & Space Complexity

- **Time Complexity**: O(log n) per operation - heap insert/remove
- **Space Complexity**: O(n) - storing elements in two heaps

---

## Pattern Variations

### Variation 1: Find Median

Use max-heap for smaller half, min-heap for larger half.

**Use Cases**:
- Find Median from Data Stream
- Sliding Window Median

### Variation 2: Balanced Heaps

Maintain two heaps with size difference at most 1.

**Use Cases**:
- Median problems
- Balanced partition

---

## Template 1: Find Median from Data Stream

**Key Points**:
- Max-heap stores smaller half (left side)
- Min-heap stores larger half (right side)
- Keep heaps balanced: |maxHeap.size() - minHeap.size()| <= 1
- Median: top of larger heap, or average of both tops
- **Time Complexity**: O(log n) per add operation
- **Space Complexity**: O(n) - storing all elements

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
#include <queue>
#include <vector>

class MedianFinder {
private:
    priority_queue<int> maxHeap; // Smaller half (left side)
    priority_queue<int, vector<int>, greater<int>> minHeap; // Larger half (right side)
    
public:
    void addNum(int num) {
        // Add to appropriate heap
        if (maxHeap.empty() || num <= maxHeap.top()) {
            maxHeap.push(num);
        } else {
            minHeap.push(num);
        }
        
        // Balance heaps: |maxHeap.size() - minHeap.size()| <= 1
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else if (minHeap.size() > maxHeap.size() + 1) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }
    
    double findMedian() {
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.top() + minHeap.top()) / 2.0;
        } else {
            return maxHeap.size() > minHeap.size() ? maxHeap.top() : minHeap.top();
        }
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.max_heap = []  # Smaller half (left side) - use negative values
        self.min_heap = []  # Larger half (right side)
    
    def add_num(self, num):
        # Add to appropriate heap
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        
        # Balance heaps: |maxHeap.size() - minHeap.size()| <= 1
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap) + 1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))
    
    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return -self.max_heap[0] if len(self.max_heap) > len(self.min_heap) else self.min_heap[0]
```

</details>

### Related Problems

- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)

---

## Template 2: Sliding Window Median

**Key Points**:
- Use two heaps for current window
- Remove elements going out of window
- Add new elements to appropriate heap
- Rebalance heaps after each operation
- **Time Complexity**: O(n log k) where k is window size
- **Space Complexity**: O(k) - storing window elements

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<double> medianSlidingWindow(vector<int>& nums, int k) {
    priority_queue<int> maxHeap;
    priority_queue<int, vector<int>, greater<int>> minHeap;
    unordered_map<int, int> toRemove; // Track elements to remove
    
    vector<double> result;
    
    // Initialize first window
    for (int i = 0; i < k; i++) {
        addToHeaps(maxHeap, minHeap, nums[i]);
    }
    result.push_back(getMedian(maxHeap, minHeap));
    
    // Slide window
    for (int i = k; i < nums.size(); i++) {
        int removeNum = nums[i - k];
        int addNum = nums[i];
        
        // Mark element for removal
        toRemove[removeNum]++;
        
        // Remove old element
        removeFromHeaps(maxHeap, minHeap, toRemove);
        
        // Add new element
        addToHeaps(maxHeap, minHeap, addNum);
        
        // Remove invalid elements
        removeFromHeaps(maxHeap, minHeap, toRemove);
        
        result.push_back(getMedian(maxHeap, minHeap));
    }
    
    return result;
}

void addToHeaps(priority_queue<int>& maxHeap, 
                priority_queue<int, vector<int>, greater<int>>& minHeap, 
                int num) {
    if (maxHeap.empty() || num <= maxHeap.top()) {
        maxHeap.push(num);
    } else {
        minHeap.push(num);
    }
    balanceHeaps(maxHeap, minHeap);
}

void balanceHeaps(priority_queue<int>& maxHeap,
                  priority_queue<int, vector<int>, greater<int>>& minHeap) {
    while (maxHeap.size() > minHeap.size() + 1) {
        minHeap.push(maxHeap.top());
        maxHeap.pop();
    }
    while (minHeap.size() > maxHeap.size() + 1) {
        maxHeap.push(minHeap.top());
        minHeap.pop();
    }
}

void removeFromHeaps(priority_queue<int>& maxHeap,
                     priority_queue<int, vector<int>, greater<int>>& minHeap,
                     unordered_map<int, int>& toRemove) {
    while (!maxHeap.empty() && toRemove[maxHeap.top()] > 0) {
        toRemove[maxHeap.top()]--;
        maxHeap.pop();
    }
    while (!minHeap.empty() && toRemove[minHeap.top()] > 0) {
        toRemove[minHeap.top()]--;
        minHeap.pop();
    }
    balanceHeaps(maxHeap, minHeap);
}

double getMedian(priority_queue<int>& maxHeap,
                 priority_queue<int, vector<int>, greater<int>>& minHeap) {
    if (maxHeap.size() == minHeap.size()) {
        return ((double)maxHeap.top() + (double)minHeap.top()) / 2.0;
    }
    return maxHeap.size() > minHeap.size() ? maxHeap.top() : minHeap.top();
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq
from collections import defaultdict

def median_sliding_window(nums, k):
    max_heap = []  # Smaller half (use negative values)
    min_heap = []  # Larger half
    to_remove = defaultdict(int)  # Track elements to remove
    
    result = []
    
    # Initialize first window
    for i in range(k):
        add_to_heaps(max_heap, min_heap, nums[i])
    result.append(get_median(max_heap, min_heap))
    
    # Slide window
    for i in range(k, len(nums)):
        remove_num = nums[i - k]
        add_num = nums[i]
        
        # Mark element for removal
        to_remove[remove_num] += 1
        
        # Remove old element
        remove_from_heaps(max_heap, min_heap, to_remove)
        
        # Add new element
        add_to_heaps(max_heap, min_heap, add_num)
        
        # Remove invalid elements
        remove_from_heaps(max_heap, min_heap, to_remove)
        
        result.append(get_median(max_heap, min_heap))
    
    return result

def add_to_heaps(max_heap, min_heap, num):
    if not max_heap or num <= -max_heap[0]:
        heapq.heappush(max_heap, -num)
    else:
        heapq.heappush(min_heap, num)
    balance_heaps(max_heap, min_heap)

def balance_heaps(max_heap, min_heap):
    while len(max_heap) > len(min_heap) + 1:
        heapq.heappush(min_heap, -heapq.heappop(max_heap))
    while len(min_heap) > len(max_heap) + 1:
        heapq.heappush(max_heap, -heapq.heappop(min_heap))

def remove_from_heaps(max_heap, min_heap, to_remove):
    while max_heap and to_remove[-max_heap[0]] > 0:
        to_remove[-max_heap[0]] -= 1
        heapq.heappop(max_heap)
    while min_heap and to_remove[min_heap[0]] > 0:
        to_remove[min_heap[0]] -= 1
        heapq.heappop(min_heap)
    balance_heaps(max_heap, min_heap)

def get_median(max_heap, min_heap):
    if len(max_heap) == len(min_heap):
        return (-max_heap[0] + min_heap[0]) / 2.0
    return -max_heap[0] if len(max_heap) > len(min_heap) else min_heap[0]
```

</details>

### Related Problems

- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

---

## Template 3: Two Heaps for Range Queries

**Key Points**:
- Use two heaps to track elements in specific range
- One heap for lower bound, one for upper bound
- Efficiently query elements within range
- **Time Complexity**: O(log n) per operation
- **Space Complexity**: O(n) - storing elements

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class RangeTracker {
private:
    priority_queue<int> maxHeap; // Elements <= threshold
    priority_queue<int, vector<int>, greater<int>> minHeap; // Elements > threshold
    
public:
    void add(int num, int threshold) {
        if (num <= threshold) {
            maxHeap.push(num);
        } else {
            minHeap.push(num);
        }
    }
    
    int countInRange(int lower, int upper) {
        // Count elements in [lower, upper]
        // This is a simplified example - actual implementation depends on problem
        int count = 0;
        
        // Count in maxHeap (elements <= threshold, check if >= lower)
        priority_queue<int> tempMax = maxHeap;
        while (!tempMax.empty()) {
            int val = tempMax.top();
            tempMax.pop();
            if (val >= lower && val <= upper) {
                count++;
            }
        }
        
        // Count in minHeap (elements > threshold, check if <= upper)
        priority_queue<int, vector<int>, greater<int>> tempMin = minHeap;
        while (!tempMin.empty()) {
            int val = tempMin.top();
            tempMin.pop();
            if (val >= lower && val <= upper) {
                count++;
            }
        }
        
        return count;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

class RangeTracker:
    def __init__(self):
        self.max_heap = []  # Elements <= threshold (use negative values)
        self.min_heap = []  # Elements > threshold
    
    def add(self, num, threshold):
        if num <= threshold:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
    
    def count_in_range(self, lower, upper):
        # Count elements in [lower, upper]
        # This is a simplified example - actual implementation depends on problem
        count = 0
        
        # Count in maxHeap
        temp_max = self.max_heap[:]
        while temp_max:
            val = -heapq.heappop(temp_max)
            if lower <= val <= upper:
                count += 1
        
        # Count in minHeap
        temp_min = self.min_heap[:]
        while temp_min:
            val = heapq.heappop(temp_min)
            if lower <= val <= upper:
                count += 1
        
        return count
```

</details>

### Related Problems

- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)

---

## Key Takeaways

1. **Max-Heap for Smaller Half**: Store smaller elements in max-heap (left side)
2. **Min-Heap for Larger Half**: Store larger elements in min-heap (right side)
3. **Balance Heaps**: Keep |maxHeap.size() - minHeap.size()| <= 1
4. **Median Calculation**: If equal sizes, average both tops; else, top of larger heap
5. **Lazy Deletion**: For sliding window, mark elements for removal instead of immediate deletion
6. **Python Max-Heap**: Use negative values since Python only has min-heap

---

## Common Mistakes

1. **Wrong heap type**: Using min-heap for smaller half or max-heap for larger half
2. **Not balancing**: Forgetting to balance heaps after insertions
3. **Median calculation**: Wrong formula when heaps have equal sizes
4. **Python max-heap**: Forgetting to negate values for max-heap in Python
5. **Sliding window deletion**: Not properly handling element removal in sliding window

---

## Practice Problems by Difficulty

### Hard
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)
- [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/)

---

## References

* [LeetCode Heap Tag](https://leetcode.com/tag/heap-priority-queue/)

