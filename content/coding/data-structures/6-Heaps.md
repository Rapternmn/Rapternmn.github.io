+++
title = "Heaps"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Heaps data structure: min/max heaps, priority queues, and top interview problems. Covers heap operations, top K elements, and more."
+++

---

## Introduction

Heaps are complete binary trees that satisfy the heap property. They are used to implement priority queues and solve problems involving finding top K elements, merging sorted lists, and scheduling.

---

## Heap Fundamentals

### What is a Heap?

**Heap**: A complete binary tree that satisfies the heap property.

**Key Characteristics**:
- **Complete Binary Tree**: All levels filled except possibly last
- **Heap Property**: 
  - Min Heap: Parent ≤ Children
  - Max Heap: Parent ≥ Children
- **Array Representation**: Stored as array, parent at i, children at 2i+1 and 2i+2
- **Priority Queue**: Natural implementation

### Types of Heaps

1. **Min Heap**: Root is minimum element
2. **Max Heap**: Root is maximum element

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Insert | O(log n) |
| Extract Min/Max | O(log n) |
| Peek | O(1) |
| Build Heap | O(n) |
| Heapify | O(log n) |

---

## Implementation

### Using heapq (Python - Min Heap)

```python
import heapq

# Min heap operations
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
min_val = heapq.heappop(heap)  # Returns 1
peek = heap[0]  # View minimum without removing

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -1)
max_val = -heapq.heappop(max_heap)  # Returns 3
```

### Custom Heap Implementation

```python
class MinHeap:
    def __init__(self):
        self.heap = []
    
    def parent(self, i):
        return (i - 1) // 2
    
    def left_child(self, i):
        return 2 * i + 1
    
    def right_child(self, i):
        return 2 * i + 2
    
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
    
    def insert(self, val):
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)
    
    def _heapify_up(self, i):
        while i > 0 and self.heap[i] < self.heap[self.parent(i)]:
            self.swap(i, self.parent(i))
            i = self.parent(i)
    
    def extract_min(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        min_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return min_val
    
    def _heapify_down(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != i:
            self.swap(i, smallest)
            self._heapify_down(smallest)
    
    def peek(self):
        return self.heap[0] if self.heap else None
```

---

## Common Patterns

### 1. Top K Elements

**Pattern**: Find K largest/smallest elements.

**When to Use**:
- Top K frequent elements
- K closest points
- K largest numbers
- Merge K sorted lists

**Template**:
```python
import heapq

def top_k_elements(arr, k):
    # For K largest: use min heap of size k
    heap = []
    for num in arr:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap

# Or use nlargest/nsmallest
top_k = heapq.nlargest(k, arr)
```

**Related Pattern**: See [Top K Elements Pattern]({{< ref "../coding-patterns/10-Top_K_Elements.md" >}})

---

### 2. Two Heaps

**Pattern**: Use min and max heaps together.

**When to Use**:
- Find median
- Sliding window median
- Balance two sets

**Template**:
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap
    
    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Related Pattern**: See [Two Heaps Pattern]({{< ref "../coding-patterns/15-Two_Heaps.md" >}})

---

### 3. K-way Merge

**Pattern**: Merge K sorted lists/arrays.

**When to Use**:
- Merge K sorted lists
- K sorted arrays
- External sorting

**Template**:
```python
import heapq

def merge_k_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(lists[list_idx]):
            heapq.heappush(heap, (lists[list_idx][elem_idx + 1], list_idx, elem_idx + 1))
    
    return result
```

**Related Pattern**: See [K-way Merge Pattern]({{< ref "../coding-patterns/16-K_way_Merge.md" >}})

---

## Top Problems

### Problem 1: Kth Largest Element in Array

**Problem**: Find Kth largest element.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> heap;
    
    for (int num : nums) {
        heap.push(num);
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    return heap.top();
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def findKthLargest(nums, k):
    import heapq
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    
    return heap[0]
```

</details>

**Time**: O(n log k) | **Space**: O(k)

**Related**: [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/)

---

### Problem 2: Top K Frequent Elements

**Problem**: Find K most frequent elements.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> count;
    for (int num : nums) {
        count[num]++;
    }
    
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> heap;
    
    for (auto& [num, freq] : count) {
        heap.push({freq, num});
        if (heap.size() > k) {
            heap.pop();
        }
    }
    
    vector<int> result;
    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def topKFrequent(nums, k):
    from collections import Counter
    import heapq
    
    count = Counter(nums)
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]
```

</details>

**Time**: O(n log k) | **Space**: O(n)

**Related**: [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)

**Pattern**: Top K Elements

---

### Problem 3: Merge K Sorted Lists

**Problem**: Merge K sorted linked lists.

**Solution**:
```python
def mergeKLists(lists):
    import heapq
    
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    
    dummy = ListNode(0)
    current = dummy
    
    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))
    
    return dummy.next
```

**Time**: O(n log k) where n is total nodes | **Space**: O(k)

**Related**: [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)

**Pattern**: K-way Merge

---

### Problem 4: Find Median from Data Stream

**Problem**: Find median of stream of numbers.

**Solution** (Two Heaps):
```python
class MedianFinder:
    def __init__(self):
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap
    
    def addNum(self, num):
        import heapq
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

**Time**: O(log n) for add, O(1) for find | **Space**: O(n)

**Related**: [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

**Pattern**: Two Heaps

---

### Problem 5: K Closest Points to Origin

**Problem**: Find K closest points to origin.

**Solution**:
```python
def kClosest(points, k):
    import heapq
    
    def distance(point):
        return point[0]**2 + point[1]**2
    
    heap = []
    for point in points:
        dist = distance(point)
        heapq.heappush(heap, (-dist, point))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [point for dist, point in heap]
```

**Time**: O(n log k) | **Space**: O(k)

**Related**: [K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/)

---

### Problem 6: Reorganize String

**Problem**: Reorganize string so no two same characters adjacent.

**Solution**:
```python
def reorganizeString(s):
    from collections import Counter
    import heapq
    
    count = Counter(s)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    
    result = []
    prev = None
    
    while heap or prev:
        if not heap and prev:
            return ""
        
        freq, char = heapq.heappop(heap)
        result.append(char)
        
        if prev:
            heapq.heappush(heap, prev)
            prev = None
        
        if freq + 1 < 0:
            prev = (freq + 1, char)
    
    return ''.join(result)
```

**Time**: O(n log k) where k is unique chars | **Space**: O(k)

**Related**: [Reorganize String](https://leetcode.com/problems/reorganize-string/)

---

### Problem 7: Meeting Rooms II

**Problem**: Find minimum meeting rooms needed.

**Solution**:
```python
def minMeetingRooms(intervals):
    import heapq
    
    intervals.sort(key=lambda x: x[0])
    heap = []
    
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        heapq.heappush(heap, end)
    
    return len(heap)
```

**Time**: O(n log n) | **Space**: O(n)

**Related**: [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)

---

### Problem 8: Task Scheduler

**Problem**: Find minimum time to complete tasks with cooldown.

**Solution**:
```python
def leastInterval(tasks, n):
    from collections import Counter
    import heapq
    
    count = Counter(tasks)
    heap = [-freq for freq in count.values()]
    heapq.heapify(heap)
    
    time = 0
    queue = []  # (freq, available_time)
    
    while heap or queue:
        time += 1
        
        if heap:
            freq = heapq.heappop(heap)
            if freq + 1 < 0:
                queue.append((freq + 1, time + n))
        
        if queue and queue[0][1] == time:
            heapq.heappush(heap, queue.pop(0)[0])
    
    return time
```

**Time**: O(n * m) where m is cooldown | **Space**: O(1)

**Related**: [Task Scheduler](https://leetcode.com/problems/task-scheduler/)

---

### Problem 9: Design Twitter

**Problem**: Design Twitter feed system.

**Solution**:
```python
class Twitter:
    def __init__(self):
        self.tweets = {}  # user_id -> [(time, tweet_id)]
        self.follows = {}  # user_id -> set of followees
        self.time = 0
    
    def postTweet(self, userId, tweetId):
        if userId not in self.tweets:
            self.tweets[userId] = []
        self.tweets[userId].append((self.time, tweetId))
        self.time -= 1
    
    def getNewsFeed(self, userId):
        import heapq
        heap = []
        
        if userId not in self.follows:
            self.follows[userId] = set()
        self.follows[userId].add(userId)
        
        for followeeId in self.follows[userId]:
            if followeeId in self.tweets:
                for time, tweetId in self.tweets[followeeId][-10:]:
                    heapq.heappush(heap, (time, tweetId))
        
        result = []
        while heap and len(result) < 10:
            result.append(heapq.heappop(heap)[1])
        
        return result
    
    def follow(self, followerId, followeeId):
        if followerId not in self.follows:
            self.follows[followerId] = set()
        self.follows[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        if followerId in self.follows:
            self.follows[followerId].discard(followeeId)
```

**Time**: O(k log k) for feed | **Space**: O(n)

**Related**: [Design Twitter](https://leetcode.com/problems/design-twitter/)

---

### Problem 10: Kth Smallest in Sorted Matrix

**Problem**: Find Kth smallest element in sorted matrix.

**Solution**:
```python
def kthSmallest(matrix, k):
    import heapq
    
    n = len(matrix)
    heap = [(matrix[0][0], 0, 0)]
    visited = {(0, 0)}
    
    for _ in range(k - 1):
        val, i, j = heapq.heappop(heap)
        
        if i + 1 < n and (i + 1, j) not in visited:
            heapq.heappush(heap, (matrix[i + 1][j], i + 1, j))
            visited.add((i + 1, j))
        
        if j + 1 < n and (i, j + 1) not in visited:
            heapq.heappush(heap, (matrix[i][j + 1], i, j + 1))
            visited.add((i, j + 1))
    
    return heap[0][0]
```

**Time**: O(k log k) | **Space**: O(k)

**Related**: [Kth Smallest Element in a Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

---

## Key Takeaways

- Heaps are complete binary trees with heap property
- Min heap: parent ≤ children, Max heap: parent ≥ children
- Perfect for priority queues and top K problems
- Two heaps pattern for median and balancing
- K-way merge uses heap to merge sorted lists
- Time complexity: O(log n) for insert/extract, O(1) for peek
- Build heap from array: O(n) time
- Practice top K and two heaps problems - very common

---

## Practice Problems

**Easy**:
- [Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [Last Stone Weight](https://leetcode.com/problems/last-stone-weight/)
- [K Closest Points](https://leetcode.com/problems/k-closest-points-to-origin/)

**Medium**:
- [Top K Frequent](https://leetcode.com/problems/top-k-frequent-elements/)
- [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

**Hard**:
- [Merge K Sorted Arrays](https://www.geeksforgeeks.org/merge-k-sorted-arrays/)
- [Reorganize String](https://leetcode.com/problems/reorganize-string/)
- [Design Twitter](https://leetcode.com/problems/design-twitter/)

