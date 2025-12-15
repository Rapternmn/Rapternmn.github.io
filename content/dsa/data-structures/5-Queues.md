+++
title = "Queues"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Queues data structure: FIFO principle, operations, and top interview problems. Covers queue variations, BFS, and more."
+++

---

## Introduction

Queues are linear data structures that follow the First-In-First-Out (FIFO) principle. They are essential for BFS, scheduling, and buffering problems.

---

## Queue Fundamentals

### What is a Queue?

**Queue**: A linear data structure where elements are added at the rear and removed from the front.

**Key Characteristics**:
- **FIFO**: First In, First Out
- **Front**: Element to be removed next
- **Rear**: Where new elements are added
- **Operations**: Enqueue (add), Dequeue (remove), Peek/Front (view)

### Operations

| Operation | Description | Time Complexity |
|-----------|-------------|----------------|
| Enqueue | Add element to rear | O(1) |
| Dequeue | Remove element from front | O(1) |
| Peek/Front | View front element | O(1) |
| IsEmpty | Check if empty | O(1) |
| Size | Get number of elements | O(1) |

---

## Implementation

### Using List (Python - inefficient)

```python
class Queue:
    def __init__(self):
        self.items = []
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.pop(0)  # O(n) operation
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Using collections.deque (Python - efficient)

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()
    
    def enqueue(self, item):
        self.items.append(item)
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()  # O(1) operation
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Using Linked List

```python
class QueueNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

class Queue:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0
    
    def enqueue(self, val):
        new_node = QueueNode(val)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self.size += 1
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        val = self.front.val
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self.size -= 1
        return val
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.front.val
    
    def is_empty(self):
        return self.front is None
```

---

## Queue Variations

### 1. Circular Queue

**Pattern**: Fixed-size queue with wrap-around.

<details open>
<summary><strong>📋 C++ Implementation</strong></summary>

```cpp
class CircularQueue {
private:
    vector<int> queue;
    int front;
    int rear;
    int size;
    int capacity;
    
public:
    CircularQueue(int k) {
        queue.resize(k);
        front = 0;
        rear = -1;
        size = 0;
        capacity = k;
    }
    
    bool enqueue(int value) {
        if (isFull()) return false;
        rear = (rear + 1) % capacity;
        queue[rear] = value;
        size++;
        return true;
    }
    
    int dequeue() {
        if (isEmpty()) return -1;
        int value = queue[front];
        front = (front + 1) % capacity;
        size--;
        return value;
    }
    
    int peek() {
        if (isEmpty()) return -1;
        return queue[front];
    }
    
    bool isEmpty() {
        return size == 0;
    }
    
    bool isFull() {
        return size == capacity;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Implementation</strong></summary>

```python
class CircularQueue:
    def __init__(self, k):
        self.queue = [0] * k
        self.front = 0
        self.rear = -1
        self.size = 0
        self.capacity = k
    
    def enqueue(self, value):
        if self.is_full():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = value
        self.size += 1
        return True
    
    def dequeue(self):
        if self.is_empty():
            return -1
        value = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return value
    
    def peek(self):
        if self.is_empty():
            return -1
        return self.queue[self.front]
    
    def is_empty(self):
        return self.size == 0
    
    def is_full(self):
        return self.size == self.capacity
```

</details>

---

### 2. Priority Queue

**Pattern**: Elements with priority, highest priority dequeued first.

<details open>
<summary><strong>📋 C++ Implementation</strong></summary>

```cpp
#include <queue>
#include <vector>

class PriorityQueue {
private:
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> heap;
    
public:
    void enqueue(int item, int priority) {
        heap.push({priority, item});
    }
    
    int dequeue() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty");
        }
        int item = heap.top().second;
        heap.pop();
        return item;
    }
    
    int peek() {
        if (isEmpty()) {
            throw runtime_error("Queue is empty");
        }
        return heap.top().second;
    }
    
    bool isEmpty() {
        return heap.empty();
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Implementation</strong></summary>

```python
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
    
    def enqueue(self, item, priority):
        heapq.heappush(self.heap, (priority, item))
    
    def dequeue(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        priority, item = heapq.heappop(self.heap)
        return item
    
    def peek(self):
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.heap[0][1]
    
    def is_empty(self):
        return len(self.heap) == 0
```

</details>

---

### 3. Double-Ended Queue (Deque)

**Pattern**: Add/remove from both ends.

```python
from collections import deque

# Built-in deque supports:
# appendleft(), popleft() - front operations
# append(), pop() - rear operations
```

---

## Common Patterns

### 1. BFS (Breadth-First Search)

**Pattern**: Use queue for level-order traversal.

**When to Use**:
- Tree level-order traversal
- Graph BFS
- Shortest path (unweighted)
- Level-by-level processing

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> bfs(TreeNode* root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> level;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left != nullptr) {
                q.push(node->left);
            }
            if (node->right != nullptr) {
                q.push(node->right);
            }
        }
        
        result.push_back(level);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def bfs(root):
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

**Related Pattern**: See [Tree BFS/DFS]({{< ref "../coding-patterns/6-Tree_BFS_DFS.md" >}}) and [Graph BFS/DFS]({{< ref "../coding-patterns/14-Graph_BFS_DFS.md" >}})

---

### 2. Sliding Window Maximum

**Pattern**: Use deque to maintain window maximum.

**When to Use**:
- Maximum in sliding window
- Minimum in sliding window
- Range queries

**Template**:
```python
def sliding_window_maximum(nums, k):
    from collections import deque
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

---

## Top Problems

### Problem 1: Implement Queue using Stacks

**Problem**: Implement queue using two stacks.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class MyQueue {
private:
    stack<int> stack1;
    stack<int> stack2;
    
public:
    void push(int x) {
        stack1.push(x);
    }
    
    int pop() {
        peek();
        int val = stack2.top();
        stack2.pop();
        return val;
    }
    
    int peek() {
        if (stack2.empty()) {
            while (!stack1.empty()) {
                stack2.push(stack1.top());
                stack1.pop();
            }
        }
        return stack2.top();
    }
    
    bool empty() {
        return stack1.empty() && stack2.empty();
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    
    def push(self, x):
        self.stack1.append(x)
    
    def pop(self):
        self.peek()
        return self.stack2.pop()
    
    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]
    
    def empty(self):
        return not self.stack1 and not self.stack2
```

</details>

**Time**: O(1) amortized | **Space**: O(n)

**Related**: [Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)

---

### Problem 2: Design Circular Queue

**Problem**: Design circular queue with fixed size.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class MyCircularQueue {
private:
    vector<int> queue;
    int front;
    int rear;
    int size;
    int capacity;
    
public:
    MyCircularQueue(int k) {
        queue.resize(k);
        front = 0;
        rear = -1;
        size = 0;
        capacity = k;
    }
    
    bool enQueue(int value) {
        if (isFull()) return false;
        rear = (rear + 1) % capacity;
        queue[rear] = value;
        size++;
        return true;
    }
    
    bool deQueue() {
        if (isEmpty()) return false;
        front = (front + 1) % capacity;
        size--;
        return true;
    }
    
    int Front() {
        return isEmpty() ? -1 : queue[front];
    }
    
    int Rear() {
        return isEmpty() ? -1 : queue[rear];
    }
    
    bool isEmpty() {
        return size == 0;
    }
    
    bool isFull() {
        return size == capacity;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
class MyCircularQueue:
    def __init__(self, k):
        self.queue = [0] * k
        self.front = 0
        self.rear = -1
        self.size = 0
        self.capacity = k
    
    def enQueue(self, value):
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = value
        self.size += 1
        return True
    
    def deQueue(self):
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True
    
    def Front(self):
        return -1 if self.isEmpty() else self.queue[self.front]
    
    def Rear(self):
        return -1 if self.isEmpty() else self.queue[self.rear]
    
    def isEmpty(self):
        return self.size == 0
    
    def isFull(self):
        return self.size == self.capacity
```

**Time**: O(1) for all operations | **Space**: O(k)

**Related**: [Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)

---

### Problem 3: Sliding Window Maximum

**Problem**: Find maximum in each sliding window of size k.

**Solution** (Monotonic Deque):
```python
def maxSlidingWindow(nums, k):
    from collections import deque
    
    dq = deque()
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

**Time**: O(n) | **Space**: O(k)

**Related**: [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)

---

### Problem 4: Binary Tree Level Order Traversal

**Problem**: Return level-order traversal of binary tree.

**Solution** (BFS):
```python
def levelOrder(root):
    if not root:
        return []
    
    from collections import deque
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

**Time**: O(n) | **Space**: O(w) where w is max width

**Related**: [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

**Pattern**: BFS

---

### Problem 5: Design Hit Counter

**Problem**: Design hit counter that counts hits in past 5 minutes.

**Solution**:
```python
class HitCounter:
    def __init__(self):
        self.hits = []
    
    def hit(self, timestamp):
        self.hits.append(timestamp)
    
    def getHits(self, timestamp):
        # Remove hits older than 5 minutes
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.pop(0)
        return len(self.hits)
```

**Time**: O(n) for getHits | **Space**: O(n)

**Related**: [Design Hit Counter](https://leetcode.com/problems/design-hit-counter/)

---

### Problem 6: Moving Average from Data Stream

**Problem**: Calculate moving average of last k numbers.

**Solution**:
```python
class MovingAverage:
    def __init__(self, size):
        self.size = size
        self.queue = deque()
        self.sum = 0
    
    def next(self, val):
        self.queue.append(val)
        self.sum += val
        
        if len(self.queue) > self.size:
            self.sum -= self.queue.popleft()
        
        return self.sum / len(self.queue)
```

**Time**: O(1) | **Space**: O(k)

**Related**: [Moving Average from Data Stream](https://leetcode.com/problems/moving-average-from-data-stream/)

---

### Problem 7: Task Scheduler

**Problem**: Find minimum time to complete tasks with cooldown.

**Solution**:
```python
def leastInterval(tasks, n):
    from collections import Counter, deque
    
    count = Counter(tasks)
    max_freq = max(count.values())
    max_count = sum(1 for v in count.values() if v == max_freq)
    
    # Calculate minimum time
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Task Scheduler](https://leetcode.com/problems/task-scheduler/)

---

### Problem 8: Design Bounded Blocking Queue

**Problem**: Thread-safe bounded blocking queue.

**Solution**:
```python
import threading

class BoundedBlockingQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = deque()
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)
    
    def enqueue(self, element):
        with self.not_full:
            while len(self.queue) >= self.capacity:
                self.not_full.wait()
            self.queue.append(element)
            self.not_empty.notify()
    
    def dequeue(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()
            element = self.queue.popleft()
            self.not_full.notify()
            return element
    
    def size(self):
        with self.lock:
            return len(self.queue)
```

**Time**: O(1) | **Space**: O(capacity)

**Related**: [Design Bounded Blocking Queue](https://leetcode.com/problems/design-bounded-blocking-queue/)

---

## Advanced Patterns

### 1. BFS for Shortest Path

**Pattern**: Find shortest path in unweighted graph.

```python
def shortest_path(graph, start, end):
    from collections import deque
    
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, distance = queue.popleft()
        
        if node == end:
            return distance
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1
```

---

### 2. Level Order Traversal Variations

**Pattern**: Zigzag, right-side view, etc.

```python
def zigzagLevelOrder(root):
    if not root:
        return []
    
    from collections import deque
    queue = deque([root])
    result = []
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level))
        left_to_right = not left_to_right
    
    return result
```

---

## Key Takeaways

- Queues follow FIFO principle - first element added is first removed
- Essential for BFS and level-order traversals
- Circular queues use modulo arithmetic for wrap-around
- Priority queues use heap for O(log n) operations
- Deque allows efficient operations at both ends
- Thread-safe queues need synchronization primitives
- Sliding window problems often use monotonic deque
- Time complexity: O(1) for basic operations
- Practice BFS problems - very common in interviews

---

## Practice Problems

**Easy**:
- [Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/)
- [Number of Recent Calls](https://leetcode.com/problems/number-of-recent-calls/)
- [Design Circular Queue](https://leetcode.com/problems/design-circular-queue/)

**Medium**:
- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [Task Scheduler](https://leetcode.com/problems/task-scheduler/)
- [Design Hit Counter](https://leetcode.com/problems/design-hit-counter/)

**Hard**:
- [Design Bounded Blocking Queue](https://leetcode.com/problems/design-bounded-blocking-queue/)
- [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
- [Word Ladder](https://leetcode.com/problems/word-ladder/)

