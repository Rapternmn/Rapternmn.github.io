+++
title = "Python Libraries for DSA & Competitive Coding"
date = 2025-12-09T10:00:00+05:30
draft = false
weight = 2
description = "Essential Python libraries for competitive coding and DSA: collections, itertools, heapq, bisect, and more. Master Python tools for efficient problem-solving."
+++

---

## Introduction

Python is widely used in competitive programming and DSA due to its simplicity and powerful built-in libraries. This guide covers essential Python libraries and modules that make problem-solving more efficient.

**Key Libraries**:
- **collections**: Advanced data structures
- **itertools**: Iterator tools
- **heapq**: Heap operations
- **bisect**: Binary search
- **defaultdict**: Default dictionaries
- **Counter**: Frequency counting
- **deque**: Double-ended queue

---

## Collections Module

### 1. Counter

**Frequency counting** made easy.

**Use Cases**:
- Count element frequencies
- Find most common elements
- Compare frequencies

**Common Operations**:

```python
from collections import Counter

# Create counter
arr = [1, 2, 2, 3, 3, 3, 4]
counter = Counter(arr)
# Counter({3: 3, 2: 2, 1: 1, 4: 1})

# Access counts
count_3 = counter[3]  # 3
count_5 = counter[5]  # 0 (doesn't raise KeyError)

# Most common
most_common = counter.most_common(2)  # [(3, 3), (2, 2)]
most_common_elem = counter.most_common(1)[0][0]  # 3

# Update counter
counter.update([3, 3, 5])  # Add more elements
counter.subtract([2, 2])   # Subtract counts

# Operations
counter1 = Counter([1, 2, 2, 3])
counter2 = Counter([2, 3, 3, 4])
sum_counter = counter1 + counter2      # Addition
diff_counter = counter1 - counter2      # Subtraction
intersection = counter1 & counter2      # Minimum
union = counter1 | counter2            # Maximum

# Convert to dict/list
freq_dict = dict(counter)
elements = list(counter.elements())     # [1, 2, 2, 3, 3, 3]
```

**Example Problems**:
- Find majority element
- Anagram detection
- Frequency-based sorting
- Top K frequent elements

---

### 2. defaultdict

**Dictionary with default factory** - no KeyError for missing keys.

**Use Cases**:
- Grouping elements
- Building adjacency lists
- Frequency counting without initialization

**Common Operations**:

```python
from collections import defaultdict

# Default to int (0)
dd_int = defaultdict(int)
dd_int['a'] += 1  # No KeyError, starts at 0
dd_int['b'] += 2
# defaultdict(<class 'int'>, {'a': 1, 'b': 2})

# Default to list ([])
dd_list = defaultdict(list)
dd_list['group1'].append(1)
dd_list['group1'].append(2)
dd_list['group2'].append(3)
# defaultdict(<class 'list'>, {'group1': [1, 2], 'group2': [3]})

# Default to set (set())
dd_set = defaultdict(set)
dd_set['set1'].add(1)
dd_set['set1'].add(2)
dd_set['set1'].add(1)  # Duplicate ignored
# defaultdict(<class 'set'>, {'set1': {1, 2}})

# Custom default factory
dd_custom = defaultdict(lambda: 'N/A')
value = dd_custom['missing']  # 'N/A'

# Grouping example
words = ['apple', 'banana', 'apricot', 'berry']
grouped = defaultdict(list)
for word in words:
    grouped[word[0]].append(word)
# {'a': ['apple', 'apricot'], 'b': ['banana', 'berry']}
```

**Example Problems**:
- Group anagrams
- Build graph adjacency list
- Group by criteria
- Frequency counting

---

### 3. deque (Double-Ended Queue)

**Efficient insertion/deletion at both ends** - O(1) operations.

**Use Cases**:
- BFS implementation
- Sliding window
- Palindrome checking
- Queue/Stack operations

**Common Operations**:

```python
from collections import deque

# Create deque
dq = deque([1, 2, 3, 4, 5])

# Add elements
dq.append(6)           # Add at right end: [1, 2, 3, 4, 5, 6]
dq.appendleft(0)       # Add at left end: [0, 1, 2, 3, 4, 5, 6]

# Remove elements
right = dq.pop()        # Remove from right: 6
left = dq.popleft()     # Remove from left: 0

# Access
first = dq[0]           # First element
last = dq[-1]          # Last element
middle = dq[2]         # Random access (O(1))

# Rotation
dq.rotate(1)            # Rotate right by 1
dq.rotate(-1)           # Rotate left by 1

# Size
length = len(dq)
is_empty = len(dq) == 0

# Clear
dq.clear()

# BFS example
def bfs(graph, start):
    queue = deque([start])
    visited = set([start])
    
    while queue:
        node = queue.popleft()
        # Process node
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

**Example Problems**:
- BFS traversal
- Sliding window maximum
- Palindrome checking
- Level-order traversal

---

### 4. OrderedDict

**Dictionary that remembers insertion order** (Python 3.7+ dicts are ordered by default, but OrderedDict has extra methods).

**Use Cases**:
- LRU Cache implementation
- Maintaining insertion order
- Moving items to end

**Common Operations**:

```python
from collections import OrderedDict

od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3

# Move to end
od.move_to_end('a')     # Move 'a' to end
od.move_to_end('b', last=False)  # Move 'b' to beginning

# Pop
last_item = od.popitem()        # Remove last (LIFO)
first_item = od.popitem(last=False)  # Remove first (FIFO)

# LRU Cache example
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

### 5. namedtuple

**Tuple with named fields** - more readable than regular tuples.

**Use Cases**:
- Representing structured data
- Point/Coordinate representation
- Graph edges

**Common Operations**:

```python
from collections import namedtuple

# Define named tuple
Point = namedtuple('Point', ['x', 'y'])
p1 = Point(1, 2)
p2 = Point(3, 4)

# Access by name
x_coord = p1.x  # 1
y_coord = p1.y  # 2

# Access by index
x_coord = p1[0]  # 1

# Unpacking
x, y = p1

# Immutable
# p1.x = 5  # Error: can't modify

# Graph edge example
Edge = namedtuple('Edge', ['src', 'dest', 'weight'])
edges = [
    Edge(0, 1, 5),
    Edge(1, 2, 3),
    Edge(2, 3, 2)
]

for edge in edges:
    print(f"{edge.src} -> {edge.dest}: {edge.weight}")
```

---

## Itertools Module

**Powerful iterator tools** for efficient iteration and combinations.

### 1. Combinations & Permutations

```python
from itertools import combinations, permutations, product

arr = [1, 2, 3, 4]

# Combinations (order doesn't matter)
comb_2 = list(combinations(arr, 2))
# [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# Combinations with replacement
comb_repl = list(combinations_with_replacement(arr, 2))
# [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]

# Permutations (order matters)
perm_2 = list(permutations(arr, 2))
# [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)]

# Cartesian product
prod = list(product([1, 2], [3, 4]))
# [(1, 3), (1, 4), (2, 3), (2, 4)]

# Multiple iterables
prod_multi = list(product([1, 2], repeat=2))
# [(1, 1), (1, 2), (2, 1), (2, 2)]
```

**Example Problems**:
- Generate all subsets
- Combination sum
- Permutation problems
- Cartesian product problems

---

### 2. Grouping & Chaining

```python
from itertools import groupby, chain, accumulate

# Group consecutive elements
arr = [1, 1, 2, 2, 2, 3, 1, 1]
grouped = groupby(arr)
for key, group in grouped:
    print(f"{key}: {list(group)}")
# 1: [1, 1]
# 2: [2, 2, 2]
# 3: [3]
# 1: [1, 1]

# Group by key function
words = ['apple', 'apricot', 'banana', 'berry']
grouped_by_first = groupby(sorted(words), key=lambda x: x[0])
for key, group in grouped_by_first:
    print(f"{key}: {list(group)}")

# Chain iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
chained = list(chain(list1, list2))
# [1, 2, 3, 4, 5, 6]

# Accumulate (prefix sums)
arr = [1, 2, 3, 4, 5]
prefix_sum = list(accumulate(arr))
# [1, 3, 6, 10, 15]

# With custom function
prefix_max = list(accumulate(arr, max))
# [1, 2, 3, 4, 5]
```

---

### 3. Cycle & Repeat

```python
from itertools import cycle, repeat, count

# Cycle through iterable
cycled = cycle([1, 2, 3])
for i, val in enumerate(cycled):
    if i >= 5:
        break
    print(val)  # 1, 2, 3, 1, 2

# Repeat value
repeated = repeat(5, 3)  # Repeat 5, 3 times
list(repeated)  # [5, 5, 5]

# Count (infinite counter)
counter = count(10, 2)  # Start at 10, step by 2
for i, val in enumerate(counter):
    if i >= 5:
        break
    print(val)  # 10, 12, 14, 16, 18
```

---

### 4. Advanced Iteration

```python
from itertools import islice, takewhile, dropwhile, filterfalse

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Slice iterator
sliced = list(islice(arr, 2, 7, 2))  # [3, 5, 7]

# Take while condition
taken = list(takewhile(lambda x: x < 5, arr))  # [1, 2, 3, 4]

# Drop while condition
dropped = list(dropwhile(lambda x: x < 5, arr))  # [5, 6, 7, 8, 9, 10]

# Filter false (opposite of filter)
filtered = list(filterfalse(lambda x: x % 2 == 0, arr))  # [1, 3, 5, 7, 9]
```

---

## Heapq Module

**Heap operations** - min heap implementation.

**Use Cases**:
- Priority queue
- Top K elements
- Merge K sorted lists
- Dijkstra's algorithm

**Common Operations**:

```python
import heapq

# Create heap (min heap by default)
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)
heapq.heappush(heap, 1)
# heap: [1, 2, 8, 5]

# Pop smallest
smallest = heapq.heappop(heap)  # 1
# heap: [2, 5, 8]

# Peek smallest (without popping)
smallest = heap[0]  # 2

# Heapify existing list
arr = [5, 2, 8, 1, 9]
heapq.heapify(arr)  # Convert to heap in-place
# arr: [1, 2, 8, 5, 9]

# Push and pop in one operation
heap = [1, 2, 3]
smallest = heapq.heappushpop(heap, 0)  # Push 0, pop smallest (1)
# smallest: 1, heap: [0, 2, 3]

# Pop and push in one operation
smallest = heapq.heapreplace(heap, 4)  # Pop smallest, push 4
# smallest: 0, heap: [2, 3, 4]

# N largest/smallest
arr = [1, 5, 3, 9, 2, 8, 4]
largest_3 = heapq.nlargest(3, arr)  # [9, 8, 5]
smallest_3 = heapq.nsmallest(3, arr)  # [1, 2, 3]

# Max heap (negate values)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -2)
heapq.heappush(max_heap, -8)
largest = -heapq.heappop(max_heap)  # 8

# Heap with tuples (sorts by first element)
heap = []
heapq.heappush(heap, (3, 'task3'))
heapq.heappush(heap, (1, 'task1'))
heapq.heappush(heap, (2, 'task2'))
priority, task = heapq.heappop(heap)  # (1, 'task1')
```

**Example Problems**:
- Top K frequent elements
- Merge K sorted lists
- Find Kth largest element
- Dijkstra's shortest path
- Meeting rooms II

---

## Bisect Module

**Binary search operations** on sorted sequences.

**Use Cases**:
- Binary search
- Insert in sorted order
- Find insertion point
- Range queries

**Common Operations**:

```python
import bisect

arr = [1, 3, 5, 7, 9, 11]

# bisect_left: leftmost insertion point
idx = bisect.bisect_left(arr, 5)   # 2 (first position where 5 can be inserted)
idx = bisect.bisect_left(arr, 6)   # 3 (first position where 6 can be inserted)

# bisect_right / bisect: rightmost insertion point
idx = bisect.bisect_right(arr, 5)  # 3 (first position after 5)
idx = bisect.bisect(arr, 5)         # Same as bisect_right

# insort_left: insert and maintain sorted order
bisect.insort_left(arr, 6)  # Insert 6 at leftmost position
# arr: [1, 3, 5, 6, 7, 9, 11]

# insort_right / insort: insert at rightmost position
bisect.insort_right(arr, 5)  # Insert 5 at rightmost position
# arr: [1, 3, 5, 5, 6, 7, 9, 11]

# Binary search implementation
def binary_search(arr, target):
    idx = bisect.bisect_left(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1

# Find range of target
def find_range(arr, target):
    left = bisect.bisect_left(arr, target)
    right = bisect.bisect_right(arr, target)
    if left == right:
        return [-1, -1]
    return [left, right - 1]

# Count elements in range
def count_in_range(arr, left_val, right_val):
    left_idx = bisect.bisect_left(arr, left_val)
    right_idx = bisect.bisect_right(arr, right_val)
    return right_idx - left_idx
```

**Example Problems**:
- Search in rotated sorted array
- Find insertion position
- Count elements in range
- Kth smallest in sorted matrix

---

## Built-in Functions & Methods

### 1. Sorting

```python
# List sorting
arr = [3, 1, 4, 1, 5, 9, 2, 6]
arr.sort()                    # In-place: [1, 1, 2, 3, 4, 5, 6, 9]
sorted_arr = sorted(arr)      # New list

# Reverse sort
arr.sort(reverse=True)        # Descending
sorted_arr = sorted(arr, reverse=True)

# Custom key
arr = ['apple', 'banana', 'apricot']
arr.sort(key=len)             # Sort by length
arr.sort(key=lambda x: x[1])  # Sort by second character

# Multiple criteria
arr = [(1, 2), (2, 1), (1, 1)]
arr.sort(key=lambda x: (x[0], -x[1]))  # Sort by first asc, second desc
```

### 2. Enumerate & Zip

```python
# Enumerate (index, value)
arr = ['a', 'b', 'c']
for i, val in enumerate(arr):
    print(f"{i}: {val}")  # 0: a, 1: b, 2: c

for i, val in enumerate(arr, start=1):
    print(f"{i}: {val}")  # 1: a, 2: b, 3: c

# Zip (combine iterables)
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
zipped = list(zip(list1, list2))  # [(1, 'a'), (2, 'b'), (3, 'c')]

# Unzip
unzipped = list(zip(*zipped))  # [(1, 2, 3), ('a', 'b', 'c')]

# Transpose matrix
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = list(zip(*matrix))  # [(1, 4), (2, 5), (3, 6)]
```

### 3. Map, Filter, Reduce

```python
from functools import reduce

# Map (apply function to all elements)
arr = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, arr))  # [1, 4, 9, 16, 25]

# Filter (keep elements matching condition)
evens = list(filter(lambda x: x % 2 == 0, arr))  # [2, 4]

# Reduce (accumulate with function)
sum_all = reduce(lambda x, y: x + y, arr)  # 15
product = reduce(lambda x, y: x * y, arr)  # 120
```

### 4. Any & All

```python
arr = [1, 2, 3, 4, 5]

# Any (returns True if any element is truthy)
has_even = any(x % 2 == 0 for x in arr)  # True
has_negative = any(x < 0 for x in arr)   # False

# All (returns True if all elements are truthy)
all_positive = all(x > 0 for x in arr)   # True
all_even = all(x % 2 == 0 for x in arr)   # False
```

---

## String Operations

### 1. String Methods

```python
s = "Hello World"

# Case conversion
s.upper()        # "HELLO WORLD"
s.lower()        # "hello world"
s.title()        # "Hello World"
s.capitalize()   # "Hello world"
s.swapcase()     # "hELLO wORLD"

# Checking
s.isalpha()      # False (has space)
s.isdigit()      # False
s.isalnum()      # False
s.isspace()      # False
s.startswith("Hello")  # True
s.endswith("World")    # True

# Splitting & Joining
words = s.split()           # ["Hello", "World"]
chars = list(s)             # ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd']
joined = " ".join(words)    # "Hello World"

# Stripping
s = "  hello  "
s.strip()        # "hello"
s.lstrip()       # "hello  "
s.rstrip()       # "  hello"

# Finding
s = "hello world"
s.find("lo")     # 3 (first occurrence)
s.rfind("lo")    # 3 (last occurrence)
s.index("lo")    # 3 (raises ValueError if not found)
s.count("l")     # 3

# Replacing
s.replace("world", "Python")  # "hello Python"
```

### 2. String Formatting

```python
# f-strings (Python 3.6+)
name = "Alice"
age = 30
msg = f"My name is {name} and I'm {age} years old"

# Format method
msg = "My name is {} and I'm {} years old".format(name, age)
msg = "My name is {name} and I'm {age} years old".format(name=name, age=age)

# Format specifiers
num = 3.14159
f"{num:.2f}"     # "3.14"
f"{num:10.2f}"   # "      3.14"
f"{num:010.2f}"  # "0000003.14"
```

---

## Common Patterns & Tips

### 1. Two Sum with Dictionary

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

### 2. Group Anagrams

```python
from collections import defaultdict

def group_anagrams(strs):
    grouped = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        grouped[key].append(s)
    return list(grouped.values())
```

### 3. Top K Frequent Elements

```python
from collections import Counter
import heapq

def top_k_frequent(nums, k):
    counter = Counter(nums)
    return [num for num, _ in counter.most_common(k)]

# Using heap
def top_k_frequent_heap(nums, k):
    counter = Counter(nums)
    heap = [(-freq, num) for num, freq in counter.items()]
    heapq.heapify(heap)
    return [heapq.heappop(heap)[1] for _ in range(k)]
```

### 4. Sliding Window Maximum

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

### 5. Binary Search Template

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

---

## Performance Tips

1. **Use list comprehensions** instead of loops when possible
2. **Use `set` for O(1) lookups** instead of `list`
3. **Use `collections.deque`** for queue operations (faster than list)
4. **Use `Counter`** for frequency counting (cleaner than dict)
5. **Use `defaultdict`** to avoid KeyError checks
6. **Use generators** for large sequences to save memory
7. **Use `bisect`** for binary search on sorted lists
8. **Avoid string concatenation in loops** (use `join` instead)

---

## Summary

**Library Selection Guide**:
- **Counter**: Frequency counting
- **defaultdict**: Grouping, building graphs
- **deque**: BFS, sliding window, queue operations
- **heapq**: Priority queue, top K problems
- **bisect**: Binary search, sorted insertion
- **itertools**: Combinations, permutations, grouping
- **collections**: Advanced data structures

**Common Use Cases**:
- **Frequency**: `Counter`, `defaultdict(int)`
- **Grouping**: `defaultdict(list)`, `groupby`
- **Queue/Stack**: `deque`
- **Priority Queue**: `heapq`
- **Binary Search**: `bisect`
- **Combinations**: `itertools.combinations`
- **Permutations**: `itertools.permutations`

Mastering these Python libraries significantly improves coding efficiency and makes problem-solving more elegant!

