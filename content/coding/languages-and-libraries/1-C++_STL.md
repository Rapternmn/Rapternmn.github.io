+++
title = "C++ STL (Standard Template Library)"
date = 2025-12-09T10:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to C++ STL: containers (vector, set, map, etc.), algorithms, iterators, and utilities. Essential for competitive coding and DSA."
+++

---

## Introduction

The **Standard Template Library (STL)** is a powerful collection of C++ template classes providing common data structures and algorithms. Mastering STL is essential for competitive programming and efficient problem-solving.

**Key Components**:
- **Containers**: Data structures like vector, list, set, map
- **Algorithms**: Functions like sort, find, binary_search
- **Iterators**: Pointers to container elements
- **Function Objects**: Functors and lambdas

---

## STL Containers

### Container Categories

1. **Sequence Containers**: Store elements in linear order
   - `vector`, `deque`, `list`, `array`, `forward_list`

2. **Associative Containers**: Store elements in sorted order
   - `set`, `map`, `multiset`, `multimap`

3. **Unordered Containers**: Hash-based containers
   - `unordered_set`, `unordered_map`, `unordered_multiset`, `unordered_multimap`

4. **Container Adapters**: Wrapper around other containers
   - `stack`, `queue`, `priority_queue`

---

## Sequence Containers

### 1. Vector

**Dynamic array** that can resize automatically.

**Characteristics**:
- Random access: O(1)
- Insert/Delete at end: O(1) amortized
- Insert/Delete in middle: O(n)
- Contiguous memory

**Common Operations**:

```cpp
#include <vector>
#include <iostream>
using namespace std;

vector<int> v;

// Insertion
v.push_back(10);           // Add at end
v.insert(v.begin(), 5);    // Insert at beginning
v.insert(v.begin() + 2, 7); // Insert at position 2

// Access
v[0] = 100;                // Random access
v.at(0) = 200;             // Bounds-checked access
int first = v.front();     // First element
int last = v.back();       // Last element

// Size
int size = v.size();       // Number of elements
bool empty = v.empty();    // Check if empty

// Deletion
v.pop_back();              // Remove last element
v.erase(v.begin() + 1);    // Remove element at index 1
v.erase(v.begin(), v.begin() + 3); // Remove range
v.clear();                 // Remove all elements

// Iteration
for (int i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
}

for (auto it = v.begin(); it != v.end(); it++) {
    cout << *it << " ";
}

for (auto& elem : v) {
    cout << elem << " ";
}
```

**Time Complexity**:
- Access: O(1)
- Search: O(n)
- Insert at end: O(1) amortized
- Insert at position: O(n)
- Delete: O(n)

---

### 2. Deque (Double-Ended Queue)

**Dynamic array** with efficient insertion/deletion at both ends.

**Characteristics**:
- Random access: O(1)
- Insert/Delete at both ends: O(1)
- Insert/Delete in middle: O(n)
- Non-contiguous memory blocks

**Common Operations**:

```cpp
#include <deque>
using namespace std;

deque<int> dq;

// Insertion
dq.push_back(10);          // Add at end
dq.push_front(5);          // Add at front
dq.insert(dq.begin() + 1, 7); // Insert at position

// Access
int front = dq.front();     // First element
int back = dq.back();       // Last element
int elem = dq[2];           // Random access

// Deletion
dq.pop_back();              // Remove from end
dq.pop_front();             // Remove from front
dq.erase(dq.begin() + 1);   // Remove at position
```

**Use Cases**:
- Sliding window problems
- BFS with efficient front operations
- When you need both front and back operations

---

### 3. List

**Doubly linked list** with efficient insertion/deletion anywhere.

**Characteristics**:
- No random access
- Insert/Delete: O(1) if iterator is known
- Search: O(n)
- Non-contiguous memory

**Common Operations**:

```cpp
#include <list>
using namespace std;

list<int> lst;

// Insertion
lst.push_back(10);         // Add at end
lst.push_front(5);          // Add at front
auto it = lst.begin();
advance(it, 2);             // Move iterator
lst.insert(it, 7);          // Insert at position

// Access
int front = lst.front();    // First element
int back = lst.back();      // Last element

// Deletion
lst.pop_back();             // Remove from end
lst.pop_front();            // Remove from front
lst.erase(it);              // Remove at iterator
lst.remove(5);              // Remove all occurrences of 5

// Special operations
lst.sort();                 // Sort the list
lst.unique();               // Remove consecutive duplicates
lst.reverse();              // Reverse the list
lst.merge(other_lst);       // Merge sorted lists
```

**Use Cases**:
- Frequent insertion/deletion in middle
- When order matters and you don't need random access

---

### 4. Array

**Fixed-size array** wrapper (C++11).

**Characteristics**:
- Fixed size at compile time
- Random access: O(1)
- Same interface as vector for fixed-size arrays

```cpp
#include <array>
using namespace std;

array<int, 5> arr = {1, 2, 3, 4, 5};

// Access
arr[0] = 10;
int size = arr.size();
bool empty = arr.empty();

// Iteration
for (auto& elem : arr) {
    cout << elem << " ";
}
```

---

## Associative Containers

### 1. Set

**Sorted collection of unique elements** (typically implemented as red-black tree).

**Characteristics**:
- Unique elements only
- Sorted order
- Search: O(log n)
- Insert/Delete: O(log n)

**Common Operations**:

```cpp
#include <set>
using namespace std;

set<int> s;

// Insertion
s.insert(10);
s.insert(5);
s.insert(15);
s.insert(10);  // Duplicate, ignored

// Search
if (s.find(10) != s.end()) {
    cout << "Found";
}

if (s.count(10)) {  // Returns 0 or 1
    cout << "Exists";
}

// Access
int first = *s.begin();     // Smallest element
int last = *s.rbegin();     // Largest element

// Deletion
s.erase(10);                // Remove element
s.erase(s.begin());         // Remove first element
auto it = s.find(5);
if (it != s.end()) {
    s.erase(it);
}

// Lower/Upper bound
auto lb = s.lower_bound(10); // First element >= 10
auto ub = s.upper_bound(10); // First element > 10

// Range operations
s.erase(s.lower_bound(5), s.upper_bound(15));
```

**Use Cases**:
- Maintaining unique sorted elements
- Range queries
- Finding nearest elements

---

### 2. Map

**Key-value pairs** stored in sorted order by key.

**Characteristics**:
- Unique keys
- Sorted by key
- Search: O(log n)
- Insert/Delete: O(log n)

**Common Operations**:

```cpp
#include <map>
using namespace std;

map<string, int> m;

// Insertion
m["apple"] = 5;
m.insert({"banana", 3});
m.insert(make_pair("cherry", 7));
m.emplace("date", 4);  // Most efficient

// Access
int count = m["apple"];     // Access (creates if not exists)
int count2 = m.at("banana"); // Access (throws if not exists)

// Check existence
if (m.find("apple") != m.end()) {
    cout << "Found";
}

if (m.count("apple")) {  // Returns 0 or 1
    cout << "Exists";
}

// Iteration
for (auto& pair : m) {
    cout << pair.first << ": " << pair.second << endl;
}

for (auto it = m.begin(); it != m.end(); it++) {
    cout << it->first << ": " << it->second << endl;
}

// Deletion
m.erase("apple");
m.erase(m.begin());
auto it = m.find("banana");
if (it != m.end()) {
    m.erase(it);
}

// Lower/Upper bound
auto lb = m.lower_bound("b"); // First key >= "b"
auto ub = m.upper_bound("c"); // First key > "c"
```

**Use Cases**:
- Frequency counting
- Dictionary/mapping problems
- Range queries on keys

---

### 3. Multiset & Multimap

**Allow duplicate elements/keys**.

```cpp
#include <set>
#include <map>
using namespace std;

multiset<int> ms;
ms.insert(10);
ms.insert(10);  // Allowed

multimap<string, int> mm;
mm.insert({"apple", 5});
mm.insert({"apple", 3});  // Allowed
```

---

## Unordered Containers

### 1. Unordered Set

**Hash-based set** with O(1) average operations.

**Characteristics**:
- Average O(1) for insert, delete, search
- Worst case O(n)
- No ordering
- Unique elements

**Common Operations**:

```cpp
#include <unordered_set>
using namespace std;

unordered_set<int> us;

// Insertion
us.insert(10);
us.insert(5);
us.insert(15);

// Search
if (us.find(10) != us.end()) {
    cout << "Found";
}

if (us.count(10)) {
    cout << "Exists";
}

// Deletion
us.erase(10);
us.erase(us.find(5));

// Bucket operations
int bucket_count = us.bucket_count();
int bucket_size = us.bucket_size(0);
```

**Use Cases**:
- Fast lookups when order doesn't matter
- Duplicate detection
- Hash table problems

---

### 2. Unordered Map

**Hash-based key-value pairs** with O(1) average operations.

**Common Operations**:

```cpp
#include <unordered_map>
using namespace std;

unordered_map<string, int> um;

// Insertion
um["apple"] = 5;
um.insert({"banana", 3});
um.emplace("cherry", 7);

// Access
int count = um["apple"];
int count2 = um.at("banana");

// Check existence
if (um.find("apple") != um.end()) {
    cout << "Found";
}

// Deletion
um.erase("apple");
um.erase(um.find("banana"));
```

**Use Cases**:
- Frequency counting (most common)
- Fast key-value lookups
- Hash table problems

---

## Container Adapters

### 1. Stack

**LIFO (Last In First Out)** container.

```cpp
#include <stack>
using namespace std;

stack<int> st;

// Operations
st.push(10);        // Add element
st.push(20);
int top = st.top(); // View top element
st.pop();           // Remove top element
bool empty = st.empty();
int size = st.size();
```

**Use Cases**:
- Expression evaluation
- Parentheses matching
- DFS (iterative)
- Monotonic stack problems

---

### 2. Queue

**FIFO (First In First Out)** container.

```cpp
#include <queue>
using namespace std;

queue<int> q;

// Operations
q.push(10);         // Add at back
q.push(20);
int front = q.front(); // View front element
int back = q.back();   // View back element
q.pop();            // Remove front element
bool empty = q.empty();
int size = q.size();
```

**Use Cases**:
- BFS
- Level-order traversal
- Sliding window (sometimes)

---

### 3. Priority Queue

**Max heap** by default (largest element at top).

```cpp
#include <queue>
using namespace std;

// Max heap (default)
priority_queue<int> pq;

// Min heap
priority_queue<int, vector<int>, greater<int>> min_pq;

// Custom comparator (lambda)
auto cmp = [](int a, int b) { return a > b; };
priority_queue<int, vector<int>, decltype(cmp)> custom_pq(cmp);

// Custom comparator (struct)
struct Compare {
    bool operator()(int a, int b) const {
        return a > b;  // Min heap
    }
};
priority_queue<int, vector<int>, Compare> struct_pq;

// Operations
pq.push(10);
pq.push(5);
pq.push(15);
int top = pq.top();  // Largest element
pq.pop();            // Remove top element
bool empty = pq.empty();
int size = pq.size();
```

**Use Cases**:
- Top K elements
- Merge K sorted lists
- Dijkstra's algorithm
- Median problems

---

## STL Algorithms

### Sorting & Searching

```cpp
#include <algorithm>
#include <vector>
using namespace std;

vector<int> v = {5, 2, 8, 1, 9};

// Sorting
sort(v.begin(), v.end());                    // Ascending
sort(v.begin(), v.end(), greater<int>());    // Descending

// Custom comparator
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b;
});

// Partial sort
partial_sort(v.begin(), v.begin() + 3, v.end()); // First 3 sorted

// Binary search (requires sorted container)
bool found = binary_search(v.begin(), v.end(), 5);
auto it = lower_bound(v.begin(), v.end(), 5);    // First >= 5
auto it2 = upper_bound(v.begin(), v.end(), 5);   // First > 5
auto range = equal_range(v.begin(), v.end(), 5); // [lower, upper)

// Linear search
auto it3 = find(v.begin(), v.end(), 5);
int count = count(v.begin(), v.end(), 5);
bool any = any_of(v.begin(), v.end(), [](int x) { return x > 10; });
bool all = all_of(v.begin(), v.end(), [](int x) { return x > 0; });
```

---

### Modifying Algorithms

```cpp
// Copy
vector<int> dest(5);
copy(v.begin(), v.end(), dest.begin());
copy_if(v.begin(), v.end(), dest.begin(), [](int x) { return x > 5; });

// Fill
fill(v.begin(), v.end(), 0);
fill_n(v.begin(), 3, 10);

// Transform
transform(v.begin(), v.end(), v.begin(), [](int x) { return x * 2; });

// Remove
auto new_end = remove(v.begin(), v.end(), 5);
v.erase(new_end, v.end());  // Actually remove

// Unique (requires sorted)
auto new_end2 = unique(v.begin(), v.end());
v.erase(new_end2, v.end());

// Reverse
reverse(v.begin(), v.end());

// Rotate
rotate(v.begin(), v.begin() + 2, v.end()); // Rotate left by 2
```

---

### Min/Max & Comparisons

```cpp
// Min/Max
int min_val = min(5, 10);
int max_val = max(5, 10);
auto minmax = minmax(5, 10);

// Min/Max element
auto min_it = min_element(v.begin(), v.end());
auto max_it = max_element(v.begin(), v.end());

// Comparison
bool equal = equal(v1.begin(), v1.end(), v2.begin());
bool lex_less = lexicographical_compare(v1.begin(), v1.end(), 
                                        v2.begin(), v2.end());
```

---

### Set Operations

```cpp
#include <algorithm>
#include <vector>
#include <set>
using namespace std;

vector<int> v1 = {1, 2, 3, 4, 5};
vector<int> v2 = {3, 4, 5, 6, 7};
vector<int> result;

// Union
set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), 
          back_inserter(result));

// Intersection
set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(),
                 back_inserter(result));

// Difference
set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),
               back_inserter(result));

// Symmetric difference
set_symmetric_difference(v1.begin(), v1.end(), v2.begin(), v2.end(),
                         back_inserter(result));

// Merge (requires sorted)
merge(v1.begin(), v1.end(), v2.begin(), v2.end(),
      back_inserter(result));
```

---

### Permutations & Combinations

```cpp
vector<int> v = {1, 2, 3};

// Next permutation
do {
    // Process permutation
} while (next_permutation(v.begin(), v.end()));

// Previous permutation
prev_permutation(v.begin(), v.end());

// Check if permutation
bool is_perm = is_permutation(v1.begin(), v1.end(), v2.begin());
```

---

## Iterators

### Iterator Types

1. **Input Iterator**: Read-only, forward
2. **Output Iterator**: Write-only, forward
3. **Forward Iterator**: Read/write, forward
4. **Bidirectional Iterator**: Forward and backward
5. **Random Access Iterator**: Jump to any position

### Common Iterator Operations

```cpp
vector<int> v = {1, 2, 3, 4, 5};

// Iterator declaration
vector<int>::iterator it;
auto it2 = v.begin();

// Operations
it = v.begin();           // First element
it = v.end();             // Past last element
int val = *it;            // Dereference
it++;                     // Next element
it--;                     // Previous element
it += 2;                  // Advance (random access only)
int dist = it2 - it;      // Distance (random access only)

// Reverse iterator
auto rit = v.rbegin();    // Last element
auto rend = v.rend();     // Before first element

// Const iterator
vector<int>::const_iterator cit = v.cbegin();
```

---

## Function Objects & Lambdas

### Function Objects (Functors)

```cpp
// Custom comparator
struct Compare {
    bool operator()(int a, int b) const {
        return a > b;
    }
};

set<int, Compare> s;

// With lambda
auto cmp = [](int a, int b) { return a > b; };
set<int, decltype(cmp)> s2(cmp);
```

### Lambda Expressions

```cpp
// Basic lambda
auto add = [](int a, int b) { return a + b; };
int sum = add(5, 3);

// Capture by value
int x = 10;
auto func1 = [x](int y) { return x + y; };

// Capture by reference
auto func2 = [&x](int y) { x += y; };

// Capture all by value
auto func3 = [=](int y) { return x + y; };

// Capture all by reference
auto func4 = [&](int y) { x += y; };

// Mutable lambda
auto func5 = [x](int y) mutable { x += y; };

// With algorithm
vector<int> v = {1, 2, 3, 4, 5};
for_each(v.begin(), v.end(), [](int& x) { x *= 2; });
auto it = find_if(v.begin(), v.end(), [](int x) { return x > 5; });
```

---

## Utility Functions

### Pair

```cpp
#include <utility>
using namespace std;

pair<int, string> p = make_pair(1, "hello");
pair<int, string> p2 = {2, "world"};

int first = p.first;
string second = p.second;

// Comparison
if (p < p2) {  // Lexicographic comparison
    cout << "p is less";
}
```

### Tuple

```cpp
#include <tuple>
using namespace std;

tuple<int, string, double> t = make_tuple(1, "hello", 3.14);
tuple<int, string, double> t2 = {2, "world", 2.71};

int first = get<0>(t);
string second = get<1>(t);
double third = get<2>(t);

// Structured binding (C++17)
auto [a, b, c] = t;
```

---

## Common Patterns & Tips

### 1. Frequency Counting

```cpp
// Using map
map<int, int> freq;
for (int x : arr) {
    freq[x]++;
}

// Using unordered_map (faster)
unordered_map<int, int> freq2;
for (int x : arr) {
    freq2[x]++;
}
```

### 2. Top K Elements

```cpp
// Using priority queue
priority_queue<int, vector<int>, greater<int>> pq;
for (int x : arr) {
    pq.push(x);
    if (pq.size() > k) {
        pq.pop();
    }
}

// Using nth_element
vector<int> v = arr;
nth_element(v.begin(), v.begin() + k, v.end(), greater<int>());
```

### 3. Removing Duplicates

```cpp
// Sort and unique
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());

// Using set
set<int> s(v.begin(), v.end());
vector<int> unique_v(s.begin(), s.end());
```

### 4. Custom Sorting

```cpp
// Sort by multiple criteria
vector<pair<int, int>> v = {{1, 3}, {2, 2}, {1, 1}};
sort(v.begin(), v.end(), [](auto& a, auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return a.second > b.second;
});
```

### 5. Binary Search on Answer

```cpp
int left = 0, right = max_val;
while (left < right) {
    int mid = left + (right - left) / 2;
    if (isValid(mid)) {
        right = mid;
    } else {
        left = mid + 1;
    }
}
```

---

## Performance Tips

1. **Use `emplace` instead of `insert`** when possible (avoids temporary objects)
2. **Reserve space** for vectors when size is known: `v.reserve(n)`
3. **Use `unordered_map`** instead of `map` when order doesn't matter
4. **Prefer `algorithm` functions** over manual loops when applicable
5. **Use `const_iterator`** for read-only operations
6. **Avoid `std::endl`** in loops (use `'\n'` instead)

---

## Summary

**Container Selection Guide**:
- **Vector**: Default choice for dynamic arrays
- **Deque**: When you need efficient front/back operations
- **Set/Map**: When you need sorted order and fast search
- **Unordered Set/Map**: When order doesn't matter and you need O(1) average
- **Priority Queue**: For heap operations
- **Stack/Queue**: For LIFO/FIFO operations

**Algorithm Selection**:
- **Sort**: `sort()` for general sorting
- **Search**: `binary_search()` for sorted, `find()` for linear
- **Transform**: `transform()` for element-wise operations
- **Count**: `count()`, `count_if()` for counting

Mastering STL significantly improves coding efficiency and problem-solving speed in competitive programming!

