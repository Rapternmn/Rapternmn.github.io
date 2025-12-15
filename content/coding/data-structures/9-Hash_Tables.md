+++
title = "Hash Tables"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 9
description = "Hash Tables data structure: hash maps, collision handling, and top interview problems. Covers hashing, dictionaries, and more."
+++

---

## Introduction

Hash tables (hash maps, dictionaries) provide key-value storage with average O(1) access time. They are fundamental for fast lookups, frequency counting, and caching.

---

## Hash Table Fundamentals

### What is a Hash Table?

**Hash Table**: A data structure that maps keys to values using a hash function.

**Key Characteristics**:
- **Key-Value Pairs**: Store data as key-value mappings
- **Hash Function**: Maps keys to array indices
- **Fast Access**: Average O(1) lookup, insert, delete
- **Collision Handling**: Multiple keys may hash to same index

### Operations

| Operation | Average Time | Worst Time |
|-----------|--------------|------------|
| Insert | O(1) | O(n) |
| Lookup | O(1) | O(n) |
| Delete | O(1) | O(n) |
| Search | O(1) | O(n) |

### Collision Handling

**1. Chaining**: Store collisions in linked list/bucket
**2. Open Addressing**: Find next available slot
   - Linear Probing
   - Quadratic Probing
   - Double Hashing

---

## Implementation

### Using Dictionary (Python)

```python
# Built-in dictionary
hash_map = {}
hash_map['key'] = 'value'
value = hash_map.get('key')
del hash_map['key']

# Using defaultdict
from collections import defaultdict
dd = defaultdict(int)
dd['key'] += 1  # No KeyError
```

### Custom Hash Table

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.buckets = [[] for _ in range(size)]
    
    def _hash(self, key):
        return hash(key) % self.size
    
    def put(self, key, value):
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    
    def get(self, key):
        bucket = self.buckets[self._hash(key)]
        for k, v in bucket:
            if k == key:
                return v
        return None
    
    def remove(self, key):
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return
```

---

## Common Patterns

### 1. Frequency Counting

**Pattern**: Count occurrences of elements.

**Template**:
```python
from collections import Counter

def count_frequency(arr):
    freq = Counter(arr)
    # Or manually
    freq = {}
    for item in arr:
        freq[item] = freq.get(item, 0) + 1
    return freq
```

---

### 2. Two Sum Pattern

**Pattern**: Find pairs that sum to target.

**Template**:
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

---

### 3. Grouping Pattern

**Pattern**: Group elements by some property.

**Template**:
```python
from collections import defaultdict

def group_by_key(items, key_func):
    groups = defaultdict(list)
    for item in items:
        groups[key_func(item)].append(item)
    return groups
```

---

## Top Problems

### Problem 1: Two Sum

**Problem**: Find two numbers that add up to target.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.find(complement) != seen.end()) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    
    return {};
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Two Sum](https://leetcode.com/problems/two-sum/)

---

### Problem 2: Group Anagrams

**Problem**: Group strings that are anagrams.

**Solution**:
```python
def groupAnagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())
```

**Time**: O(n * k log k) where k is max string length | **Space**: O(n * k)

**Related**: [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

---

### Problem 3: Longest Substring Without Repeating Characters

**Problem**: Find length of longest substring without repeating chars.

**Solution**:
```python
def lengthOfLongestSubstring(s):
    char_map = {}
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        if s[right] in char_map and char_map[s[right]] >= left:
            left = char_map[s[right]] + 1
        
        char_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

**Time**: O(n) | **Space**: O(min(n, m)) where m is charset size

**Related**: [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

---

### Problem 4: LRU Cache

**Problem**: Design LRU (Least Recently Used) cache.

**Solution**:
```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return -1
    
    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            if len(self.cache) >= self.capacity:
                lru = self.order.pop(0)
                del self.cache[lru]
            self.cache[key] = value
            self.order.append(key)
```

**Time**: O(1) average | **Space**: O(capacity)

**Related**: [LRU Cache](https://leetcode.com/problems/lru-cache/)

---

### Problem 5: First Unique Character

**Problem**: Find first non-repeating character.

**Solution**:
```python
def firstUniqChar(s):
    from collections import Counter
    
    count = Counter(s)
    for i, char in enumerate(s):
        if count[char] == 1:
            return i
    return -1
```

**Time**: O(n) | **Space**: O(1) - limited by charset

**Related**: [First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/)

---

### Problem 6: Contains Duplicate

**Problem**: Check if array contains duplicates.

**Solution**:
```python
def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

---

### Problem 7: Valid Anagram

**Problem**: Check if two strings are anagrams.

**Solution**:
```python
def isAnagram(s, t):
    from collections import Counter
    return Counter(s) == Counter(t)
```

**Time**: O(n) | **Space**: O(1) - limited by charset

**Related**: [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

---

### Problem 8: Design HashMap

**Problem**: Design hash map without using built-in hash table.

**Solution**:
```python
class MyHashMap:
    def __init__(self):
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]
    
    def _hash(self, key):
        return key % self.size
    
    def put(self, key, value):
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
    
    def get(self, key):
        bucket = self.buckets[self._hash(key)]
        for k, v in bucket:
            if k == key:
                return v
        return -1
    
    def remove(self, key):
        bucket = self.buckets[self._hash(key)]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return
```

**Time**: O(1) average | **Space**: O(n)

**Related**: [Design HashMap](https://leetcode.com/problems/design-hashmap/)

---

### Problem 9: Copy List with Random Pointer

**Problem**: Deep copy linked list with random pointers.

**Solution**:
```python
def copyRandomList(head):
    if not head:
        return None
    
    mapping = {}
    current = head
    
    # First pass: create nodes
    while current:
        mapping[current] = Node(current.val)
        current = current.next
    
    # Second pass: set pointers
    current = head
    while current:
        if current.next:
            mapping[current].next = mapping[current.next]
        if current.random:
            mapping[current].random = mapping[current.random]
        current = current.next
    
    return mapping[head]
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

---

### Problem 10: Longest Consecutive Sequence

**Problem**: Find length of longest consecutive sequence.

**Solution**:
```python
def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1
            
            longest = max(longest, current_streak)
    
    return longest
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

---

## Advanced Patterns

### 1. Subarray Sum Equals K

**Pattern**: Use prefix sum with hash map.

**Related Pattern**: See [Prefix Sum Pattern]({{< ref "../coding-patterns/22-Prefix_Sum.md" >}})

```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    sum_map = {0: 1}
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in sum_map:
            count += sum_map[prefix_sum - k]
        sum_map[prefix_sum] = sum_map.get(prefix_sum, 0) + 1
    
    return count
```

---

### 2. Design HashSet

**Pattern**: Implement set using hash table.

```python
class MyHashSet:
    def __init__(self):
        self.size = 1000
        self.buckets = [set() for _ in range(self.size)]
    
    def _hash(self, key):
        return key % self.size
    
    def add(self, key):
        self.buckets[self._hash(key)].add(key)
    
    def remove(self, key):
        self.buckets[self._hash(key)].discard(key)
    
    def contains(self, key):
        return key in self.buckets[self._hash(key)]
```

---

## Key Takeaways

- Hash tables provide O(1) average access time
- Use for fast lookups, frequency counting, caching
- Python dict, Counter, defaultdict are powerful
- Two sum pattern: complement lookup
- Grouping pattern: defaultdict for grouping
- Collision handling: chaining or open addressing
- Time complexity: O(1) average, O(n) worst case
- Space complexity: O(n) for storing elements
- Practice frequency counting and grouping problems

---

## Practice Problems

**Easy**:
- [Two Sum](https://leetcode.com/problems/two-sum/)
- [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

**Medium**:
- [Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [Longest Substring Without Repeating](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/)

**Hard**:
- [LRU Cache](https://leetcode.com/problems/lru-cache/)
- [LFU Cache](https://leetcode.com/problems/lfu-cache/)
- [Design HashMap](https://leetcode.com/problems/design-hashmap/)

