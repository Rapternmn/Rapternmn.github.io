+++
title = "Sets"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 10
description = "Sets data structure: set operations, implementations, and top interview problems. Covers set theory, union, intersection, and more."
+++

---

## Introduction

Sets are collections of unique elements with no duplicates. They provide fast membership testing and set operations like union, intersection, and difference.

---

## Set Fundamentals

### What is a Set?

**Set**: A collection of unique elements with no duplicates.

**Key Characteristics**:
- **Unique Elements**: No duplicates allowed
- **Unordered**: No specific order (in most implementations)
- **Fast Lookup**: O(1) average membership testing
- **Set Operations**: Union, intersection, difference

### Operations

| Operation | Time Complexity |
|-----------|----------------|
| Add | O(1) average |
| Remove | O(1) average |
| Contains | O(1) average |
| Union | O(n + m) |
| Intersection | O(min(n, m)) |
| Difference | O(n) |

---

## Implementation

### Using set (Python)

```python
# Create set
my_set = {1, 2, 3}
my_set = set([1, 2, 3])

# Add element
my_set.add(4)

# Remove element
my_set.remove(3)  # Raises KeyError if not found
my_set.discard(3)  # No error if not found

# Check membership
if 2 in my_set:
    print("Found")

# Set operations
set1 = {1, 2, 3}
set2 = {3, 4, 5}
union = set1 | set2  # {1, 2, 3, 4, 5}
intersection = set1 & set2  # {3}
difference = set1 - set2  # {1, 2}
symmetric_diff = set1 ^ set2  # {1, 2, 4, 5}
```

---

## Common Patterns

### 1. Duplicate Detection

**Pattern**: Use set to track seen elements.

**Template**:
```python
def has_duplicates(arr):
    seen = set()
    for item in arr:
        if item in seen:
            return True
        seen.add(item)
    return False
```

---

### 2. Unique Elements

**Pattern**: Remove duplicates using set.

**Template**:
```python
def unique_elements(arr):
    return list(set(arr))
```

---

### 3. Set Operations

**Pattern**: Use set operations for comparisons.

**Template**:
```python
def common_elements(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1 & set2)  # Intersection
```

---

## Top Problems

### Problem 1: Contains Duplicate

**Problem**: Check if array contains duplicates.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> seen;
    
    for (int num : nums) {
        if (seen.find(num) != seen.end()) {
            return true;
        }
        seen.insert(num);
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

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

### Problem 2: Intersection of Two Arrays

**Problem**: Find intersection of two arrays.

**Solution**:
```python
def intersection(nums1, nums2):
    set1 = set(nums1)
    set2 = set(nums2)
    return list(set1 & set2)
```

**Time**: O(n + m) | **Space**: O(n + m)

**Related**: [Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)

---

### Problem 3: Happy Number

**Problem**: Check if number is happy (sum of squares of digits eventually equals 1).

**Solution**:
```python
def isHappy(n):
    seen = set()
    
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit)**2 for digit in str(n))
    
    return n == 1
```

**Time**: O(log n) | **Space**: O(log n)

**Related**: [Happy Number](https://leetcode.com/problems/happy-number/)

---

### Problem 4: Longest Consecutive Sequence

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

### Problem 5: Single Number

**Problem**: Find single number that appears once (others appear twice).

**Solution**:
```python
def singleNumber(nums):
    seen = set()
    for num in nums:
        if num in seen:
            seen.remove(num)
        else:
            seen.add(num)
    return seen.pop()

# XOR solution (better)
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
```

**Time**: O(n) | **Space**: O(n) for set, O(1) for XOR

**Related**: [Single Number](https://leetcode.com/problems/single-number/)

---

### Problem 6: Jewels and Stones

**Problem**: Count how many stones are jewels.

**Solution**:
```python
def numJewelsInStones(jewels, stones):
    jewel_set = set(jewels)
    return sum(1 for stone in stones if stone in jewel_set)
```

**Time**: O(n + m) | **Space**: O(m) where m is jewels length

**Related**: [Jewels and Stones](https://leetcode.com/problems/jewels-and-stones/)

---

### Problem 7: Unique Email Addresses

**Problem**: Count unique email addresses after processing.

**Solution**:
```python
def numUniqueEmails(emails):
    unique = set()
    
    for email in emails:
        local, domain = email.split('@')
        local = local.split('+')[0].replace('.', '')
        unique.add(local + '@' + domain)
    
    return len(unique)
```

**Time**: O(n * m) where m is email length | **Space**: O(n)

**Related**: [Unique Email Addresses](https://leetcode.com/problems/unique-email-addresses/)

---

### Problem 8: Find the Difference

**Problem**: Find character added to string t.

**Solution**:
```python
def findTheDifference(s, t):
    from collections import Counter
    return list((Counter(t) - Counter(s)).keys())[0]

# Using set
def findTheDifference(s, t):
    for char in set(t):
        if t.count(char) != s.count(char):
            return char
```

**Time**: O(n) | **Space**: O(1) - limited by charset

**Related**: [Find the Difference](https://leetcode.com/problems/find-the-difference/)

---

### Problem 9: Design HashSet

**Problem**: Design hash set without using built-in set.

**Solution**:
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

**Time**: O(1) average | **Space**: O(n)

**Related**: [Design HashSet](https://leetcode.com/problems/design-hashset/)

---

### Problem 10: Intersection of Two Arrays II

**Problem**: Find intersection with frequency consideration.

**Solution**:
```python
def intersect(nums1, nums2):
    from collections import Counter
    
    count1 = Counter(nums1)
    result = []
    
    for num in nums2:
        if count1[num] > 0:
            result.append(num)
            count1[num] -= 1
    
    return result
```

**Time**: O(n + m) | **Space**: O(min(n, m))

**Related**: [Intersection of Two Arrays II](https://leetcode.com/problems/intersection-of-two-arrays-ii/)

---

## Advanced Patterns

### 1. Set Union and Intersection

**Pattern**: Combine or find common elements.

```python
def union_intersection(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    
    union = set1 | set2
    intersection = set1 & set2
    
    return union, intersection
```

---

### 2. Symmetric Difference

**Pattern**: Elements in either set but not both.

```python
def symmetric_difference(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    return list(set1 ^ set2)
```

---

## Key Takeaways

- Sets store unique elements with no duplicates
- Fast O(1) average membership testing
- Useful for duplicate detection and tracking seen elements
- Set operations: union, intersection, difference, symmetric difference
- Python set is implemented as hash table
- Time complexity: O(1) for basic operations
- Space complexity: O(n) for storing elements
- Practice problems involving uniqueness and set operations

---

## Practice Problems

**Easy**:
- [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)
- [Intersection of Two Arrays](https://leetcode.com/problems/intersection-of-two-arrays/)
- [Happy Number](https://leetcode.com/problems/happy-number/)

**Medium**:
- [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)
- [Single Number](https://leetcode.com/problems/single-number/)
- [Unique Email Addresses](https://leetcode.com/problems/unique-email-addresses/)

**Hard**:
- [Design HashSet](https://leetcode.com/problems/design-hashset/)
- [Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)
- [First Missing Positive](https://leetcode.com/problems/first-missing-positive/)

