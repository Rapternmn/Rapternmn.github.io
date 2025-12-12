+++
title = "Strings"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 2
description = "Strings data structure: operations, pattern matching, and top interview problems. Covers string manipulation, KMP, sliding window, and more."
+++

---

## Introduction

Strings are sequences of characters and are fundamental in programming. Many problems involve string manipulation, pattern matching, and text processing.

---

## String Fundamentals

### What is a String?

**String**: A sequence of characters, typically immutable in many languages.

**Key Characteristics**:
- **Immutable**: Cannot be modified in place (in Python, Java)
- **Indexed Access**: O(1) access by index
- **Character Encoding**: ASCII, Unicode support
- **Length Property**: O(1) or O(n) depending on language

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(1) |
| Length | O(1) or O(n) |
| Concatenation | O(n + m) |
| Substring | O(n) |
| Search | O(n) naive, O(n + m) KMP |
| Comparison | O(n) |

---

## Common Patterns

### 1. Two Pointers

**Pattern**: Use two pointers for palindrome checks, reversing, etc.

**When to Use**:
- Palindrome problems
- Reversing strings
- Comparing strings
- Removing characters

**Template**:
```python
def two_pointers_string(s):
    left = 0
    right = len(s) - 1
    
    while left < right:
        # Process characters
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True
```

**Related Pattern**: See [Two Pointers Pattern]({{< ref "../2-Two_Pointers.md" >}})

---

### 2. Sliding Window

**Pattern**: Maintain window for substring problems.

**When to Use**:
- Longest substring without repeating characters
- Minimum window substring
- Substring with K distinct characters

**Template**:
```python
def sliding_window_string(s):
    left = 0
    char_count = {}
    max_len = 0
    
    for right in range(len(s)):
        char_count[s[right]] = char_count.get(s[right], 0) + 1
        
        # Shrink window
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

**Related Pattern**: See [Sliding Window Pattern]({{< ref "../3-Sliding_Window.md" >}})

---

### 3. Character Frequency Map

**Pattern**: Use hash map to count character frequencies.

**When to Use**:
- Anagrams
- Character counting
- Frequency analysis

**Template**:
```python
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Or using Counter
from collections import Counter
freq = Counter(s)
```

---

### 4. String Reversal

**Pattern**: Reverse string or parts of string.

**Methods**:
```python
# Method 1: Two pointers
def reverse_string(s):
    s = list(s)
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return ''.join(s)

# Method 2: Built-in
s[::-1]

# Method 3: Recursive
def reverse_recursive(s):
    if len(s) <= 1:
        return s
    return reverse_recursive(s[1:]) + s[0]
```

---

### 5. Pattern Matching (KMP Algorithm)

**Pattern**: Efficient substring search using prefix function.

**When to Use**:
- Substring search
- Pattern matching
- Repeated pattern detection

**Template**:
```python
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = build_lps(pattern)
    i = j = 0
    
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == len(pattern):
            return i - j
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return -1
```

---

## Top Problems

### Problem 1: Valid Palindrome

**Problem**: Check if string is palindrome (ignoring case and non-alphanumeric).

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isPalindrome(string s) {
    int left = 0;
    int right = s.length() - 1;
    
    while (left < right) {
        while (left < right && !isalnum(s[left])) {
            left++;
        }
        while (left < right && !isalnum(s[right])) {
            right--;
        }
        
        if (tolower(s[left]) != tolower(s[right])) {
            return false;
        }
        
        left++;
        right--;
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isPalindrome(s):
    left, right = 0, len(s) - 1
    
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

**Pattern**: Two Pointers

---

### Problem 2: Longest Substring Without Repeating Characters

**Problem**: Find length of longest substring without repeating characters.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> charMap;
    int left = 0;
    int maxLen = 0;
    
    for (int right = 0; right < s.length(); right++) {
        if (charMap.find(s[right]) != charMap.end() && 
            charMap[s[right]] >= left) {
            left = charMap[s[right]] + 1;
        }
        
        charMap[s[right]] = right;
        maxLen = max(maxLen, right - left + 1);
    }
    
    return maxLen;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

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

**Pattern**: Sliding Window

---

### Problem 3: Valid Anagram

**Problem**: Check if two strings are anagrams.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;
    
    unordered_map<char, int> count;
    for (char c : s) {
        count[c]++;
    }
    
    for (char c : t) {
        if (count.find(c) == count.end() || count[c] == 0) {
            return false;
        }
        count[c]--;
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isAnagram(s, t):
    if len(s) != len(t):
        return False
    
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    for char in t:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] == 0:
            del count[char]
    
    return len(count) == 0

# Alternative using Counter
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
```

</details>

**Time**: O(n) | **Space**: O(1) - limited by charset

**Related**: [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

---

### Problem 4: Group Anagrams

**Problem**: Group strings that are anagrams of each other.

**Solution**:
```python
def groupAnagrams(strs):
    from collections import defaultdict
    
    anagram_map = defaultdict(list)
    
    for s in strs:
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())
```

**Time**: O(n * k log k) where k is max string length | **Space**: O(n * k)

**Related**: [Group Anagrams](https://leetcode.com/problems/group-anagrams/)

---

### Problem 5: Longest Palindromic Substring

**Problem**: Find longest palindromic substring.

**Solution** (Expand Around Centers):
```python
def longestPalindrome(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        # Odd length palindromes
        palindrome1 = expand_around_center(i, i)
        # Even length palindromes
        palindrome2 = expand_around_center(i, i + 1)
        
        longest = max(longest, palindrome1, palindrome2, key=len)
    
    return longest
```

**Time**: O(n²) | **Space**: O(1)

**Related**: [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

---

### Problem 6: Minimum Window Substring

**Problem**: Find minimum window in s containing all characters of t.

**Solution**:
```python
def minWindow(s, t):
    from collections import Counter
    
    need = Counter(t)
    missing = len(t)
    left = start = end = 0
    
    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        
        if missing == 0:
            while left < right and need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            
            if end == 0 or right - left < end - start:
                start, end = left, right
    
    return s[start:end]
```

**Time**: O(|s| + |t|) | **Space**: O(|s| + |t|)

**Related**: [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

**Pattern**: Sliding Window

---

### Problem 7: Valid Parentheses

**Problem**: Check if string has valid parentheses.

**Solution**:
```python
def isValid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    
    return len(stack) == 0
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)

**Pattern**: Stack

---

### Problem 8: Reverse Words in String

**Problem**: Reverse order of words in string.

**Solution**:
```python
def reverseWords(s):
    words = s.split()
    return ' '.join(reversed(words))

# Without built-in split
def reverseWords(s):
    result = []
    word = []
    
    for char in s:
        if char == ' ':
            if word:
                result.append(''.join(word))
                word = []
        else:
            word.append(char)
    
    if word:
        result.append(''.join(word))
    
    return ' '.join(reversed(result))
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/)

---

### Problem 9: String to Integer (atoi)

**Problem**: Convert string to integer (handle edge cases).

**Solution**:
```python
def myAtoi(s):
    s = s.strip()
    if not s:
        return 0
    
    sign = 1
    if s[0] == '-':
        sign = -1
        s = s[1:]
    elif s[0] == '+':
        s = s[1:]
    
    result = 0
    for char in s:
        if not char.isdigit():
            break
        result = result * 10 + int(char)
        
        if sign * result > 2**31 - 1:
            return 2**31 - 1
        if sign * result < -2**31:
            return -2**31
    
    return sign * result
```

**Time**: O(n) | **Space**: O(1)

**Related**: [String to Integer (atoi)](https://leetcode.com/problems/string-to-integer-atoi/)

---

### Problem 10: Longest Common Prefix

**Problem**: Find longest common prefix among array of strings.

**Solution**:
```python
def longestCommonPrefix(strs):
    if not strs:
        return ""
    
    prefix = strs[0]
    for i in range(1, len(strs)):
        while not strs[i].startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix
```

**Time**: O(S) where S is sum of all characters | **Space**: O(1)

**Related**: [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

---

## Advanced Patterns

### 1. Rabin-Karp Algorithm

**Pattern**: Rolling hash for substring search.

```python
def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    if m > n:
        return -1
    
    base = 256
    mod = 10**9 + 7
    
    # Calculate hash of pattern and first window
    pattern_hash = 0
    window_hash = 0
    h = pow(base, m - 1, mod)
    
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        window_hash = (window_hash * base + ord(text[i])) % mod
    
    for i in range(n - m + 1):
        if pattern_hash == window_hash:
            if text[i:i + m] == pattern:
                return i
        
        if i < n - m:
            window_hash = (base * (window_hash - ord(text[i]) * h) + ord(text[i + m])) % mod
            window_hash = (window_hash + mod) % mod
    
    return -1
```

---

### 2. Z-Algorithm

**Pattern**: Find all occurrences of pattern in text.

```python
def z_algorithm(s):
    n = len(s)
    z = [0] * n
    left = right = 0
    
    for i in range(1, n):
        if i <= right:
            z[i] = min(right - i + 1, z[i - left])
        
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        
        if i + z[i] - 1 > right:
            left = i
            right = i + z[i] - 1
    
    return z
```

---

## Key Takeaways

- Strings are immutable in many languages - use list for modifications
- Two pointers for palindrome and reversal problems
- Sliding window for substring problems
- Character frequency maps for anagram problems
- KMP algorithm for efficient pattern matching
- Stack for parentheses and nested structures
- Practice string manipulation operations
- Handle edge cases (empty, single char, special chars)

---

## Practice Problems

**Easy**:
- [Reverse String](https://leetcode.com/problems/reverse-string/)
- [First Unique Character](https://leetcode.com/problems/first-unique-character-in-a-string/)
- [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

**Medium**:
- [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

**Hard**:
- [Edit Distance](https://leetcode.com/problems/edit-distance/)
- [Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)
- [Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)

