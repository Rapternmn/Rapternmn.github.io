+++
title = "Trie"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 19
description = "Complete guide to Trie (Prefix Tree) data structure with templates in C++ and Python. Covers insertion, search, prefix matching, word search, and autocomplete problems with LeetCode problem references."
+++

---

## Introduction

Trie (Prefix Tree) is a tree-like data structure that stores strings efficiently. It's particularly useful for prefix matching, autocomplete, and dictionary problems. Each node represents a character, and paths from root to nodes represent strings.

This guide provides templates and patterns for Trie with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Trie

- **Prefix matching**: Check if string has prefix
- **Autocomplete**: Find all strings with given prefix
- **Dictionary problems**: Word search, word validation
- **String search**: Efficient string lookup
- **IP routing**: Longest prefix matching

### Time & Space Complexity

- **Insert**: O(m) where m is string length
- **Search**: O(m) - traverse path of length m
- **Prefix Search**: O(m) - same as search
- **Space Complexity**: O(ALPHABET_SIZE × N × M) where N is number of strings, M is average length

---

## Pattern Variations

### Variation 1: Basic Trie

Standard trie with insert, search, and startsWith.

**Use Cases**:
- Basic dictionary operations
- Prefix matching

### Variation 2: Word Search Trie

Trie for searching words in 2D grid.

**Use Cases**:
- Word Search II
- Boggle game

### Variation 3: Autocomplete Trie

Trie with frequency tracking for autocomplete.

**Use Cases**:
- Autocomplete systems
- Search suggestions

---

## Template 1: Basic Trie Implementation

**Key Points**:
- Each node has children array (26 for lowercase letters)
- isEnd flag marks end of word
- Insert: traverse/create path, mark end
- Search: traverse path, check isEnd
- **Time Complexity**: O(m) per operation where m is string length
- **Space Complexity**: O(ALPHABET_SIZE × N × M) worst case

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class TrieNode {
public:
    vector<TrieNode*> children;
    bool isEnd;
    
    TrieNode() {
        children = vector<TrieNode*>(26, nullptr);
        isEnd = false;
    }
};

class Trie {
private:
    TrieNode* root;
    
public:
    Trie() {
        root = new TrieNode();
    }
    
    void insert(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
        }
        curr->isEnd = true;
    }
    
    bool search(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                return false;
            }
            curr = curr->children[index];
        }
        return curr->isEnd;
    }
    
    bool startsWith(string prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                return false;
            }
            curr = curr->children[index];
        }
        return true;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True
    
    def search(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.is_end
    
    def starts_with(self, prefix):
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True
```

</details>

### Related Problems

- [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
- [Add and Search Word](https://leetcode.com/problems/add-and-search-word-data-structure-design/)

---

## Template 2: Word Search II (Trie + DFS)

**Key Points**:
- Build trie from words
- DFS on board, check if path forms word in trie
- Mark visited cells, backtrack
- **Time Complexity**: O(M × 4^L) where M is board cells, L is max word length
- **Space Complexity**: O(N × M) for trie where N is words, M is average length

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class TrieNode {
public:
    vector<TrieNode*> children;
    string word;
    
    TrieNode() {
        children = vector<TrieNode*>(26, nullptr);
        word = "";
    }
};

class Solution {
private:
    TrieNode* root;
    vector<string> result;
    
    void buildTrie(vector<string>& words) {
        root = new TrieNode();
        for (string& word : words) {
            TrieNode* curr = root;
            for (char c : word) {
                int index = c - 'a';
                if (curr->children[index] == nullptr) {
                    curr->children[index] = new TrieNode();
                }
                curr = curr->children[index];
            }
            curr->word = word; // Store word at end node
        }
    }
    
    void dfs(vector<vector<char>>& board, int i, int j, TrieNode* node) {
        char c = board[i][j];
        int index = c - 'a';
        
        // Check if current path exists in trie
        if (node->children[index] == nullptr) {
            return;
        }
        
        node = node->children[index];
        
        // If word found, add to result
        if (node->word != "") {
            result.push_back(node->word);
            node->word = ""; // Avoid duplicates
        }
        
        // Mark as visited
        board[i][j] = '#';
        
        // Explore neighbors
        int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        for (auto& dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];
            if (ni >= 0 && ni < board.size() && 
                nj >= 0 && nj < board[0].size() && 
                board[ni][nj] != '#') {
                dfs(board, ni, nj, node);
            }
        }
        
        // Backtrack
        board[i][j] = c;
    }
    
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        buildTrie(words);
        
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                dfs(board, i, j, root);
            }
        }
        
        return result;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = ""

class Solution:
    def __init__(self):
        self.root = TrieNode()
        self.result = []
    
    def build_trie(self, words):
        for word in words:
            curr = self.root
            for c in word:
                if c not in curr.children:
                    curr.children[c] = TrieNode()
                curr = curr.children[c]
            curr.word = word  # Store word at end node
    
    def dfs(self, board, i, j, node):
        c = board[i][j]
        if c not in node.children:
            return
        
        node = node.children[c]
        
        # If word found, add to result
        if node.word:
            self.result.append(node.word)
            node.word = ""  # Avoid duplicates
        
        # Mark as visited
        board[i][j] = '#'
        
        # Explore neighbors
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if (0 <= ni < len(board) and 0 <= nj < len(board[0]) and 
                board[ni][nj] != '#'):
                self.dfs(board, ni, nj, node)
        
        # Backtrack
        board[i][j] = c
    
    def find_words(self, board, words):
        self.build_trie(words)
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, i, j, self.root)
        
        return self.result
```

</details>

### Related Problems

- [Word Search II](https://leetcode.com/problems/word-search-ii/)
- [Word Search](https://leetcode.com/problems/word-search/)

---

## Template 3: Autocomplete with Frequency

**Key Points**:
- Store frequency at each node
- When searching prefix, find all words with that prefix
- Sort by frequency, return top k
- **Time Complexity**: O(m + k log k) where m is prefix length, k is results
- **Space Complexity**: O(N × M) for trie

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class TrieNode {
public:
    vector<TrieNode*> children;
    unordered_map<string, int> frequencies; // word -> frequency
    bool isEnd;
    
    TrieNode() {
        children = vector<TrieNode*>(26, nullptr);
        isEnd = false;
    }
};

class AutocompleteSystem {
private:
    TrieNode* root;
    string currentQuery;
    
    void insert(string sentence, int times) {
        TrieNode* curr = root;
        for (char c : sentence) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
            curr->frequencies[sentence] += times;
        }
        curr->isEnd = true;
    }
    
    vector<string> search(string prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                return {};
            }
            curr = curr->children[index];
        }
        
        // Get all words with this prefix
        vector<pair<string, int>> candidates;
        for (auto& [word, freq] : curr->frequencies) {
            candidates.push_back({word, freq});
        }
        
        // Sort by frequency (descending), then lexicographically
        sort(candidates.begin(), candidates.end(),
             [](const pair<string, int>& a, const pair<string, int>& b) {
                 if (a.second != b.second) {
                     return a.second > b.second;
                 }
                 return a.first < b.first;
             });
        
        vector<string> result;
        for (int i = 0; i < min(3, (int)candidates.size()); i++) {
            result.push_back(candidates[i].first);
        }
        
        return result;
    }
    
public:
    AutocompleteSystem(vector<string>& sentences, vector<int>& times) {
        root = new TrieNode();
        for (int i = 0; i < sentences.size(); i++) {
            insert(sentences[i], times[i]);
        }
    }
    
    vector<string> input(char c) {
        if (c == '#') {
            insert(currentQuery, 1);
            currentQuery = "";
            return {};
        }
        
        currentQuery += c;
        return search(currentQuery);
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.frequencies = {}  # word -> frequency
        self.is_end = False

class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.current_query = ""
        
        for sentence, time in zip(sentences, times):
            self.insert(sentence, time)
    
    def insert(self, sentence, times):
        curr = self.root
        for c in sentence:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
            curr.frequencies[sentence] = curr.frequencies.get(sentence, 0) + times
        curr.is_end = True
    
    def search(self, prefix):
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return []
            curr = curr.children[c]
        
        # Get all words with this prefix
        candidates = [(word, freq) for word, freq in curr.frequencies.items()]
        
        # Sort by frequency (descending), then lexicographically
        candidates.sort(key=lambda x: (-x[1], x[0]))
        
        return [word for word, _ in candidates[:3]]
    
    def input(self, c):
        if c == '#':
            self.insert(self.current_query, 1)
            self.current_query = ""
            return []
        
        self.current_query += c
        return self.search(self.current_query)
```

</details>

### Related Problems

- [Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/)

---

## Template 4: Add and Search Word (Wildcard)

**Key Points**:
- Support '.' wildcard that matches any character
- Use DFS for wildcard matching
- **Time Complexity**: O(m) for exact match, O(26^m) worst case for wildcard
- **Space Complexity**: O(N × M) for trie

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class WordDictionary {
private:
    class TrieNode {
    public:
        vector<TrieNode*> children;
        bool isEnd;
        
        TrieNode() {
            children = vector<TrieNode*>(26, nullptr);
            isEnd = false;
        }
    };
    
    TrieNode* root;
    
    bool searchHelper(string word, int index, TrieNode* node) {
        if (index == word.length()) {
            return node->isEnd;
        }
        
        char c = word[index];
        if (c == '.') {
            // Try all possible characters
            for (int i = 0; i < 26; i++) {
                if (node->children[i] != nullptr) {
                    if (searchHelper(word, index + 1, node->children[i])) {
                        return true;
                    }
                }
            }
            return false;
        } else {
            int idx = c - 'a';
            if (node->children[idx] == nullptr) {
                return false;
            }
            return searchHelper(word, index + 1, node->children[idx]);
        }
    }
    
public:
    WordDictionary() {
        root = new TrieNode();
    }
    
    void addWord(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
        }
        curr->isEnd = true;
    }
    
    bool search(string word) {
        return searchHelper(word, 0, root);
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class WordDictionary:
    def __init__(self):
        self.root = TrieNode()
    
    def add_word(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True
    
    def search_helper(self, word, index, node):
        if index == len(word):
            return node.is_end
        
        c = word[index]
        if c == '.':
            # Try all possible characters
            for child in node.children.values():
                if self.search_helper(word, index + 1, child):
                    return True
            return False
        else:
            if c not in node.children:
                return False
            return self.search_helper(word, index + 1, node.children[c])
    
    def search(self, word):
        return self.search_helper(word, 0, self.root)
```

</details>

### Related Problems

- [Add and Search Word](https://leetcode.com/problems/add-and-search-word-data-structure-design/)
- [Word Search II](https://leetcode.com/problems/word-search-ii/)

---

## Template 5: Longest Common Prefix

**Key Points**:
- Insert all strings into trie
- Find longest path where all nodes have single child and isEnd is false
- **Time Complexity**: O(S) where S is sum of all string lengths
- **Space Complexity**: O(S) for trie

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
string longestCommonPrefix(vector<string>& strs) {
    if (strs.empty()) return "";
    
    TrieNode* root = new TrieNode();
    
    // Insert all strings
    for (string& str : strs) {
        TrieNode* curr = root;
        for (char c : str) {
            int index = c - 'a';
            if (curr->children[index] == nullptr) {
                curr->children[index] = new TrieNode();
            }
            curr = curr->children[index];
        }
        curr->isEnd = true;
    }
    
    // Find longest common prefix
    string result = "";
    TrieNode* curr = root;
    while (true) {
        int childCount = 0;
        int childIndex = -1;
        
        for (int i = 0; i < 26; i++) {
            if (curr->children[i] != nullptr) {
                childCount++;
                childIndex = i;
            }
        }
        
        if (childCount != 1 || curr->isEnd) {
            break;
        }
        
        result += ('a' + childIndex);
        curr = curr->children[childIndex];
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def longest_common_prefix(strs):
    if not strs:
        return ""
    
    root = TrieNode()
    
    # Insert all strings
    for s in strs:
        curr = root
        for c in s:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.is_end = True
    
    # Find longest common prefix
    result = []
    curr = root
    while True:
        children = [c for c in curr.children.keys()]
        if len(children) != 1 or curr.is_end:
            break
        
        result.append(children[0])
        curr = curr.children[children[0]]
    
    return "".join(result)
```

</details>

### Related Problems

- [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)
- [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)

---

## Key Takeaways

1. **Node Structure**: Each node has children array/map and isEnd flag
2. **Insert**: Create path for each character, mark end at last character
3. **Search**: Traverse path, check isEnd at end
4. **Prefix Search**: Similar to search but don't need isEnd
5. **Wildcard Matching**: Use DFS to try all possibilities for '.'
6. **Space Optimization**: Use map instead of array if alphabet is large or sparse

---

## Common Mistakes

1. **Not marking isEnd**: Forgetting to set isEnd flag after insertion
2. **Index errors**: Wrong character to index conversion (case sensitivity)
3. **Null checks**: Not checking if child exists before accessing
4. **Backtracking**: Not restoring board state in Word Search II
5. **Memory leaks**: Not deallocating trie nodes (C++)

---

## Practice Problems by Difficulty

### Medium
- [Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
- [Add and Search Word](https://leetcode.com/problems/add-and-search-word-data-structure-design/)
- [Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)

### Hard
- [Word Search II](https://leetcode.com/problems/word-search-ii/)
- [Design Search Autocomplete System](https://leetcode.com/problems/design-search-autocomplete-system/)

---

## References

* [LeetCode Trie Tag](https://leetcode.com/tag/trie/)
* [Trie (Wikipedia)](https://en.wikipedia.org/wiki/Trie)

