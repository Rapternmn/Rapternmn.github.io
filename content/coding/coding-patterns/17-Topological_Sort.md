+++
title = "Topological Sort"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 17
description = "Complete guide to Topological Sort pattern with templates in C++ and Python. Covers DFS-based and Kahn's algorithm approaches for ordering nodes based on dependencies, cycle detection, and scheduling problems with LeetCode problem references."
+++

---

## Introduction

Topological Sort is an ordering of nodes in a directed acyclic graph (DAG) such that for every directed edge (u, v), node u comes before node v. It's essential for solving dependency ordering problems.

This guide provides templates and patterns for Topological Sort with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Topological Sort

- **Dependency ordering**: Tasks with prerequisites
- **Course scheduling**: Courses with prerequisites
- **Build order**: Compiling files with dependencies
- **Event ordering**: Events with dependencies
- **Cycle detection**: Check if DAG has cycles

### Time & Space Complexity

- **DFS approach**: O(V + E) - visit each vertex and edge once
- **Kahn's algorithm**: O(V + E) - process each vertex and edge once
- **Space Complexity**: O(V) for recursion stack or queue + O(V) for visited/indegree

---

## Pattern Variations

### Variation 1: DFS-based Topological Sort

Use DFS with finishing time to order nodes.

**Use Cases**:
- Course Schedule
- Build Order
- General topological ordering

### Variation 2: Kahn's Algorithm (BFS-based)

Use BFS with in-degree tracking.

**Use Cases**:
- When need to detect cycles early
- Level-by-level processing
- Parallel task execution

---

## Template 1: DFS-based Topological Sort

**Key Points**:
- Use DFS to explore graph
- Mark nodes with colors: white (unvisited), gray (processing), black (processed)
- Push node to stack after all neighbors processed
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for recursion stack + O(V) for color array = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> topologicalSortDFS(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> result;
    vector<int> color(n, 0); // 0: white, 1: gray, 2: black
    stack<int> st;
    
    for (int i = 0; i < n; i++) {
        if (color[i] == 0) {
            if (!topologicalDFS(graph, i, color, st)) {
                return {}; // Cycle detected
            }
        }
    }
    
    while (!st.empty()) {
        result.push_back(st.top());
        st.pop();
    }
    
    return result;
}

bool topologicalDFS(vector<vector<int>>& graph, int node, 
                   vector<int>& color, stack<int>& st) {
    color[node] = 1; // Gray: being processed
    
    for (int neighbor : graph[node]) {
        if (color[neighbor] == 1) {
            return false; // Back edge found (cycle)
        }
        if (color[neighbor] == 0) {
            if (!topologicalDFS(graph, neighbor, color, st)) {
                return false;
            }
        }
    }
    
    color[node] = 2; // Black: processed
    st.push(node);
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def topological_sort_dfs(graph):
    n = len(graph)
    result = []
    color = [0] * n  # 0: white, 1: gray, 2: black
    stack = []
    
    for i in range(n):
        if color[i] == 0:
            if not topological_dfs(graph, i, color, stack):
                return []  # Cycle detected
    
    return stack[::-1]  # Reverse stack

def topological_dfs(graph, node, color, stack):
    color[node] = 1  # Gray: being processed
    
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return False  # Back edge found (cycle)
        if color[neighbor] == 0:
            if not topological_dfs(graph, neighbor, color, stack):
                return False
    
    color[node] = 2  # Black: processed
    stack.append(node)
    return True
```

</details>

### Related Problems

- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

---

## Template 2: Kahn's Algorithm (BFS-based)

**Key Points**:
- Calculate in-degree for each node
- Start with nodes having in-degree 0
- Remove edges, add nodes with new in-degree 0
- If result size != n, cycle exists
- **Time Complexity**: O(V + E) - process each vertex and edge once
- **Space Complexity**: O(V) for queue + O(V) for in-degree array = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> topologicalSortKahn(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> inDegree(n, 0);
    
    // Calculate in-degrees
    for (int i = 0; i < n; i++) {
        for (int neighbor : graph[i]) {
            inDegree[neighbor]++;
        }
    }
    
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    
    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : graph[node]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    // If result size != n, cycle exists
    if (result.size() != n) {
        return {};
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque

def topological_sort_kahn(graph):
    n = len(graph)
    in_degree = [0] * n
    
    # Calculate in-degrees
    for i in range(n):
        for neighbor in graph[i]:
            in_degree[neighbor] += 1
    
    q = deque([i for i in range(n) if in_degree[i] == 0])
    
    result = []
    while q:
        node = q.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)
    
    # If result size != n, cycle exists
    if len(result) != n:
        return []
    
    return result
```

</details>

### Related Problems

- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

---

## Template 3: Course Schedule (Cycle Detection)

**Key Points**:
- Check if graph has cycle
- If cycle exists, cannot complete all courses
- Use DFS with color coding or Kahn's algorithm
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for recursion stack or queue

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    
    // Build graph
    for (auto& edge : prerequisites) {
        graph[edge[1]].push_back(edge[0]);
    }
    
    vector<int> color(numCourses, 0); // 0: white, 1: gray, 2: black
    
    for (int i = 0; i < numCourses; i++) {
        if (color[i] == 0) {
            if (hasCycleDFS(graph, i, color)) {
                return false; // Cycle detected
            }
        }
    }
    
    return true;
}

bool hasCycleDFS(vector<vector<int>>& graph, int node, vector<int>& color) {
    color[node] = 1; // Gray
    
    for (int neighbor : graph[node]) {
        if (color[neighbor] == 1) {
            return true; // Back edge (cycle)
        }
        if (color[neighbor] == 0) {
            if (hasCycleDFS(graph, neighbor, color)) {
                return true;
            }
        }
    }
    
    color[node] = 2; // Black
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    
    # Build graph
    for edge in prerequisites:
        graph[edge[1]].append(edge[0])
    
    color = [0] * num_courses  # 0: white, 1: gray, 2: black
    
    for i in range(num_courses):
        if color[i] == 0:
            if has_cycle_dfs(graph, i, color):
                return False  # Cycle detected
    
    return True

def has_cycle_dfs(graph, node, color):
    color[node] = 1  # Gray
    
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return True  # Back edge (cycle)
        if color[neighbor] == 0:
            if has_cycle_dfs(graph, neighbor, color):
                return True
    
    color[node] = 2  # Black
    return False
```

</details>

### Related Problems

- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

---

## Template 4: Course Schedule II (Get Order)

**Key Points**:
- Return order of courses if possible
- Use topological sort to get order
- Return empty if cycle exists
- **Time Complexity**: O(V + E) - topological sort
- **Space Complexity**: O(V) for result and auxiliary structures

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> inDegree(numCourses, 0);
    
    // Build graph and calculate in-degrees
    for (auto& edge : prerequisites) {
        graph[edge[1]].push_back(edge[0]);
        inDegree[edge[0]]++;
    }
    
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    
    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        
        for (int neighbor : graph[node]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    // If result size != numCourses, cycle exists
    if (result.size() != numCourses) {
        return {};
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque

def find_order(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    
    # Build graph and calculate in-degrees
    for edge in prerequisites:
        graph[edge[1]].append(edge[0])
        in_degree[edge[0]] += 1
    
    q = deque([i for i in range(num_courses) if in_degree[i] == 0])
    
    result = []
    while q:
        node = q.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)
    
    # If result size != numCourses, cycle exists
    if len(result) != num_courses:
        return []
    
    return result
```

</details>

### Related Problems

- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Course Schedule](https://leetcode.com/problems/course-schedule/)

---

## Template 5: Alien Dictionary

**Key Points**:
- Build graph from word comparisons
- Find topological order of characters
- Handle edge cases (invalid ordering)
- **Time Complexity**: O(C) where C is total characters
- **Space Complexity**: O(1) - at most 26 characters

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
string alienOrder(vector<string>& words) {
    unordered_map<char, unordered_set<char>> graph;
    unordered_map<char, int> inDegree;
    
    // Initialize in-degree for all characters
    for (string& word : words) {
        for (char c : word) {
            inDegree[c] = 0;
        }
    }
    
    // Build graph
    for (int i = 0; i < words.size() - 1; i++) {
        string word1 = words[i];
        string word2 = words[i + 1];
        
        // Check invalid case: word1 is prefix of word2 but longer
        if (word1.length() > word2.length() && 
            word1.substr(0, word2.length()) == word2) {
            return "";
        }
        
        int minLen = min(word1.length(), word2.length());
        for (int j = 0; j < minLen; j++) {
            if (word1[j] != word2[j]) {
                if (graph[word1[j]].find(word2[j]) == graph[word1[j]].end()) {
                    graph[word1[j]].insert(word2[j]);
                    inDegree[word2[j]]++;
                }
                break;
            }
        }
    }
    
    // Kahn's algorithm
    queue<char> q;
    for (auto& [c, degree] : inDegree) {
        if (degree == 0) {
            q.push(c);
        }
    }
    
    string result;
    while (!q.empty()) {
        char c = q.front();
        q.pop();
        result += c;
        
        for (char neighbor : graph[c]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    // If result size != total characters, cycle exists
    if (result.length() != inDegree.size()) {
        return "";
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque, defaultdict

def alien_order(words):
    graph = defaultdict(set)
    in_degree = {}
    
    # Initialize in-degree for all characters
    for word in words:
        for c in word:
            in_degree[c] = 0
    
    # Build graph
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i + 1]
        
        # Check invalid case: word1 is prefix of word2 but longer
        if len(word1) > len(word2) and word1[:len(word2)] == word2:
            return ""
        
        min_len = min(len(word1), len(word2))
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Kahn's algorithm
    q = deque([c for c in in_degree if in_degree[c] == 0])
    
    result = []
    while q:
        c = q.popleft()
        result.append(c)
        
        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)
    
    # If result size != total characters, cycle exists
    if len(result) != len(in_degree):
        return ""
    
    return "".join(result)
```

</details>

### Related Problems

- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

---

## Key Takeaways

1. **DFS Approach**: Use DFS with color coding (white/gray/black) for cycle detection
2. **Kahn's Algorithm**: Use BFS with in-degree tracking, better for cycle detection
3. **Cycle Detection**: If result size != number of nodes, cycle exists
4. **Color Coding**: Gray = processing, Black = processed, White = unvisited
5. **Back Edge**: Edge to gray node indicates cycle
6. **Stack Order**: In DFS, push to stack after processing all neighbors

---

## Common Mistakes

1. **Not detecting cycles**: Forgetting to check for cycles
2. **Wrong graph direction**: Building graph with wrong edge direction
3. **Color coding errors**: Not properly updating colors in DFS
4. **In-degree calculation**: Wrong in-degree calculation in Kahn's algorithm
5. **Edge cases**: Not handling empty graph, single node, or disconnected components

---

## Practice Problems by Difficulty

### Medium
- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)

### Hard
- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

---

## References

* [LeetCode Graph Tag](https://leetcode.com/tag/graph/)
* [Topological Sorting (Wikipedia)](https://en.wikipedia.org/wiki/Topological_sorting)

