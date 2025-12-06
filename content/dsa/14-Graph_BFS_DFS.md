+++
title = "Graph BFS/DFS"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 14
description = "Complete guide to Graph BFS (Breadth-First Search) and DFS (Depth-First Search) patterns with templates in C++ and Python. Covers traversal, shortest path (BFS and Dijkstra's), cycle detection, topological sort, bipartite graph check, and graph algorithms with LeetCode problem references."
+++

---

## Introduction

Graph BFS and DFS are fundamental algorithms for traversing and exploring graphs. BFS explores level by level, making it ideal for shortest path problems, while DFS explores as deep as possible, useful for path finding and cycle detection.

This guide provides templates and patterns for Graph BFS and DFS with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use BFS vs DFS

**Use BFS when**:
- Finding shortest path in unweighted graph
- Level-order traversal needed
- Finding nodes at specific distance
- Minimum steps/problems
- Bipartite graph checking

**Use DFS when**:
- Exploring all paths
- Cycle detection
- Topological sorting
- Connected components
- Path existence problems

### Time & Space Complexity

- **BFS Time**: O(V + E) where V is vertices, E is edges
- **BFS Space**: O(V) for queue
- **DFS Time**: O(V + E) - visit each vertex and edge once
- **DFS Space**: O(V) for recursion stack or visited set

---

## Pattern Variations

### Variation 1: BFS (Level-Order Traversal)

Traverse graph level by level using queue.

**Use Cases**:
- Level Order Traversal
- Shortest Path
- Word Ladder

### Variation 2: DFS (Recursive)

Recursive depth-first traversal.

**Use Cases**:
- Path finding
- Cycle detection
- Connected components

### Variation 3: DFS (Iterative)

Iterative depth-first using stack.

**Use Cases**:
- When recursion depth is concern
- Explicit stack control

### Variation 4: Topological Sort

Order nodes based on dependencies.

**Use Cases**:
- Course Schedule
- Build Order
- Task Scheduling

### Variation 5: Bipartite Graph Check

Check if graph can be colored with two colors.

**Use Cases**:
- Is Graph Bipartite?
- Possible Bipartition

### Variation 6: Dijkstra's Algorithm

Find shortest path in weighted graphs with non-negative weights.

**Use Cases**:
- Network Delay Time
- Shortest path in weighted graphs

---

## Template 1: BFS - Level Order Traversal

**Key Points**:
- Use queue to store nodes
- Mark nodes as visited
- Process nodes level by level
- Track distance/level if needed
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) - queue and visited set

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
#include <vector>
#include <queue>
#include <unordered_set>

vector<vector<int>> bfsLevelOrder(vector<vector<int>>& graph, int start) {
    vector<vector<int>> result;
    queue<int> q;
    unordered_set<int> visited;
    
    q.push(start);
    visited.insert(start);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        for (int i = 0; i < levelSize; i++) {
            int node = q.front();
            q.pop();
            currentLevel.push_back(node);
            
            // Add neighbors to queue
            for (int neighbor : graph[node]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
        
        result.push_back(currentLevel);
    }
    
    return result;
}

// BFS with distance tracking
vector<int> bfsShortestPath(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<int> distance(n, -1);
    queue<int> q;
    
    q.push(start);
    distance[start] = 0;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int neighbor : graph[node]) {
            if (distance[neighbor] == -1) {
                distance[neighbor] = distance[node] + 1;
                q.push(neighbor);
            }
        }
    }
    
    return distance;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque

def bfs_level_order(graph, start):
    result = []
    q = deque([start])
    visited = {start}
    
    while q:
        level_size = len(q)
        current_level = []
        
        for i in range(level_size):
            node = q.popleft()
            current_level.append(node)
            
            # Add neighbors to queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        
        result.append(current_level)
    
    return result

# BFS with distance tracking
def bfs_shortest_path(graph, start):
    n = len(graph)
    distance = [-1] * n
    q = deque([start])
    distance[start] = 0
    
    while q:
        node = q.popleft()
        
        for neighbor in graph[node]:
            if distance[neighbor] == -1:
                distance[neighbor] = distance[node] + 1
                q.append(neighbor)
    
    return distance
```

</details>

### Related Problems

- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Word Ladder](https://leetcode.com/problems/word-ladder/)
- [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

---

## Template 2: DFS - Recursive

**Key Points**:
- Mark node as visited
- Process node
- Recursively visit all unvisited neighbors
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for recursion stack + O(V) for visited = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
void dfsRecursive(vector<vector<int>>& graph, int node, 
                  unordered_set<int>& visited, vector<int>& result) {
    // Mark as visited
    visited.insert(node);
    result.push_back(node);
    
    // Visit all neighbors
    for (int neighbor : graph[node]) {
        if (visited.find(neighbor) == visited.end()) {
            dfsRecursive(graph, neighbor, visited, result);
        }
    }
}

vector<int> dfsTraversal(vector<vector<int>>& graph, int start) {
    vector<int> result;
    unordered_set<int> visited;
    dfsRecursive(graph, start, visited, result);
    return result;
}

// DFS for cycle detection in directed graph
bool hasCycleDFS(vector<vector<int>>& graph, int node, 
                 vector<int>& color) {
    color[node] = 1; // Gray: being processed
    
    for (int neighbor : graph[node]) {
        if (color[neighbor] == 1) {
            return true; // Back edge found (cycle)
        }
        if (color[neighbor] == 0 && hasCycleDFS(graph, neighbor, color)) {
            return true;
        }
    }
    
    color[node] = 2; // Black: processed
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def dfs_recursive(graph, node, visited, result):
    # Mark as visited
    visited.add(node)
    result.append(node)
    
    # Visit all neighbors
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, result)

def dfs_traversal(graph, start):
    result = []
    visited = set()
    dfs_recursive(graph, start, visited, result)
    return result

# DFS for cycle detection in directed graph
def has_cycle_dfs(graph, node, color):
    color[node] = 1  # Gray: being processed
    
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return True  # Back edge found (cycle)
        if color[neighbor] == 0 and has_cycle_dfs(graph, neighbor, color):
            return True
    
    color[node] = 2  # Black: processed
    return False
```

</details>

### Related Problems

- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Course Schedule](https://leetcode.com/problems/course-schedule/)

---

## Template 3: DFS - Iterative

**Key Points**:
- Use stack instead of recursion
- Push node, mark visited, process, push neighbors
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for stack + O(V) for visited = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> dfsIterative(vector<vector<int>>& graph, int start) {
    vector<int> result;
    stack<int> st;
    unordered_set<int> visited;
    
    st.push(start);
    visited.insert(start);
    
    while (!st.empty()) {
        int node = st.top();
        st.pop();
        result.push_back(node);
        
        // Add neighbors to stack (reverse order for same traversal)
        for (int i = graph[node].size() - 1; i >= 0; i--) {
            int neighbor = graph[node][i];
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                st.push(neighbor);
            }
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def dfs_iterative(graph, start):
    result = []
    stack = [start]
    visited = {start}
    
    while stack:
        node = stack.pop()
        result.append(node)
        
        # Add neighbors to stack (reverse order for same traversal)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    
    return result
```

</details>

### Related Problems

- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)

---

## Template 4: Topological Sort

**Key Points**:
- Use DFS with finishing time
- Or use Kahn's algorithm (BFS-based)
- Process nodes after all dependencies
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for stack/queue + O(V) for visited = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// DFS-based Topological Sort
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
    color[node] = 1; // Gray
    
    for (int neighbor : graph[node]) {
        if (color[neighbor] == 1) {
            return false; // Cycle detected
        }
        if (color[neighbor] == 0) {
            if (!topologicalDFS(graph, neighbor, color, st)) {
                return false;
            }
        }
    }
    
    color[node] = 2; // Black
    st.push(node);
    return true;
}

// Kahn's Algorithm (BFS-based)
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
# DFS-based Topological Sort
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
    color[node] = 1  # Gray
    
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return False  # Cycle detected
        if color[neighbor] == 0:
            if not topological_dfs(graph, neighbor, color, stack):
                return False
    
    color[node] = 2  # Black
    stack.append(node)
    return True

# Kahn's Algorithm (BFS-based)
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

## Template 5: Connected Components / Union-Find

**Key Points**:
- Find all connected components in graph
- Use DFS/BFS to explore each component
- Or use Union-Find data structure
- **Time Complexity**: O(V + E) for DFS/BFS, O(V × α(V)) for Union-Find
- **Space Complexity**: O(V) for visited/union-find

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Find number of connected components using DFS
int numConnectedComponents(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<bool> visited(n, false);
    int count = 0;
    
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfsComponent(graph, i, visited);
            count++;
        }
    }
    
    return count;
}

void dfsComponent(vector<vector<int>>& graph, int node, 
                  vector<bool>& visited) {
    visited[node] = true;
    
    for (int neighbor : graph[node]) {
        if (!visited[neighbor]) {
            dfsComponent(graph, neighbor, visited);
        }
    }
}

// Union-Find for connected components
class UnionFind {
private:
    vector<int> parent;
    vector<int> rank;
    int count;
    
public:
    UnionFind(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        count = n;
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX != rootY) {
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            count--;
        }
    }
    
    int getCount() {
        return count;
    }
};
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Find number of connected components using DFS
def num_connected_components(graph):
    n = len(graph)
    visited = [False] * n
    count = 0
    
    for i in range(n):
        if not visited[i]:
            dfs_component(graph, i, visited)
            count += 1
    
    return count

def dfs_component(graph, node, visited):
    visited[node] = True
    
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs_component(graph, neighbor, visited)

# Union-Find for connected components
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def unite(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            self.count -= 1
    
    def get_count(self):
        return self.count
```

</details>

### Related Problems

- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Friend Circles](https://leetcode.com/problems/friend-circles/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)

---

## Template 6: Bipartite Graph Check

**Key Points**:
- Use BFS/DFS with coloring (two colors)
- Color nodes alternately (0 and 1)
- If adjacent nodes have same color, graph is not bipartite
- **Time Complexity**: O(V + E) - visit each vertex and edge once
- **Space Complexity**: O(V) for color array + O(V) for queue/stack = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> color(n, -1); // -1: uncolored, 0: color1, 1: color2
    
    for (int i = 0; i < n; i++) {
        if (color[i] == -1) {
            if (!bfsBipartite(graph, i, color)) {
                return false;
            }
        }
    }
    
    return true;
}

bool bfsBipartite(vector<vector<int>>& graph, int start, vector<int>& color) {
    queue<int> q;
    q.push(start);
    color[start] = 0;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        
        for (int neighbor : graph[node]) {
            if (color[neighbor] == -1) {
                color[neighbor] = 1 - color[node]; // Alternate color
                q.push(neighbor);
            } else if (color[neighbor] == color[node]) {
                return false; // Same color as neighbor, not bipartite
            }
        }
    }
    
    return true;
}

// DFS version
bool dfsBipartite(vector<vector<int>>& graph, int node, vector<int>& color, int currentColor) {
    color[node] = currentColor;
    
    for (int neighbor : graph[node]) {
        if (color[neighbor] == -1) {
            if (!dfsBipartite(graph, neighbor, color, 1 - currentColor)) {
                return false;
            }
        } else if (color[neighbor] == currentColor) {
            return false; // Same color as neighbor, not bipartite
        }
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
from collections import deque

def is_bipartite(graph):
    n = len(graph)
    color = [-1] * n  # -1: uncolored, 0: color1, 1: color2
    
    for i in range(n):
        if color[i] == -1:
            if not bfs_bipartite(graph, i, color):
                return False
    
    return True

def bfs_bipartite(graph, start, color):
    q = deque([start])
    color[start] = 0
    
    while q:
        node = q.popleft()
        
        for neighbor in graph[node]:
            if color[neighbor] == -1:
                color[neighbor] = 1 - color[node]  # Alternate color
                q.append(neighbor)
            elif color[neighbor] == color[node]:
                return False  # Same color as neighbor, not bipartite
    
    return True

# DFS version
def dfs_bipartite(graph, node, color, current_color):
    color[node] = current_color
    
    for neighbor in graph[node]:
        if color[neighbor] == -1:
            if not dfs_bipartite(graph, neighbor, color, 1 - current_color):
                return False
        elif color[neighbor] == current_color:
            return False  # Same color as neighbor, not bipartite
    
    return True
```

</details>

### Related Problems

- [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)
- [Possible Bipartition](https://leetcode.com/problems/possible-bipartition/)

---

## Template 7: Dijkstra's Algorithm (Shortest Path)

**Key Points**:
- Find shortest path from source to all nodes in weighted graph
- Use priority queue (min-heap) to always process closest unvisited node
- Update distances to neighbors if shorter path found
- Works for non-negative edge weights
- **Time Complexity**: O((V + E) log V) with binary heap, O(V log V + E) with Fibonacci heap
- **Space Complexity**: O(V) for distance array + O(V) for priority queue = O(V)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
#include <vector>
#include <queue>
#include <climits>

vector<int> dijkstra(vector<vector<pair<int, int>>>& graph, int start) {
    int n = graph.size();
    vector<int> distance(n, INT_MAX);
    distance[start] = 0;
    
    // Min-heap: (distance, node)
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> pq;
    pq.push({0, start});
    
    while (!pq.empty()) {
        int dist = pq.top().first;
        int node = pq.top().second;
        pq.pop();
        
        // Skip if we've already found a shorter path
        if (dist > distance[node]) {
            continue;
        }
        
        // Relax edges
        for (auto& [neighbor, weight] : graph[node]) {
            int newDist = distance[node] + weight;
            if (newDist < distance[neighbor]) {
                distance[neighbor] = newDist;
                pq.push({newDist, neighbor});
            }
        }
    }
    
    return distance;
}

// Dijkstra with path reconstruction
vector<int> dijkstraWithPath(vector<vector<pair<int, int>>>& graph, 
                             int start, int end) {
    int n = graph.size();
    vector<int> distance(n, INT_MAX);
    vector<int> parent(n, -1);
    distance[start] = 0;
    
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> pq;
    pq.push({0, start});
    
    while (!pq.empty()) {
        int dist = pq.top().first;
        int node = pq.top().second;
        pq.pop();
        
        if (dist > distance[node]) {
            continue;
        }
        
        if (node == end) {
            break; // Found shortest path to end
        }
        
        for (auto& [neighbor, weight] : graph[node]) {
            int newDist = distance[node] + weight;
            if (newDist < distance[neighbor]) {
                distance[neighbor] = newDist;
                parent[neighbor] = node;
                pq.push({newDist, neighbor});
            }
        }
    }
    
    // Reconstruct path
    vector<int> path;
    if (distance[end] == INT_MAX) {
        return path; // No path exists
    }
    
    int curr = end;
    while (curr != -1) {
        path.push_back(curr);
        curr = parent[curr];
    }
    reverse(path.begin(), path.end());
    return path;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

def dijkstra(graph, start):
    n = len(graph)
    distance = [float('inf')] * n
    distance[start] = 0
    
    # Min-heap: (distance, node)
    pq = [(0, start)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        
        # Skip if we've already found a shorter path
        if dist > distance[node]:
            continue
        
        # Relax edges
        for neighbor, weight in graph[node]:
            new_dist = distance[node] + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    
    return distance

# Dijkstra with path reconstruction
def dijkstra_with_path(graph, start, end):
    n = len(graph)
    distance = [float('inf')] * n
    parent = [-1] * n
    distance[start] = 0
    
    pq = [(0, start)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        
        if dist > distance[node]:
            continue
        
        if node == end:
            break  # Found shortest path to end
        
        for neighbor, weight in graph[node]:
            new_dist = distance[node] + weight
            if new_dist < distance[neighbor]:
                distance[neighbor] = new_dist
                parent[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct path
    path = []
    if distance[end] == float('inf'):
        return path  # No path exists
    
    curr = end
    while curr != -1:
        path.append(curr)
        curr = parent[curr]
    
    return path[::-1]  # Reverse path
```

</details>

### Related Problems

- [Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
- [Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/)

---

## Key Takeaways

1. **BFS for Shortest Path**: Use BFS in unweighted graphs for shortest path
2. **DFS for Path Finding**: Use DFS to explore all paths
3. **Visited Set**: Always mark nodes as visited to avoid cycles
4. **Graph Representation**: Adjacency list is most common (vector<vector<int>>)
5. **Topological Sort**: Use for dependency ordering (DFS or Kahn's)
6. **Union-Find**: Efficient for connected components and cycle detection
7. **Cycle Detection**: Use color coding (white/gray/black) for directed graphs
8. **Bipartite Check**: Use BFS/DFS with two-color coloring
9. **Dijkstra's Algorithm**: Use priority queue for shortest path in weighted graphs (non-negative weights)

---

## Common Mistakes

1. **Not marking visited**: Forgetting to mark nodes as visited, causing infinite loops
2. **Wrong graph representation**: Confusing adjacency list vs matrix
3. **Cycle detection errors**: Not handling self-loops or multiple edges
4. **Index errors**: Off-by-one errors in node indexing (0-indexed vs 1-indexed)
5. **Not handling disconnected graphs**: Assuming graph is connected
6. **Stack overflow**: Deep recursion in large graphs (use iterative DFS)

---

## Practice Problems by Difficulty

### Easy
- [Clone Graph](https://leetcode.com/problems/clone-graph/)
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)

### Medium
- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
- [Word Ladder](https://leetcode.com/problems/word-ladder/)
- [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)
- [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)
- [Possible Bipartition](https://leetcode.com/problems/possible-bipartition/)
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/)

### Hard
- [Word Ladder II](https://leetcode.com/problems/word-ladder-ii/)
- [Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)

---

## References

* [LeetCode Graph Tag](https://leetcode.com/tag/graph/)
* [Breadth-First Search (Wikipedia)](https://en.wikipedia.org/wiki/Breadth-first_search)
* [Depth-First Search (Wikipedia)](https://en.wikipedia.org/wiki/Depth-first_search)

