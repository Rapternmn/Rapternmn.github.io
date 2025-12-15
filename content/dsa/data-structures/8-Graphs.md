+++
title = "Graphs"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Graphs data structure: representation, BFS, DFS, and top interview problems. Covers graph algorithms, shortest paths, and more."
+++

---

## Introduction

Graphs are collections of nodes (vertices) connected by edges. They model relationships and are fundamental for solving problems involving networks, paths, and dependencies.

---

## Graph Fundamentals

### What is a Graph?

**Graph**: A collection of vertices (nodes) and edges (connections).

**Key Characteristics**:
- **Vertices (V)**: Nodes in the graph
- **Edges (E)**: Connections between vertices
- **Directed/Undirected**: Edges may have direction
- **Weighted/Unweighted**: Edges may have weights
- **Cyclic/Acyclic**: May or may not contain cycles

### Graph Representations

**1. Adjacency List** (Most common):

<details open>
<summary><strong>📋 C++</strong></summary>

```cpp
vector<vector<int>> graph = {
    {1, 2},
    {0, 3},
    {0, 3},
    {1, 2}
};

// Or using unordered_map
unordered_map<int, vector<int>> graph;
graph[0] = {1, 2};
graph[1] = {0, 3};
graph[2] = {0, 3};
graph[3] = {1, 2};
```

</details>

<details>
<summary><strong>🐍 Python</strong></summary>

```python
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}
```

</details>

**2. Adjacency Matrix**:

<details open>
<summary><strong>📋 C++</strong></summary>

```cpp
vector<vector<int>> graph = {
    {0, 1, 1, 0},
    {1, 0, 0, 1},
    {1, 0, 0, 1},
    {0, 1, 1, 0}
};
```

</details>

<details>
<summary><strong>🐍 Python</strong></summary>

```python
graph = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

</details>

**3. Edge List**:

<details open>
<summary><strong>📋 C++</strong></summary>

```cpp
vector<pair<int, int>> edges = {
    {0, 1}, {0, 2}, {1, 3}, {2, 3}
};
```

</details>

<details>
<summary><strong>🐍 Python</strong></summary>

```python
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
```

</details>

---

## Common Patterns

### 1. BFS (Breadth-First Search)

**Pattern**: Explore level by level using queue.

**When to Use**:
- Shortest path (unweighted)
- Level-order traversal
- Finding nodes at distance k

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> bfs(vector<vector<int>>& graph, int start) {
    queue<int> q;
    unordered_set<int> visited;
    vector<int> result;
    
    q.push(start);
    visited.insert(start);
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);
        // Process node
        
        for (int neighbor : graph[node]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
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
def bfs(graph, start):
    from collections import deque
    
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

</details>

**Related Pattern**: See [Graph BFS/DFS Pattern]({{< ref "../coding-patterns/14-Graph_BFS_DFS.md" >}})

---

### 2. DFS (Depth-First Search)

**Pattern**: Explore as deep as possible.

**When to Use**:
- Path finding
- Cycle detection
- Connected components
- Topological sort

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
void dfs(vector<vector<int>>& graph, int node, 
         unordered_set<int>& visited, vector<int>& result) {
    visited.insert(node);
    result.push_back(node);
    // Process node
    
    for (int neighbor : graph[node]) {
        if (visited.find(neighbor) == visited.end()) {
            dfs(graph, neighbor, visited, result);
        }
    }
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def dfs(graph, node, visited):
    visited.add(node)
    # Process node
    
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

</details>

---

### 3. Topological Sort

**Pattern**: Order nodes based on dependencies.

**When to Use**:
- Course scheduling
- Build order
- Task dependencies

**Template**:

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<int> topologicalSort(vector<vector<int>>& graph, int n) {
    vector<int> inDegree(n, 0);
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
    
    return (result.size() == n) ? result : vector<int>();
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []
```

</details>

**Related Pattern**: See [Topological Sort Pattern]({{< ref "../coding-patterns/17-Topological_Sort.md" >}})

---

## Top Problems

### Problem 1: Number of Islands

**Problem**: Count number of islands in 2D grid.

**Solution** (DFS):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int numIslands(vector<vector<char>>& grid) {
    if (grid.empty()) return 0;
    
    int rows = grid.size();
    int cols = grid[0].size();
    int islands = 0;
    
    function<void(int, int)> dfs = [&](int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] == '0') {
            return;
        }
        grid[r][c] = '0';
        dfs(r + 1, c);
        dfs(r - 1, c);
        dfs(r, c + 1);
        dfs(r, c - 1);
    };
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                islands++;
                dfs(r, c);
            }
        }
    }
    
    return islands;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def numIslands(grid):
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    islands = 0
    
    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                islands += 1
                dfs(r, c)
    
    return islands
```

**Time**: O(m * n) | **Space**: O(m * n)

**Related**: [Number of Islands](https://leetcode.com/problems/number-of-islands/)

**Pattern**: DFS

---

### Problem 2: Clone Graph

**Problem**: Deep clone undirected graph.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
};

Node* cloneGraph(Node* node) {
    if (node == nullptr) return nullptr;
    
    unordered_map<Node*, Node*> cloneMap;
    
    function<Node*(Node*)> dfs = [&](Node* original) -> Node* {
        if (cloneMap.find(original) != cloneMap.end()) {
            return cloneMap[original];
        }
        
        Node* clone = new Node(original->val);
        cloneMap[original] = clone;
        
        for (Node* neighbor : original->neighbors) {
            clone->neighbors.push_back(dfs(neighbor));
        }
        
        return clone;
    };
    
    return dfs(node);
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def cloneGraph(node):
    if not node:
        return None
    
    clone_map = {}
    
    def dfs(original):
        if original in clone_map:
            return clone_map[original]
        
        clone = Node(original.val)
        clone_map[original] = clone
        
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
        
        return clone
    
    return dfs(node)
```

</details>

**Time**: O(V + E) | **Space**: O(V)

**Related**: [Clone Graph](https://leetcode.com/problems/clone-graph/)

---

### Problem 3: Course Schedule

**Problem**: Check if can finish all courses (no cycles).

**Solution** (Topological Sort):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses);
    vector<int> inDegree(numCourses, 0);
    
    for (auto& edge : prerequisites) {
        int course = edge[0];
        int prereq = edge[1];
        graph[prereq].push_back(course);
        inDegree[course]++;
    }
    
    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    
    int count = 0;
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        count++;
        
        for (int neighbor : graph[course]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }
    
    return count == numCourses;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def canFinish(numCourses, prerequisites):
    from collections import deque, defaultdict
    
    graph = defaultdict(list)
    in_degree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    count = 0
    
    while queue:
        course = queue.popleft()
        count += 1
        
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return count == numCourses
```

</details>

**Time**: O(V + E) | **Space**: O(V + E)

**Related**: [Course Schedule](https://leetcode.com/problems/course-schedule/)

**Pattern**: Topological Sort

---

### Problem 4: Word Ladder

**Problem**: Find shortest transformation sequence.

**Solution** (BFS):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> wordSet(wordList.begin(), wordList.end());
    if (wordSet.find(endWord) == wordSet.end()) {
        return 0;
    }
    
    queue<pair<string, int>> q;
    q.push({beginWord, 1});
    unordered_set<string> visited;
    visited.insert(beginWord);
    
    while (!q.empty()) {
        auto [word, length] = q.front();
        q.pop();
        
        if (word == endWord) {
            return length;
        }
        
        for (int i = 0; i < word.length(); i++) {
            string newWord = word;
            for (char c = 'a'; c <= 'z'; c++) {
                newWord[i] = c;
                if (wordSet.find(newWord) != wordSet.end() && 
                    visited.find(newWord) == visited.end()) {
                    visited.insert(newWord);
                    q.push({newWord, length + 1});
                }
            }
        }
    }
    
    return 0;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def ladderLength(beginWord, endWord, wordList):
    from collections import deque
    
    wordSet = set(wordList)
    if endWord not in wordSet:
        return 0
    
    queue = deque([(beginWord, 1)])
    visited = {beginWord}
    
    while queue:
        word, length = queue.popleft()
        
        if word == endWord:
            return length
        
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word in wordSet and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))
    
    return 0
```

</details>

**Time**: O(M * N) where M is word length, N is word list size | **Space**: O(N)

**Related**: [Word Ladder](https://leetcode.com/problems/word-ladder/)

**Pattern**: BFS

---

### Problem 5: Pacific Atlantic Water Flow

**Problem**: Find cells that can flow to both oceans.

**Solution** (DFS from boundaries):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    if (heights.empty()) return {};
    
    int rows = heights.size();
    int cols = heights[0].size();
    unordered_set<int> pacific;
    unordered_set<int> atlantic;
    
    function<void(int, int, unordered_set<int>&)> dfs = 
        [&](int r, int c, unordered_set<int>& ocean) {
        int key = r * cols + c;
        ocean.insert(key);
        
        vector<pair<int, int>> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (auto& [dr, dc] : directions) {
            int nr = r + dr;
            int nc = c + dc;
            int nkey = nr * cols + nc;
            
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                ocean.find(nkey) == ocean.end() &&
                heights[nr][nc] >= heights[r][c]) {
                dfs(nr, nc, ocean);
            }
        }
    };
    
    for (int r = 0; r < rows; r++) {
        dfs(r, 0, pacific);
        dfs(r, cols - 1, atlantic);
    }
    
    for (int c = 0; c < cols; c++) {
        dfs(0, c, pacific);
        dfs(rows - 1, c, atlantic);
    }
    
    vector<vector<int>> result;
    for (int key : pacific) {
        if (atlantic.find(key) != atlantic.end()) {
            result.push_back({key / cols, key % cols});
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def pacificAtlantic(heights):
    if not heights:
        return []
    
    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()
    
    def dfs(r, c, ocean):
        ocean.add((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and 
                (nr, nc) not in ocean and 
                heights[nr][nc] >= heights[r][c]):
                dfs(nr, nc, ocean)
    
    for r in range(rows):
        dfs(r, 0, pacific)
        dfs(r, cols - 1, atlantic)
    
    for c in range(cols):
        dfs(0, c, pacific)
        dfs(rows - 1, c, atlantic)
    
    return list(pacific & atlantic)
```

</details>

**Time**: O(m * n) | **Space**: O(m * n)

**Related**: [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/)

---

### Problem 6: Is Graph Bipartite?

**Problem**: Check if graph can be colored with 2 colors.

**Solution** (BFS with coloring):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isBipartite(vector<vector<int>>& graph) {
    vector<int> color(graph.size(), -1);
    
    for (int node = 0; node < graph.size(); node++) {
        if (color[node] == -1) {
            queue<int> q;
            q.push(node);
            color[node] = 0;
            
            while (!q.empty()) {
                int curr = q.front();
                q.pop();
                
                for (int neighbor : graph[curr]) {
                    if (color[neighbor] == -1) {
                        color[neighbor] = 1 - color[curr];
                        q.push(neighbor);
                    } else if (color[neighbor] == color[curr]) {
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isBipartite(graph):
    color = {}
    
    for node in range(len(graph)):
        if node not in color:
            from collections import deque
            queue = deque([node])
            color[node] = 0
            
            while queue:
                curr = queue.popleft()
                
                for neighbor in graph[curr]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[curr]
                        queue.append(neighbor)
                    elif color[neighbor] == color[curr]:
                        return False
    
    return True
```

</details>

**Time**: O(V + E) | **Space**: O(V)

**Related**: [Is Graph Bipartite?](https://leetcode.com/problems/is-graph-bipartite/)

**Pattern**: BFS

---

### Problem 7: Network Delay Time

**Problem**: Find minimum time for signal to reach all nodes.

**Solution** (Dijkstra's):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int networkDelayTime(vector<vector<int>>& times, int n, int k) {
    vector<vector<pair<int, int>>> graph(n + 1);
    for (auto& edge : times) {
        int u = edge[0];
        int v = edge[1];
        int w = edge[2];
        graph[u].push_back({v, w});
    }
    
    vector<int> dist(n + 1, INT_MAX);
    dist[k] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, 
                   greater<pair<int, int>>> heap;
    heap.push({0, k});
    
    while (!heap.empty()) {
        auto [d, node] = heap.top();
        heap.pop();
        
        if (d > dist[node]) continue;
        
        for (auto& [neighbor, weight] : graph[node]) {
            int newDist = d + weight;
            if (newDist < dist[neighbor]) {
                dist[neighbor] = newDist;
                heap.push({newDist, neighbor});
            }
        }
    }
    
    int maxTime = *max_element(dist.begin() + 1, dist.end());
    return (maxTime == INT_MAX) ? -1 : maxTime;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def networkDelayTime(times, n, k):
    import heapq
    from collections import defaultdict
    
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))
    
    dist = {i: float('inf') for i in range(1, n + 1)}
    dist[k] = 0
    heap = [(0, k)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    max_time = max(dist.values())
    return max_time if max_time != float('inf') else -1
```

</details>

**Time**: O(E log V) | **Space**: O(V + E)

**Related**: [Network Delay Time](https://leetcode.com/problems/network-delay-time/)

---

### Problem 8: Redundant Connection

**Problem**: Find edge that creates cycle in tree.

**Solution** (Union-Find):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    vector<int> parent(n + 1);
    iota(parent.begin(), parent.end(), 0);
    
    function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    };
    
    function<bool(int, int)> unionNodes = [&](int x, int y) -> bool {
        int px = find(x);
        int py = find(y);
        if (px == py) return false;
        parent[px] = py;
        return true;
    };
    
    for (auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        if (!unionNodes(u, v)) {
            return {u, v};
        }
    }
    
    return {};
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def findRedundantConnection(edges):
    parent = list(range(len(edges) + 1))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True
    
    for u, v in edges:
        if not union(u, v):
            return [u, v]
```

</details>

**Time**: O(n * α(n)) | **Space**: O(n)

**Related**: [Redundant Connection](https://leetcode.com/problems/redundant-connection/)

**Pattern**: Union-Find

---

### Problem 9: Cheapest Flights Within K Stops

**Problem**: Find cheapest flight with at most K stops.

**Solution** (Bellman-Ford variant):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    vector<int> prices(n, INT_MAX);
    prices[src] = 0;
    
    for (int i = 0; i <= k; i++) {
        vector<int> temp = prices;
        for (auto& flight : flights) {
            int u = flight[0];
            int v = flight[1];
            int w = flight[2];
            if (prices[u] != INT_MAX) {
                temp[v] = min(temp[v], prices[u] + w);
            }
        }
        prices = temp;
    }
    
    return (prices[dst] == INT_MAX) ? -1 : prices[dst];
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def findCheapestPrice(n, flights, src, dst, k):
    prices = [float('inf')] * n
    prices[src] = 0
    
    for _ in range(k + 1):
        temp = prices[:]
        for u, v, w in flights:
            if prices[u] != float('inf'):
                temp[v] = min(temp[v], prices[u] + w)
        prices = temp
    
    return prices[dst] if prices[dst] != float('inf') else -1
```

</details>

**Time**: O(K * E) | **Space**: O(V)

**Related**: [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

---

### Problem 10: Critical Connections

**Problem**: Find bridges in graph (critical connections).

**Solution** (Tarjan's algorithm):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
    vector<vector<int>> graph(n);
    for (auto& edge : connections) {
        int u = edge[0];
        int v = edge[1];
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    
    vector<int> disc(n, -1);
    vector<int> low(n, -1);
    int time = 0;
    vector<vector<int>> result;
    
    function<void(int, int)> dfs = [&](int node, int parent) {
        disc[node] = low[node] = time++;
        
        for (int neighbor : graph[node]) {
            if (neighbor == parent) continue;
            
            if (disc[neighbor] == -1) {
                dfs(neighbor, node);
                low[node] = min(low[node], low[neighbor]);
                if (low[neighbor] > disc[node]) {
                    result.push_back({node, neighbor});
                }
            } else {
                low[node] = min(low[node], disc[neighbor]);
            }
        }
    };
    
    dfs(0, -1);
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def criticalConnections(n, connections):
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    disc = [-1] * n
    low = [-1] * n
    time = 0
    result = []
    
    def dfs(node, parent):
        nonlocal time
        disc[node] = low[node] = time
        time += 1
        
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if disc[neighbor] == -1:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    result.append([node, neighbor])
            else:
                low[node] = min(low[node], disc[neighbor])
    
    dfs(0, -1)
    return result
```

</details>

**Time**: O(V + E) | **Space**: O(V)

**Related**: [Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/)

---

## Key Takeaways

- Graphs model relationships and networks
- Adjacency list is most common representation
- BFS for shortest path (unweighted), level-order
- DFS for path finding, cycle detection, components
- Topological sort for dependency ordering
- Union-Find for connectivity and cycles
- Dijkstra's for shortest path (weighted, non-negative)
- Time complexity: O(V + E) for most graph algorithms
- Practice BFS/DFS templates - very common

---

## Practice Problems

**Easy**:
- [Find the Town Judge](https://leetcode.com/problems/find-the-town-judge/)
- [Find Center of Star Graph](https://leetcode.com/problems/find-center-of-star-graph/)

**Medium**:
- [Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [Course Schedule](https://leetcode.com/problems/course-schedule/)
- [Clone Graph](https://leetcode.com/problems/clone-graph/)

**Hard**:
- [Word Ladder](https://leetcode.com/problems/word-ladder/)
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/)
- [Critical Connections](https://leetcode.com/problems/critical-connections-in-a-network/)

