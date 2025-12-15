+++
title = "Union-Find (Disjoint Set)"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 18
description = "Complete guide to Union-Find (Disjoint Set) data structure with templates in C++ and Python. Covers union, find operations, path compression, union by rank, cycle detection, and connected components with LeetCode problem references."
+++

---

## Introduction

Union-Find (Disjoint Set) is a data structure that efficiently tracks elements partitioned into disjoint sets. It supports two main operations: finding which set an element belongs to, and uniting two sets. It's essential for connectivity problems and cycle detection.

This guide provides templates and patterns for Union-Find with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Union-Find

- **Connected components**: Find number of connected components
- **Cycle detection**: Detect cycles in undirected graphs
- **Dynamic connectivity**: Check if two elements are connected
- **MST algorithms**: Kruskal's algorithm uses Union-Find
- **Network problems**: Social networks, network connectivity

### Time & Space Complexity

- **Time Complexity**: O(α(n)) per operation (nearly constant) with path compression and union by rank
- **Space Complexity**: O(n) - parent and rank arrays

---

## Pattern Variations

### Variation 1: Basic Union-Find

Simple implementation with path compression.

**Use Cases**:
- Basic connectivity problems
- Simple cycle detection

### Variation 2: Union by Rank

Optimize union operation by always attaching smaller tree to larger.

**Use Cases**:
- Performance-critical applications
- Large datasets

### Variation 3: Path Compression

Optimize find operation by flattening tree structure.

**Use Cases**:
- Frequent find operations
- Performance optimization

---

## Template 1: Basic Union-Find

**Key Points**:
- Parent array: parent[i] = parent of i
- Find: recursively find root, update parent (path compression)
- Union: make root of one tree point to root of other
- **Time Complexity**: O(α(n)) per operation with optimizations
- **Space Complexity**: O(n) - parent array

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
class UnionFind {
private:
    vector<int> parent;
    int count;
    
public:
    UnionFind(int n) {
        parent.resize(n);
        count = n;
        for (int i = 0; i < n; i++) {
            parent[i] = i; // Each element is its own parent initially
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
            parent[rootX] = rootY; // Union
            count--;
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
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
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each element is its own parent initially
        self.count = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def unite(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            self.parent[root_x] = root_y  # Union
            self.count -= 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_count(self):
        return self.count
```

</details>

### Related Problems

- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)

---

## Template 2: Union-Find with Union by Rank

**Key Points**:
- Rank array: track tree height
- Always attach smaller tree to larger tree
- Keep tree balanced for better performance
- **Time Complexity**: O(α(n)) per operation
- **Space Complexity**: O(n) - parent and rank arrays

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
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
            // Union by rank: attach smaller tree to larger tree
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
    
    bool connected(int x, int y) {
        return find(x) == find(y);
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
            # Union by rank: attach smaller tree to larger tree
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            self.count -= 1
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
    
    def get_count(self):
        return self.count
```

</details>

### Related Problems

- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)
- [Friend Circles](https://leetcode.com/problems/friend-circles/)

---

## Template 3: Cycle Detection in Undirected Graph

**Key Points**:
- For each edge, check if both endpoints are in same set
- If yes, cycle exists
- If no, union the two sets
- **Time Complexity**: O(E × α(V)) where E is edges, V is vertices
- **Space Complexity**: O(V) - Union-Find structure

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool hasCycle(vector<vector<int>>& edges, int n) {
    UnionFind uf(n);
    
    for (auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        
        if (uf.connected(u, v)) {
            return true; // Cycle detected
        }
        
        uf.unite(u, v);
    }
    
    return false;
}

// Find redundant edge (edge that creates cycle)
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    UnionFind uf(n + 1); // 1-indexed
    
    for (auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        
        if (uf.connected(u, v)) {
            return edge; // This edge creates cycle
        }
        
        uf.unite(u, v);
    }
    
    return {};
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def has_cycle(edges, n):
    uf = UnionFind(n)
    
    for edge in edges:
        u, v = edge[0], edge[1]
        
        if uf.connected(u, v):
            return True  # Cycle detected
        
        uf.unite(u, v)
    
    return False

# Find redundant edge (edge that creates cycle)
def find_redundant_connection(edges):
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed
    
    for edge in edges:
        u, v = edge[0], edge[1]
        
        if uf.connected(u, v):
            return edge  # This edge creates cycle
        
        uf.unite(u, v)
    
    return []
```

</details>

### Related Problems

- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)
- [Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)
- [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)

---

## Template 4: Number of Connected Components

**Key Points**:
- Union all edges
- Count number of distinct roots
- Or use count variable that decrements on union
- **Time Complexity**: O(E × α(V)) - process all edges
- **Space Complexity**: O(V) - Union-Find structure

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int countComponents(int n, vector<vector<int>>& edges) {
    UnionFind uf(n);
    
    // Union all edges
    for (auto& edge : edges) {
        uf.unite(edge[0], edge[1]);
    }
    
    return uf.getCount();
}

// Alternative: Count distinct roots
int countComponentsAlternative(int n, vector<vector<int>>& edges) {
    UnionFind uf(n);
    
    for (auto& edge : edges) {
        uf.unite(edge[0], edge[1]);
    }
    
    unordered_set<int> roots;
    for (int i = 0; i < n; i++) {
        roots.insert(uf.find(i));
    }
    
    return roots.size();
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def count_components(n, edges):
    uf = UnionFind(n)
    
    # Union all edges
    for edge in edges:
        uf.unite(edge[0], edge[1])
    
    return uf.get_count()

# Alternative: Count distinct roots
def count_components_alternative(n, edges):
    uf = UnionFind(n)
    
    for edge in edges:
        uf.unite(edge[0], edge[1])
    
    roots = set()
    for i in range(n):
        roots.add(uf.find(i))
    
    return len(roots)
```

</details>

### Related Problems

- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Friend Circles](https://leetcode.com/problems/friend-circles/)
- [Number of Islands](https://leetcode.com/problems/number-of-islands/) (can use Union-Find)

---

## Template 5: Kruskal's Algorithm (MST)

**Key Points**:
- Sort edges by weight
- Add edge if it doesn't create cycle
- Use Union-Find to check connectivity
- **Time Complexity**: O(E log E) - sorting edges + O(E × α(V)) union operations
- **Space Complexity**: O(V) - Union-Find structure

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int kruskalMST(int n, vector<vector<int>>& edges) {
    // Sort edges by weight
    sort(edges.begin(), edges.end(), 
         [](const vector<int>& a, const vector<int>& b) {
             return a[2] < b[2]; // Assuming [u, v, weight] format
         });
    
    UnionFind uf(n);
    int mstWeight = 0;
    int edgesAdded = 0;
    
    for (auto& edge : edges) {
        int u = edge[0];
        int v = edge[1];
        int weight = edge[2];
        
        if (!uf.connected(u, v)) {
            uf.unite(u, v);
            mstWeight += weight;
            edgesAdded++;
            
            if (edgesAdded == n - 1) {
                break; // MST complete
            }
        }
    }
    
    return mstWeight;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def kruskal_mst(n, edges):
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])  # Assuming [u, v, weight] format
    
    uf = UnionFind(n)
    mst_weight = 0
    edges_added = 0
    
    for edge in edges:
        u, v, weight = edge[0], edge[1], edge[2]
        
        if not uf.connected(u, v):
            uf.unite(u, v)
            mst_weight += weight
            edges_added += 1
            
            if edges_added == n - 1:
                break  # MST complete
    
    return mst_weight
```

</details>

### Related Problems

- [Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/)
- [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

---

## Key Takeaways

1. **Path Compression**: Make find operation faster by flattening tree
2. **Union by Rank**: Keep tree balanced by attaching smaller to larger
3. **Amortized Complexity**: O(α(n)) per operation (nearly constant)
4. **Cycle Detection**: If two nodes are already connected, adding edge creates cycle
5. **Connected Components**: Count distinct roots or use count variable
6. **Initialization**: Each element is its own parent initially

---

## Common Mistakes

1. **Not using path compression**: Forgetting to update parent in find
2. **Wrong union direction**: Not using union by rank for optimization
3. **Index errors**: Off-by-one errors (0-indexed vs 1-indexed)
4. **Not initializing**: Forgetting to initialize parent array
5. **Cycle detection logic**: Wrong logic for detecting cycles

---

## Practice Problems by Difficulty

### Medium
- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/)
- [Friend Circles](https://leetcode.com/problems/friend-circles/)
- [Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)
- [Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/)

### Hard
- [Redundant Connection II](https://leetcode.com/problems/redundant-connection-ii/)
- [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)

---

## References

* [LeetCode Union-Find Tag](https://leetcode.com/tag/union-find/)
* [Disjoint Set (Wikipedia)](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)

