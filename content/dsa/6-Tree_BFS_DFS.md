+++
title = "Tree BFS/DFS"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 6
description = "Complete guide to Tree BFS (Breadth-First Search) and DFS (Depth-First Search) patterns with templates in C++ and Python. Covers level-order traversal, preorder/inorder/postorder, iterative and recursive approaches with LeetCode problem references."
+++

---

## Introduction

Tree traversal is fundamental to solving tree problems. BFS (Breadth-First Search) explores nodes level by level, while DFS (Depth-First Search) explores as deep as possible before backtracking. Both techniques are essential for tree manipulation, path finding, and tree analysis.

This guide provides templates and patterns for Tree BFS and DFS with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use BFS vs DFS

**Use BFS when**:
- Finding shortest path in unweighted tree
- Level-order traversal needed
- Finding nodes at specific level
- Printing tree level by level

**Use DFS when**:
- Exploring all paths
- Finding maximum depth
- Tree validation
- Path sum problems
- Tree construction from traversal

### Time & Space Complexity

- **BFS Time**: O(n) - visit each node once
- **BFS Space**: O(w) where w is maximum width of tree
- **DFS Time**: O(n) - visit each node once
- **DFS Space**: O(h) where h is height of tree (recursion stack)

---

## Pattern Variations

### Variation 1: BFS (Level-Order Traversal)

Traverse tree level by level using a queue.

**Use Cases**:
- Level-order traversal
- Binary Tree Level Order Traversal
- Find nodes at each level

### Variation 2: DFS Preorder (Root-Left-Right)

Visit root, then left subtree, then right subtree.

**Use Cases**:
- Tree serialization
- Copy tree
- Tree construction

### Variation 3: DFS Inorder (Left-Root-Right)

Visit left subtree, then root, then right subtree.

**Use Cases**:
- Binary Search Tree operations
- Validate BST
- Inorder successor/predecessor

### Variation 4: DFS Postorder (Left-Right-Root)

Visit left subtree, then right subtree, then root.

**Use Cases**:
- Tree deletion
- Calculate subtree properties
- Expression tree evaluation

---

## Template 1: BFS (Level-Order Traversal)

**Key Points**:
- Use queue to store nodes
- Process nodes level by level
- Add children to queue for next level
- Track level if needed
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(w) where w is maximum width of tree (queue size)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            currentLevel.push_back(node->val);
            
            if (node->left != nullptr) {
                q.push(node->left);
            }
            if (node->right != nullptr) {
                q.push(node->right);
            }
        }
        
        result.push_back(currentLevel);
    }
    
    return result;
}

// Simple level-order (single list)
vector<int> levelOrderSimple(TreeNode* root) {
    vector<int> result;
    if (root == nullptr) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();
        result.push_back(node->val);
        
        if (node->left != nullptr) {
            q.push(node->left);
        }
        if (node->right != nullptr) {
            q.push(node->right);
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def level_order(root):
    result = []
    if root is None:
        return result
    
    q = [root]
    
    while q:
        level_size = len(q)
        current_level = []
        
        for i in range(level_size):
            node = q.pop(0)
            current_level.append(node.val)
            
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
        
        result.append(current_level)
    
    return result

# Simple level-order (single list)
def level_order_simple(root):
    result = []
    if root is None:
        return result
    
    q = [root]
    
    while q:
        node = q.pop(0)
        result.append(node.val)
        
        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)
    
    return result
```

</details>

### Related Problems

- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
- [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/)

---

## Template 2: DFS Preorder (Recursive & Iterative)

**Key Points**:
- Visit: Root → Left → Right
- Recursive: Process root, recurse left, recurse right
- Iterative: Use stack, push right then left
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(h) where h is height (recursion stack or explicit stack)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Recursive Preorder
vector<int> preorderRecursive(TreeNode* root) {
    vector<int> result;
    preorderHelper(root, result);
    return result;
}

void preorderHelper(TreeNode* root, vector<int>& result) {
    if (root == nullptr) return;
    
    result.push_back(root->val);  // Visit root
    preorderHelper(root->left, result);   // Left
    preorderHelper(root->right, result);  // Right
}

// Iterative Preorder
vector<int> preorderIterative(TreeNode* root) {
    vector<int> result;
    if (root == nullptr) return result;
    
    stack<TreeNode*> st;
    st.push(root);
    
    while (!st.empty()) {
        TreeNode* node = st.top();
        st.pop();
        result.push_back(node->val);
        
        // Push right first, then left (stack is LIFO)
        if (node->right != nullptr) {
            st.push(node->right);
        }
        if (node->left != nullptr) {
            st.push(node->left);
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Recursive Preorder
def preorder_recursive(root):
    result = []
    preorder_helper(root, result)
    return result

def preorder_helper(root, result):
    if root is None:
        return
    
    result.append(root.val)  # Visit root
    preorder_helper(root.left, result)   # Left
    preorder_helper(root.right, result)  # Right

# Iterative Preorder
def preorder_iterative(root):
    result = []
    if root is None:
        return result
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # Push right first, then left (stack is LIFO)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    
    return result
```

</details>

### Related Problems

- [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

---

## Template 3: DFS Inorder (Recursive & Iterative)

**Key Points**:
- Visit: Left → Root → Right
- Recursive: Recurse left, process root, recurse right
- Iterative: Use stack, go left first, then process, then go right
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(h) where h is height (recursion stack or explicit stack)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Recursive Inorder
vector<int> inorderRecursive(TreeNode* root) {
    vector<int> result;
    inorderHelper(root, result);
    return result;
}

void inorderHelper(TreeNode* root, vector<int>& result) {
    if (root == nullptr) return;
    
    inorderHelper(root->left, result);   // Left
    result.push_back(root->val);         // Visit root
    inorderHelper(root->right, result);   // Right
}

// Iterative Inorder
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while (curr != nullptr || !st.empty()) {
        // Go to leftmost node
        while (curr != nullptr) {
            st.push(curr);
            curr = curr->left;
        }
        
        // Process node
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        
        // Go to right subtree
        curr = curr->right;
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Recursive Inorder
def inorder_recursive(root):
    result = []
    inorder_helper(root, result)
    return result

def inorder_helper(root, result):
    if root is None:
        return
    
    inorder_helper(root.left, result)   # Left
    result.append(root.val)              # Visit root
    inorder_helper(root.right, result)   # Right

# Iterative Inorder
def inorder_iterative(root):
    result = []
    stack = []
    curr = root
    
    while curr is not None or stack:
        # Go to leftmost node
        while curr is not None:
            stack.append(curr)
            curr = curr.left
        
        # Process node
        curr = stack.pop()
        result.append(curr.val)
        
        # Go to right subtree
        curr = curr.right
    
    return result
```

</details>

### Related Problems

- [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)

---

## Template 4: DFS Postorder (Recursive & Iterative)

**Key Points**:
- Visit: Left → Right → Root
- Recursive: Recurse left, recurse right, process root
- Iterative: Use two stacks or reverse preorder
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(h) where h is height (recursion stack or explicit stack)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Recursive Postorder
vector<int> postorderRecursive(TreeNode* root) {
    vector<int> result;
    postorderHelper(root, result);
    return result;
}

void postorderHelper(TreeNode* root, vector<int>& result) {
    if (root == nullptr) return;
    
    postorderHelper(root->left, result);   // Left
    postorderHelper(root->right, result);   // Right
    result.push_back(root->val);           // Visit root
}

// Iterative Postorder (using two stacks)
vector<int> postorderIterative(TreeNode* root) {
    vector<int> result;
    if (root == nullptr) return result;
    
    stack<TreeNode*> st1, st2;
    st1.push(root);
    
    while (!st1.empty()) {
        TreeNode* node = st1.top();
        st1.pop();
        st2.push(node);
        
        if (node->left != nullptr) {
            st1.push(node->left);
        }
        if (node->right != nullptr) {
            st1.push(node->right);
        }
    }
    
    while (!st2.empty()) {
        result.push_back(st2.top()->val);
        st2.pop();
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Recursive Postorder
def postorder_recursive(root):
    result = []
    postorder_helper(root, result)
    return result

def postorder_helper(root, result):
    if root is None:
        return
    
    postorder_helper(root.left, result)   # Left
    postorder_helper(root.right, result)  # Right
    result.append(root.val)               # Visit root

# Iterative Postorder (using two stacks)
def postorder_iterative(root):
    result = []
    if root is None:
        return result
    
    stack1 = [root]
    stack2 = []
    
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        
        if node.left is not None:
            stack1.append(node.left)
        if node.right is not None:
            stack1.append(node.right)
    
    while stack2:
        result.append(stack2.pop().val)
    
    return result
```

</details>

### Related Problems

- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- [Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

---

## Template 5: DFS for Path Problems

**Key Points**:
- Track path while traversing
- Backtrack when returning from recursion
- Check conditions at leaf nodes
- Accumulate results
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(h) for recursion stack + O(h) for path storage = O(h)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Find all root-to-leaf paths
vector<vector<int>> allPaths(TreeNode* root) {
    vector<vector<int>> result;
    vector<int> path;
    dfsPaths(root, path, result);
    return result;
}

void dfsPaths(TreeNode* root, vector<int>& path, vector<vector<int>>& result) {
    if (root == nullptr) return;
    
    path.push_back(root->val);
    
    // Leaf node
    if (root->left == nullptr && root->right == nullptr) {
        result.push_back(path);
    } else {
        dfsPaths(root->left, path, result);
        dfsPaths(root->right, path, result);
    }
    
    // Backtrack
    path.pop_back();
}

// Path Sum
bool hasPathSum(TreeNode* root, int targetSum) {
    if (root == nullptr) return false;
    
    if (root->left == nullptr && root->right == nullptr) {
        return root->val == targetSum;
    }
    
    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
# Find all root-to-leaf paths
def all_paths(root):
    result = []
    path = []
    dfs_paths(root, path, result)
    return result

def dfs_paths(root, path, result):
    if root is None:
        return
    
    path.append(root.val)
    
    # Leaf node
    if root.left is None and root.right is None:
        result.append(path[:])  # Copy of path
    else:
        dfs_paths(root.left, path, result)
        dfs_paths(root.right, path, result)
    
    # Backtrack
    path.pop()

# Path Sum
def has_path_sum(root, target_sum):
    if root is None:
        return False
    
    if root.left is None and root.right is None:
        return root.val == target_sum
    
    return (has_path_sum(root.left, target_sum - root.val) or
            has_path_sum(root.right, target_sum - root.val))
```

</details>

### Related Problems

- [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)
- [Path Sum](https://leetcode.com/problems/path-sum/)
- [Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)

---

## Key Takeaways

1. **BFS**: Use queue, process level by level, good for shortest path
2. **DFS Preorder**: Root → Left → Right, good for tree copying/serialization
3. **DFS Inorder**: Left → Root → Right, essential for BST operations
4. **DFS Postorder**: Left → Right → Root, good for deletion/subtree properties
5. **Iterative vs Recursive**: Iterative uses stack, recursive uses call stack
6. **Path Problems**: Track path, backtrack when returning
7. **Null Checks**: Always check for null before accessing children

---

## Common Mistakes

1. **Forgetting null checks**: Accessing children without checking null
2. **Wrong traversal order**: Confusing preorder/inorder/postorder
3. **Not backtracking**: In path problems, forgetting to remove from path
4. **Stack order**: In iterative preorder, wrong order of pushing children
5. **Level tracking**: In BFS, not tracking level size correctly
6. **Memory issues**: Not copying path in path problems (reference vs value)

---

## Practice Problems by Difficulty

### Easy
- [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)
- [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)
- [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)
- [Path Sum](https://leetcode.com/problems/path-sum/)
- [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/)

### Medium
- [Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)
- [Binary Tree Zigzag Level Order Traversal](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)
- [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)
- [Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
- [Path Sum II](https://leetcode.com/problems/path-sum-ii/)
- [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/)
- [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

### Hard
- [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

---

## References

* [LeetCode Tree Tag](https://leetcode.com/tag/tree/)
* [LeetCode Binary Tree Tag](https://leetcode.com/tag/binary-tree/)
* [Tree Traversal (Wikipedia)](https://en.wikipedia.org/wiki/Tree_traversal)

