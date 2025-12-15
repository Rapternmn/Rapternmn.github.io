+++
title = "Trees"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 7
description = "Trees data structure: binary trees, BST, tree traversals, and top interview problems. Covers tree operations, BFS/DFS, and more."
+++

---

## Introduction

Trees are hierarchical data structures with nodes connected by edges. They are fundamental for representing hierarchical relationships, searching, and organizing data efficiently.

---

## Tree Fundamentals

### What is a Tree?

**Tree**: A connected acyclic graph with N nodes and N-1 edges.

**Key Characteristics**:
- **Root Node**: Topmost node
- **Parent/Child**: Nodes have parent-child relationships
- **Leaf Nodes**: Nodes with no children
- **Height**: Longest path from root to leaf
- **Depth**: Distance from root to node

### Tree Node Structure

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Types of Trees

1. **Binary Tree**: Each node has at most 2 children
2. **Binary Search Tree (BST)**: Left < Root < Right
3. **Balanced Tree**: AVL, Red-Black trees
4. **Complete Tree**: All levels filled except last
5. **Full Tree**: Every node has 0 or 2 children

---

## Common Patterns

### 1. Tree Traversal

**Pattern**: Visit all nodes in specific order.

**Types**:
- **Preorder**: Root → Left → Right
- **Inorder**: Left → Root → Right
- **Postorder**: Left → Right → Root
- **Level-order**: Level by level (BFS)

**Related Pattern**: See [Tree BFS/DFS Pattern]({{< ref "../coding-patterns/6-Tree_BFS_DFS.md" >}})

---

### 2. Recursive DFS

**Pattern**: Recursively traverse subtrees.

**Template**:
```python
def dfs(root):
    if not root:
        return
    
    # Process root (preorder)
    dfs(root.left)
    # Process root (inorder)
    dfs(root.right)
    # Process root (postorder)
```

---

### 3. Iterative DFS

**Pattern**: Use stack for iterative traversal.

**Template**:
```python
def iterative_dfs(root):
    if not root:
        return []
    
    stack = [root]
    result = []
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

---

### 4. Level-Order Traversal (BFS)

**Pattern**: Use queue for level-by-level traversal.

**Template**:
```python
def level_order(root):
    from collections import deque
    
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

---

## Top Problems

### Problem 1: Maximum Depth of Binary Tree

**Problem**: Find maximum depth of binary tree.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

**Time**: O(n) | **Space**: O(h) where h is height

**Related**: [Maximum Depth of Binary Tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

---

### Problem 2: Same Tree

**Problem**: Check if two trees are identical.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isSameTree(TreeNode* p, TreeNode* q) {
    if (p == nullptr && q == nullptr) return true;
    if (p == nullptr || q == nullptr) return false;
    return (p->val == q->val && 
            isSameTree(p->left, q->left) && 
            isSameTree(p->right, q->right));
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val == q.val and 
            isSameTree(p.left, q.left) and 
            isSameTree(p.right, q.right))
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Same Tree](https://leetcode.com/problems/same-tree/)

---

### Problem 3: Invert Binary Tree

**Problem**: Invert binary tree (mirror).

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (root == nullptr) return nullptr;
    
    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    
    return root;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def invertTree(root):
    if not root:
        return None
    
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    
    return root
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/)

---

### Problem 4: Binary Tree Level Order Traversal

**Problem**: Return level-order traversal.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (root == nullptr) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> level;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);
            
            if (node->left != nullptr) {
                q.push(node->left);
            }
            if (node->right != nullptr) {
                q.push(node->right);
            }
        }
        
        result.push_back(level);
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def levelOrder(root):
    from collections import deque
    
    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

**Time**: O(n) | **Space**: O(w) where w is max width

**Related**: [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)

**Pattern**: BFS

---

### Problem 5: Validate Binary Search Tree

**Problem**: Check if tree is valid BST.

**Solution**:
```python
def isValidBST(root):
    def validate(node, min_val, max_val):
        if not node:
            return True
        if node.val <= min_val or node.val >= max_val:
            return False
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))
    
    return validate(root, float('-inf'), float('inf'))
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)

---

### Problem 6: Path Sum

**Problem**: Check if path exists with target sum.

**Solution**:
```python
def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    return (hasPathSum(root.left, targetSum - root.val) or
            hasPathSum(root.right, targetSum - root.val))
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Path Sum](https://leetcode.com/problems/path-sum/)

---

### Problem 7: Construct Binary Tree from Preorder and Inorder

**Problem**: Build tree from preorder and inorder traversals.

**Solution**:
```python
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    
    root_idx = inorder.index(root_val)
    
    root.left = buildTree(preorder[1:1+root_idx], inorder[:root_idx])
    root.right = buildTree(preorder[1+root_idx:], inorder[root_idx+1:])
    
    return root
```

**Time**: O(n²) | **Space**: O(n)

**Related**: [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

---

### Problem 8: Lowest Common Ancestor

**Problem**: Find lowest common ancestor of two nodes.

**Solution**:
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root
    return left or right
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

---

### Problem 9: Serialize and Deserialize Binary Tree

**Problem**: Serialize tree to string and deserialize back.

**Solution**:
```python
class Codec:
    def serialize(self, root):
        def dfs(node):
            if not node:
                return "None,"
            return str(node.val) + "," + dfs(node.left) + dfs(node.right)
        return dfs(root)
    
    def deserialize(self, data):
        def dfs():
            val = next(vals)
            if val == "None":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        
        vals = iter(data.split(","))
        return dfs()
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

---

### Problem 10: Binary Tree Maximum Path Sum

**Problem**: Find maximum path sum (path can start/end anywhere).

**Solution**:
```python
def maxPathSum(root):
    max_sum = float('-inf')
    
    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0
        
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))
        
        current_path = node.val + left + right
        max_sum = max(max_sum, current_path)
        
        return node.val + max(left, right)
    
    dfs(root)
    return max_sum
```

**Time**: O(n) | **Space**: O(h)

**Related**: [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

---

## Binary Search Tree Operations

### Search in BST

```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)
```

### Insert in BST

```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root
```

### Delete in BST

```python
def deleteNode(root, key):
    if not root:
        return None
    
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        root.val = min_node.val
        root.right = deleteNode(root.right, min_node.val)
    
    return root
```

---

## Key Takeaways

- Trees represent hierarchical relationships
- Binary trees: at most 2 children per node
- BST: maintains ordering property (Left < Root < Right)
- Tree traversals: Preorder, Inorder, Postorder, Level-order
- BFS uses queue, DFS uses recursion/stack
- Many tree problems use recursive DFS
- Time complexity: O(n) for most operations
- Space complexity: O(h) for recursion, O(w) for BFS
- Practice recursive thinking for tree problems

---

## Practice Problems

**Easy**:
- [Maximum Depth](https://leetcode.com/problems/maximum-depth-of-binary-tree/)
- [Same Tree](https://leetcode.com/problems/same-tree/)
- [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/)

**Medium**:
- [Binary Tree Level Order](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/)
- [Construct Binary Tree](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

**Hard**:
- [Serialize and Deserialize](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
- [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

