+++
title = "In-place Reversal of Linked List"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 11
description = "Complete guide to In-place Reversal of Linked List pattern with templates in C++ and Python. Covers reversing entire list, reversing in groups, reversing between positions, and iterative/recursive approaches with LeetCode problem references."
+++

---

## Introduction

In-place reversal of linked lists is a fundamental technique that modifies the list structure by reversing pointers without using extra space. It's essential for solving many linked list problems efficiently.

This guide provides templates and patterns for in-place linked list reversal with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use In-place Reversal

- **Reverse entire list**: Reverse all nodes in linked list
- **Reverse in groups**: Reverse nodes in groups of k
- **Reverse between positions**: Reverse nodes between two positions
- **Reverse in pairs**: Reverse nodes in pairs
- **Palindrome checking**: Reverse and compare

### Time & Space Complexity

- **Time Complexity**: O(n) - single pass through list
- **Space Complexity**: O(1) for iterative, O(n) for recursive (call stack)

---

## Pattern Variations

### Variation 1: Reverse Entire List

Reverse all nodes from head to tail.

**Use Cases**:
- Reverse Linked List
- Reverse Linked List II

### Variation 2: Reverse in Groups

Reverse nodes in groups of k.

**Use Cases**:
- Reverse Nodes in k-Group
- Swap Nodes in Pairs

### Variation 3: Reverse Between Positions

Reverse nodes between two specific positions.

**Use Cases**:
- Reverse Linked List II
- Reverse nodes in specific range

---

## Template 1: Reverse Entire List (Iterative)

**Key Points**:
- Use three pointers: prev, curr, next
- Reverse links as you traverse
- Update pointers: prev = curr, curr = next
- **Time Complexity**: O(n) - single pass through list
- **Space Complexity**: O(1) - only three pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr != nullptr) {
        ListNode* next = curr->next;  // Store next node
        curr->next = prev;             // Reverse link
        prev = curr;                   // Move prev forward
        curr = next;                   // Move curr forward
    }
    
    return prev; // New head
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def reverse_list(head):
    prev = None
    curr = head
    
    while curr is not None:
        next_node = curr.next  # Store next node
        curr.next = prev       # Reverse link
        prev = curr            # Move prev forward
        curr = next_node       # Move curr forward
    
    return prev  # New head
```

</details>

### Related Problems

- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
- [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

---

## Template 2: Reverse Entire List (Recursive)

**Key Points**:
- Base case: head is null or single node
- Recursively reverse rest of list
- Reverse current node's link
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(n) - recursion stack depth

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* reverseListRecursive(ListNode* head) {
    // Base case: empty list or single node
    if (head == nullptr || head->next == nullptr) {
        return head;
    }
    
    // Recursively reverse rest of list
    ListNode* newHead = reverseListRecursive(head->next);
    
    // Reverse current node's link
    head->next->next = head;
    head->next = nullptr;
    
    return newHead;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def reverse_list_recursive(head):
    # Base case: empty list or single node
    if head is None or head.next is None:
        return head
    
    # Recursively reverse rest of list
    new_head = reverse_list_recursive(head.next)
    
    # Reverse current node's link
    head.next.next = head
    head.next = None
    
    return new_head
```

</details>

### Related Problems

- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

---

## Template 3: Reverse Between Positions

**Key Points**:
- Find start position (left-1) and end position (right)
- Reverse nodes between these positions
- Connect reversed portion back to original list
- **Time Complexity**: O(n) - single pass to find positions and reverse
- **Space Complexity**: O(1) - only pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* reverseBetween(ListNode* head, int left, int right) {
    if (head == nullptr || left == right) {
        return head;
    }
    
    // Dummy node to handle edge case (left = 1)
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    
    // Find node before left position
    ListNode* prev = dummy;
    for (int i = 0; i < left - 1; i++) {
        prev = prev->next;
    }
    
    // Reverse nodes from left to right
    ListNode* curr = prev->next;
    for (int i = 0; i < right - left; i++) {
        ListNode* next = curr->next;
        curr->next = next->next;
        next->next = prev->next;
        prev->next = next;
    }
    
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def reverse_between(head, left, right):
    if head is None or left == right:
        return head
    
    # Dummy node to handle edge case (left = 1)
    dummy = ListNode(0)
    dummy.next = head
    
    # Find node before left position
    prev = dummy
    for i in range(left - 1):
        prev = prev.next
    
    # Reverse nodes from left to right
    curr = prev.next
    for i in range(right - left):
        next_node = curr.next
        curr.next = next_node.next
        next_node.next = prev.next
        prev.next = next_node
    
    return dummy.next
```

</details>

### Related Problems

- [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)

---

## Template 4: Reverse in Groups of K

**Key Points**:
- Reverse first k nodes
- Recursively reverse remaining groups
- Connect groups together
- **Time Complexity**: O(n) - each node visited once
- **Space Complexity**: O(n/k) for recursion stack, or O(1) for iterative

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* reverseKGroup(ListNode* head, int k) {
    // Check if k nodes exist
    ListNode* curr = head;
    int count = 0;
    while (curr != nullptr && count < k) {
        curr = curr->next;
        count++;
    }
    
    // If k nodes exist, reverse them
    if (count == k) {
        // Reverse first k nodes
        curr = reverseKGroup(curr, k); // Reverse remaining groups
        
        // Reverse current group
        while (count > 0) {
            ListNode* next = head->next;
            head->next = curr;
            curr = head;
            head = next;
            count--;
        }
        head = curr;
    }
    
    return head;
}

// Iterative version
ListNode* reverseKGroupIterative(ListNode* head, int k) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    
    ListNode* groupPrev = dummy;
    
    while (true) {
        // Check if k nodes exist
        ListNode* kth = getKth(groupPrev, k);
        if (kth == nullptr) break;
        
        ListNode* groupNext = kth->next;
        
        // Reverse group
        ListNode* prev = groupNext;
        ListNode* curr = groupPrev->next;
        
        while (curr != groupNext) {
            ListNode* next = curr->next;
            curr->next = prev;
            prev = curr;
            curr = next;
        }
        
        ListNode* temp = groupPrev->next;
        groupPrev->next = kth;
        groupPrev = temp;
    }
    
    return dummy->next;
}

ListNode* getKth(ListNode* curr, int k) {
    while (curr != nullptr && k > 0) {
        curr = curr->next;
        k--;
    }
    return curr;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def reverse_k_group(head, k):
    # Check if k nodes exist
    curr = head
    count = 0
    while curr is not None and count < k:
        curr = curr.next
        count += 1
    
    # If k nodes exist, reverse them
    if count == k:
        # Reverse first k nodes
        curr = reverse_k_group(curr, k)  # Reverse remaining groups
        
        # Reverse current group
        while count > 0:
            next_node = head.next
            head.next = curr
            curr = head
            head = next_node
            count -= 1
        head = curr
    
    return head

# Iterative version
def reverse_k_group_iterative(head, k):
    dummy = ListNode(0)
    dummy.next = head
    
    group_prev = dummy
    
    while True:
        # Check if k nodes exist
        kth = get_kth(group_prev, k)
        if kth is None:
            break
        
        group_next = kth.next
        
        # Reverse group
        prev = group_next
        curr = group_prev.next
        
        while curr != group_next:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        temp = group_prev.next
        group_prev.next = kth
        group_prev = temp
    
    return dummy.next

def get_kth(curr, k):
    while curr is not None and k > 0:
        curr = curr.next
        k -= 1
    return curr
```

</details>

### Related Problems

- [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)

---

## Template 5: Reverse in Pairs

**Key Points**:
- Swap adjacent pairs of nodes
- Can be done recursively or iteratively
- Handle odd length lists
- **Time Complexity**: O(n) - visit each node once
- **Space Complexity**: O(1) for iterative, O(n) for recursive

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* swapPairs(ListNode* head) {
    if (head == nullptr || head->next == nullptr) {
        return head;
    }
    
    ListNode* first = head;
    ListNode* second = head->next;
    
    // Swap first two nodes
    first->next = swapPairs(second->next);
    second->next = first;
    
    return second; // New head
}

// Iterative version
ListNode* swapPairsIterative(ListNode* head) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    
    ListNode* prev = dummy;
    
    while (prev->next != nullptr && prev->next->next != nullptr) {
        ListNode* first = prev->next;
        ListNode* second = prev->next->next;
        
        // Swap
        prev->next = second;
        first->next = second->next;
        second->next = first;
        
        prev = first;
    }
    
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def swap_pairs(head):
    if head is None or head.next is None:
        return head
    
    first = head
    second = head.next
    
    # Swap first two nodes
    first.next = swap_pairs(second.next)
    second.next = first
    
    return second  # New head

# Iterative version
def swap_pairs_iterative(head):
    dummy = ListNode(0)
    dummy.next = head
    
    prev = dummy
    
    while prev.next is not None and prev.next.next is not None:
        first = prev.next
        second = prev.next.next
        
        # Swap
        prev.next = second
        first.next = second.next
        second.next = first
        
        prev = first
    
    return dummy.next
```

</details>

### Related Problems

- [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

---

## Key Takeaways

1. **Three Pointers**: Use prev, curr, next for iterative reversal
2. **Dummy Node**: Use dummy node to handle edge cases (reversing from head)
3. **Link Reversal**: Always store next before reversing link
4. **Recursive vs Iterative**: Recursive uses O(n) stack space, iterative uses O(1)
5. **Group Reversal**: Check if k nodes exist before reversing
6. **Connection**: Always connect reversed portion back to original list

---

## Common Mistakes

1. **Losing reference**: Not storing next node before reversing link
2. **Wrong return value**: Returning old head instead of new head
3. **Not connecting**: Forgetting to connect reversed portion to rest of list
4. **Null pointer**: Not checking for null before accessing next
5. **Edge cases**: Not handling empty list, single node, or k=1

---

## Practice Problems by Difficulty

### Easy
- [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

### Medium
- [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/)
- [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

---

## References

* [LeetCode Linked List Tag](https://leetcode.com/tag/linked-list/)

