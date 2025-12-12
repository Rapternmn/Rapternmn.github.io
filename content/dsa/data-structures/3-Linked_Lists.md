+++
title = "Linked Lists"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 3
description = "Linked Lists: singly, doubly, circular linked lists, operations, and top interview problems. Covers fast/slow pointers, reversal, and more."
+++

---

## Introduction

Linked lists are linear data structures where elements are stored in nodes, each containing data and a reference to the next node. They provide dynamic memory allocation and efficient insertion/deletion.

---

## Linked List Fundamentals

### What is a Linked List?

**Linked List**: A linear data structure where elements (nodes) are linked using pointers/references.

**Key Characteristics**:
- **Dynamic Size**: Can grow/shrink at runtime
- **Non-Contiguous**: Nodes not stored in contiguous memory
- **Sequential Access**: Must traverse from head to access elements
- **Memory Efficient**: Only allocates memory as needed

### Types of Linked Lists

1. **Singly Linked List**: Each node points to next node
2. **Doubly Linked List**: Each node points to both next and previous
3. **Circular Linked List**: Last node points back to first

### Time Complexity

| Operation | Time Complexity |
|-----------|----------------|
| Access by index | O(n) |
| Search | O(n) |
| Insert at head | O(1) |
| Insert at tail | O(1) with tail pointer, O(n) without |
| Insert at position | O(n) |
| Delete | O(n) |
| Update | O(n) |

---

## Node Structure

### Singly Linked List Node

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Doubly Linked List Node

```python
class DoublyListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev
```

---

## Common Patterns

### 1. Fast & Slow Pointers (Two Pointers)

**Pattern**: Use two pointers moving at different speeds.

**When to Use**:
- Finding middle node
- Detecting cycles
- Finding kth node from end
- Palindrome checks

**Template**:
```python
def fast_slow_pointers(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # slow is at middle
```

**Related Pattern**: See [Fast & Slow Pointers Pattern]({{< ref "../5-Fast_Slow_Pointers.md" >}})

---

### 2. Reversing Linked List

**Pattern**: Reverse linked list iteratively or recursively.

**Iterative**:
```python
def reverse_list(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

**Recursive**:
```python
def reverse_list_recursive(head):
    if not head or not head.next:
        return head
    
    new_head = reverse_list_recursive(head.next)
    head.next.next = head
    head.next = None
    
    return new_head
```

**Related Pattern**: See [In-Place Reversal Pattern]({{< ref "../11-In_Place_Reversal_Linked_List.md" >}})

---

### 3. Dummy Node Pattern

**Pattern**: Use dummy node to simplify edge cases.

**When to Use**:
- Merging lists
- Removing nodes
- Creating new list
- Handling empty list

**Template**:
```python
def dummy_node_pattern(head):
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    # Process nodes
    while current.next:
        # Logic here
        current = current.next
    
    return dummy.next
```

---

### 4. Two Pointers (Different Starting Points)

**Pattern**: Use two pointers starting at different positions.

**When to Use**:
- Finding intersection
- Removing nth node from end
- Finding kth node

**Template**:
```python
def two_pointers_different(head, n):
    first = second = head
    
    # Move first pointer n steps ahead
    for _ in range(n):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    return second
```

---

## Top Problems

### Problem 1: Reverse Linked List

**Problem**: Reverse a singly linked list.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* current = head;
    
    while (current != nullptr) {
        ListNode* nextNode = current->next;
        current->next = prev;
        prev = current;
        current = nextNode;
    }
    
    return prev;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def reverseList(head):
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)

---

### Problem 2: Merge Two Sorted Lists

**Problem**: Merge two sorted linked lists.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode* dummy = new ListNode(0);
    ListNode* current = dummy;
    
    while (list1 != nullptr && list2 != nullptr) {
        if (list1->val <= list2->val) {
            current->next = list1;
            list1 = list1->next;
        } else {
            current->next = list2;
            list2 = list2->next;
        }
        current = current->next;
    }
    
    current->next = (list1 != nullptr) ? list1 : list2;
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def mergeTwoLists(list1, list2):
    dummy = ListNode(0)
    current = dummy
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    current.next = list1 or list2
    return dummy.next
```

**Time**: O(n + m) | **Space**: O(1)

**Related**: [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

---

### Problem 3: Linked List Cycle

**Problem**: Detect if linked list has cycle.

**Solution** (Floyd's Cycle Detection):

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            return true;
        }
    }
    
    return false;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def hasCycle(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    
    return False
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)

**Pattern**: Fast & Slow Pointers

---

### Problem 4: Linked List Cycle II

**Problem**: Find node where cycle begins.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
ListNode* detectCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    // Find meeting point
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            break;
        }
    }
    
    if (fast == nullptr || fast->next == nullptr) {
        return nullptr;
    }
    
    // Find cycle start
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def detectCycle(head):
    slow = fast = head
    
    # Find meeting point
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)

**Pattern**: Fast & Slow Pointers

---

### Problem 5: Remove Nth Node From End

**Problem**: Remove nth node from end of list.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(0);
    dummy->next = head;
    ListNode* first = dummy;
    ListNode* second = dummy;
    
    // Move first n+1 steps ahead
    for (int i = 0; i <= n; i++) {
        first = first->next;
    }
    
    // Move both until first reaches end
    while (first != nullptr) {
        first = first->next;
        second = second->next;
    }
    
    // Remove node
    second->next = second->next->next;
    return dummy->next;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    
    # Move first n+1 steps ahead
    for _ in range(n + 1):
        first = first.next
    
    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next
    
    # Remove node
    second.next = second.next.next
    return dummy.next
```

</details>

**Time**: O(n) | **Space**: O(1)

**Related**: [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

**Pattern**: Two Pointers

---

### Problem 6: Middle of Linked List

**Problem**: Find middle node of linked list.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
ListNode* middleNode(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return slow;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def middleNode(head):
    slow = fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)

**Pattern**: Fast & Slow Pointers

---

### Problem 7: Palindrome Linked List

**Problem**: Check if linked list is palindrome.

**Solution**:

<details open>
<summary><strong>📋 C++ Solution</strong></summary>

```cpp
bool isPalindrome(ListNode* head) {
    // Find middle
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    
    // Reverse second half
    ListNode* prev = nullptr;
    while (slow != nullptr) {
        ListNode* nextNode = slow->next;
        slow->next = prev;
        prev = slow;
        slow = nextNode;
    }
    
    // Compare
    while (prev != nullptr) {
        if (head->val != prev->val) {
            return false;
        }
        head = head->next;
        prev = prev->next;
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Solution</strong></summary>

```python
def isPalindrome(head):
    # Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    prev = None
    while slow:
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node
    
    # Compare
    while prev:
        if head.val != prev.val:
            return False
        head = head.next
        prev = prev.next
    
    return True
```

**Time**: O(n) | **Space**: O(1)

**Related**: [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)

**Pattern**: Fast & Slow Pointers + Reversal

---

### Problem 8: Intersection of Two Linked Lists

**Problem**: Find intersection node of two linked lists.

**Solution**:
```python
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None
    
    a, b = headA, headB
    
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA
    
    return a
```

**Time**: O(m + n) | **Space**: O(1)

**Related**: [Intersection of Two Linked Lists](https://leetcode.com/problems/intersection-of-two-linked-lists/)

---

### Problem 9: Reverse Nodes in k-Group

**Problem**: Reverse nodes in groups of k.

**Solution**:
```python
def reverseKGroup(head, k):
    def reverse_group(node, k):
        prev = None
        current = node
        count = 0
        
        # Check if k nodes exist
        temp = node
        for _ in range(k):
            if not temp:
                return node
            temp = temp.next
        
        # Reverse k nodes
        while count < k and current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
            count += 1
        
        # Recursively reverse next group
        if current:
            node.next = reverse_group(current, k)
        
        return prev
    
    return reverse_group(head, k)
```

**Time**: O(n) | **Space**: O(n/k) for recursion

**Related**: [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)

---

### Problem 10: Copy List with Random Pointer

**Problem**: Deep copy linked list with random pointers.

**Solution**:
```python
def copyRandomList(head):
    if not head:
        return None
    
    # Create mapping
    mapping = {}
    current = head
    
    # First pass: create nodes
    while current:
        mapping[current] = Node(current.val)
        current = current.next
    
    # Second pass: set pointers
    current = head
    while current:
        if current.next:
            mapping[current].next = mapping[current.next]
        if current.random:
            mapping[current].random = mapping[current.random]
        current = current.next
    
    return mapping[head]
```

**Time**: O(n) | **Space**: O(n)

**Related**: [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)

---

## Advanced Patterns

### 1. Flatten Multilevel Doubly Linked List

**Problem**: Flatten multilevel doubly linked list.

```python
def flatten(head):
    if not head:
        return head
    
    current = head
    while current:
        if current.child:
            next_node = current.next
            child_head = flatten(current.child)
            current.next = child_head
            child_head.prev = current
            current.child = None
            
            while current.next:
                current = current.next
            current.next = next_node
            if next_node:
                next_node.prev = current
        current = current.next
    
    return head
```

---

### 2. LRU Cache (Using Doubly Linked List)

**Pattern**: Combine hash map with doubly linked list.

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        self._remove_node(node)
        self._add_node(node)
    
    def get(self, key):
        node = self.cache.get(key)
        if not node:
            return -1
        self._move_to_head(node)
        return node.value
    
    def put(self, key, value):
        node = self.cache.get(key)
        if not node:
            new_node = Node(key, value)
            if len(self.cache) >= self.capacity:
                tail = self.tail.prev
                self._remove_node(tail)
                del self.cache[tail.key]
            self.cache[key] = new_node
            self._add_node(new_node)
        else:
            node.value = value
            self._move_to_head(node)
```

---

## Key Takeaways

- Linked lists provide dynamic size and efficient insertion/deletion
- Fast & slow pointers for cycle detection and finding middle
- Dummy nodes simplify edge case handling
- Reversal is common pattern - learn iterative and recursive
- Two pointers at different positions for various problems
- Practice pointer manipulation carefully
- Draw diagrams to visualize pointer movements
- Handle edge cases: empty list, single node, null pointers

---

## Practice Problems

**Easy**:
- [Delete Node in Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/)
- [Remove Duplicates from Sorted List](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)
- [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)

**Medium**:
- [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/)
- [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/)
- [Rotate List](https://leetcode.com/problems/rotate-list/)

**Hard**:
- [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
- [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/)
- [LFU Cache](https://leetcode.com/problems/lfu-cache/)

