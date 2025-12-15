+++
title = "Fast & Slow Pointers (Floyd's Cycle Detection)"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 5
description = "Complete guide to Fast & Slow Pointers pattern (Floyd's Cycle Detection) with templates in C++ and Python. Covers cycle detection, finding cycle start, middle element, and palindrome detection with LeetCode problem references."
+++

---

## Introduction

The Fast & Slow Pointers technique (also known as Floyd's Cycle Detection or the "Tortoise and Hare" algorithm) uses two pointers moving at different speeds to solve problems efficiently. The slow pointer moves one step at a time, while the fast pointer moves two steps, enabling cycle detection and other optimizations.

This guide provides templates and patterns for Fast & Slow Pointers with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Fast & Slow Pointers

- **Cycle detection**: Detect cycles in linked lists or arrays
- **Find cycle start**: Locate the starting node of a cycle
- **Find middle element**: Find the middle of a linked list
- **Palindrome detection**: Check if linked list is palindrome
- **Find kth element from end**: Find kth node from the end
- **Intersection of linked lists**: Find intersection point

### Time & Space Complexity

- **Time Complexity**: O(n) - single pass through the structure
- **Space Complexity**: O(1) - only using two pointer variables

---

## Pattern Variations

### Variation 1: Cycle Detection

Detect if a cycle exists in a linked list.

**Use Cases**:
- Linked List Cycle
- Detect cycle in array (using indices as pointers)

### Variation 2: Find Cycle Start

Find the node where the cycle begins.

**Use Cases**:
- Linked List Cycle II
- Find duplicate number (array as linked list)

### Variation 3: Find Middle Element

Find the middle element of a linked list.

**Use Cases**:
- Middle of the Linked List
- Split linked list in half

### Variation 4: Palindrome Detection

Check if a linked list is a palindrome.

**Use Cases**:
- Palindrome Linked List
- Check symmetry in linked list

---

## Template 1: Cycle Detection

**Key Points**:
- Slow pointer moves one step, fast pointer moves two steps
- If there's a cycle, fast and slow will eventually meet
- If fast reaches null, there's no cycle
- **Time Complexity**: O(n) - fast pointer visits each node at most once
- **Space Complexity**: O(1) - only two pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

bool hasCycle(ListNode* head) {
    if (head == nullptr || head->next == nullptr) {
        return false;
    }
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;        // Move one step
        fast = fast->next->next;  // Move two steps
        
        if (slow == fast) {
            return true; // Cycle detected
        }
    }
    
    return false; // No cycle
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

def has_cycle(head):
    if head is None or head.next is None:
        return False
    
    slow = head
    fast = head
    
    while fast is not None and fast.next is not None:
        slow = slow.next        # Move one step
        fast = fast.next.next   # Move two steps
        
        if slow == fast:
            return True  # Cycle detected
    
    return False  # No cycle
```

</details>

### Related Problems

- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

---

## Template 2: Find Cycle Start

**Key Points**:
- First detect cycle using fast/slow pointers
- Reset slow to head, keep fast at meeting point
- Move both one step at a time
- They will meet at cycle start
- **Time Complexity**: O(n) - linear time to find cycle and cycle start
- **Space Complexity**: O(1) - only two pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* detectCycle(ListNode* head) {
    if (head == nullptr || head->next == nullptr) {
        return nullptr;
    }
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    // Detect cycle
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
        
        if (slow == fast) {
            break; // Cycle detected
        }
    }
    
    // No cycle
    if (fast == nullptr || fast->next == nullptr) {
        return nullptr;
    }
    
    // Find cycle start
    slow = head;
    while (slow != fast) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow; // Cycle start
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def detect_cycle(head):
    if head is None or head.next is None:
        return None
    
    slow = head
    fast = head
    
    # Detect cycle
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            break  # Cycle detected
    
    # No cycle
    if fast is None or fast.next is None:
        return None
    
    # Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # Cycle start
```

</details>

### Related Problems

- [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)

---

## Template 3: Find Middle Element

**Key Points**:
- Slow pointer moves one step, fast pointer moves two steps
- When fast reaches end, slow is at middle
- Handle even/odd length lists appropriately
- **Time Complexity**: O(n) - single pass through list
- **Space Complexity**: O(1) - only two pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* findMiddle(ListNode* head) {
    if (head == nullptr) {
        return nullptr;
    }
    
    ListNode* slow = head;
    ListNode* fast = head;
    
    // For odd length: fast->next == nullptr
    // For even length: fast == nullptr
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return slow; // Middle node
}

// Alternative: Return first middle for even length
ListNode* findMiddleFirst(ListNode* head) {
    if (head == nullptr) {
        return nullptr;
    }
    
    ListNode* slow = head;
    ListNode* fast = head->next; // Start fast one step ahead
    
    while (fast != nullptr && fast->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    
    return slow; // First middle for even length
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def find_middle(head):
    if head is None:
        return None
    
    slow = head
    fast = head
    
    # For odd length: fast.next == None
    # For even length: fast == None
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # Middle node

# Alternative: Return first middle for even length
def find_middle_first(head):
    if head is None:
        return None
    
    slow = head
    fast = head.next  # Start fast one step ahead
    
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    
    return slow  # First middle for even length
```

</details>

### Related Problems

- [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- [Reorder List](https://leetcode.com/problems/reorder-list/)
- [Delete the Middle Node of a Linked List](https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/)

---

## Template 4: Palindrome Linked List

**Key Points**:
- Find middle using fast/slow pointers
- Reverse second half
- Compare first half with reversed second half
- Restore list if needed
- **Time Complexity**: O(n) - find middle O(n), reverse O(n), compare O(n)
- **Space Complexity**: O(1) - only pointer variables (excluding recursion stack)

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
bool isPalindrome(ListNode* head) {
    if (head == nullptr || head->next == nullptr) {
        return true;
    }
    
    // Find middle
    ListNode* slow = head;
    ListNode* fast = head;
    
    while (fast->next != nullptr && fast->next->next != nullptr) {
        slow = slow->next;
        fast = fast->next->next;
    }
    
    // Reverse second half
    ListNode* secondHalf = reverseList(slow->next);
    ListNode* firstHalf = head;
    
    // Compare
    bool isPal = true;
    ListNode* secondHalfCopy = secondHalf;
    while (secondHalf != nullptr) {
        if (firstHalf->val != secondHalf->val) {
            isPal = false;
            break;
        }
        firstHalf = firstHalf->next;
        secondHalf = secondHalf->next;
    }
    
    // Restore list (optional)
    slow->next = reverseList(secondHalfCopy);
    
    return isPal;
}

ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* curr = head;
    
    while (curr != nullptr) {
        ListNode* next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    
    return prev;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def is_palindrome(head):
    if head is None or head.next is None:
        return True
    
    # Find middle
    slow = head
    fast = head
    
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    
    # Reverse second half
    second_half = reverse_list(slow.next)
    first_half = head
    
    # Compare
    is_pal = True
    second_half_copy = second_half
    while second_half is not None:
        if first_half.val != second_half.val:
            is_pal = False
            break
        first_half = first_half.next
        second_half = second_half.next
    
    # Restore list (optional)
    slow.next = reverse_list(second_half_copy)
    
    return is_pal

def reverse_list(head):
    prev = None
    curr = head
    
    while curr is not None:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    
    return prev
```

</details>

### Related Problems

- [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
- [Reorder List](https://leetcode.com/problems/reorder-list/)

---

## Template 5: Find kth Node from End

**Key Points**:
- Move fast pointer k steps ahead
- Then move both pointers one step at a time
- When fast reaches end, slow is at kth from end
- **Time Complexity**: O(n) - single pass through list
- **Space Complexity**: O(1) - only two pointer variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
ListNode* findKthFromEnd(ListNode* head, int k) {
    ListNode* slow = head;
    ListNode* fast = head;
    
    // Move fast k steps ahead
    for (int i = 0; i < k; i++) {
        if (fast == nullptr) {
            return nullptr; // List shorter than k
        }
        fast = fast->next;
    }
    
    // Move both until fast reaches end
    while (fast != nullptr) {
        slow = slow->next;
        fast = fast->next;
    }
    
    return slow; // kth node from end
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def find_kth_from_end(head, k):
    slow = head
    fast = head
    
    # Move fast k steps ahead
    for i in range(k):
        if fast is None:
            return None  # List shorter than k
        fast = fast.next
    
    # Move both until fast reaches end
    while fast is not None:
        slow = slow.next
        fast = fast.next
    
    return slow  # kth node from end
```

</details>

### Related Problems

- [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)
- [Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/)

---

## Key Takeaways

1. **Cycle Detection**: Fast and slow will meet if cycle exists
2. **Cycle Start**: After meeting, reset slow to head, move both one step
3. **Middle Element**: When fast reaches end, slow is at middle
4. **Palindrome**: Find middle, reverse second half, compare
5. **kth from End**: Move fast k steps ahead, then move both together
6. **Null Checks**: Always check for null before accessing next

---

## Common Mistakes

1. **Not checking null**: Accessing `next` without null check
2. **Wrong pointer movement**: Not moving fast pointer correctly
3. **Infinite loops**: Not handling cycle detection properly
4. **Off-by-one errors**: In middle element or kth from end
5. **Not restoring list**: In palindrome, forgetting to restore reversed half

---

## Practice Problems by Difficulty

### Easy
- [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/)
- [Middle of the Linked List](https://leetcode.com/problems/middle-of-the-linked-list/)
- [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

### Medium
- [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/)
- [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/)
- [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/)
- [Reorder List](https://leetcode.com/problems/reorder-list/)
- [Swapping Nodes in a Linked List](https://leetcode.com/problems/swapping-nodes-in-a-linked-list/)
- [Delete the Middle Node of a Linked List](https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/)

---

## References

* [LeetCode Linked List Tag](https://leetcode.com/tag/linked-list/)
* [Floyd's Cycle Detection Algorithm](https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_tortoise_and_hare)

