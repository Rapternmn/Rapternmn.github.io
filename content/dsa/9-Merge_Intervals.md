+++
title = "Merge Intervals"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 9
description = "Complete guide to Merge Intervals pattern with templates in C++ and Python. Covers merging overlapping intervals, inserting intervals, interval intersections, and scheduling problems with LeetCode problem references."
+++

---

## Introduction

The Merge Intervals pattern is used to solve problems involving intervals, ranges, or time periods. It typically involves sorting intervals and then merging or processing overlapping intervals efficiently.

This guide provides templates and patterns for Merge Intervals with references to LeetCode problems for practice.

---

## Pattern Overview

### When to Use Merge Intervals

- **Overlapping intervals**: Merge or find overlapping intervals
- **Interval scheduling**: Meeting rooms, event scheduling
- **Range problems**: Insert intervals, remove intervals
- **Time-based problems**: Calendar conflicts, availability
- **Coverage problems**: Minimum intervals to cover range

### Time & Space Complexity

- **Time Complexity**: O(n log n) - sorting intervals + O(n) processing = O(n log n)
- **Space Complexity**: O(n) - result array, or O(1) if modifying in-place

---

## Pattern Variations

### Variation 1: Merge Overlapping Intervals

Combine intervals that overlap or are adjacent.

**Use Cases**:
- Merge Intervals
- Insert Interval
- Non-overlapping Intervals

### Variation 2: Find Overlapping Intervals

Identify which intervals overlap.

**Use Cases**:
- Interval List Intersections
- Meeting Rooms II
- Employee Free Time

### Variation 3: Interval Scheduling

Schedule intervals with constraints.

**Use Cases**:
- Meeting Rooms
- Meeting Rooms II
- Non-overlapping Intervals

---

## Template 1: Merge Overlapping Intervals

**Key Points**:
- Sort intervals by start time
- Compare current interval with last merged interval
- Merge if overlapping, otherwise add new interval
- **Time Complexity**: O(n log n) - sorting dominates
- **Space Complexity**: O(n) for result array

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> mergeIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return {};
    
    // Sort intervals by start time
    sort(intervals.begin(), intervals.end(), 
         [](const vector<int>& a, const vector<int>& b) {
             return a[0] < b[0];
         });
    
    vector<vector<int>> merged;
    merged.push_back(intervals[0]);
    
    for (int i = 1; i < intervals.size(); i++) {
        vector<int>& current = intervals[i];
        vector<int>& last = merged.back();
        
        // Check if current overlaps with last merged interval
        if (current[0] <= last[1]) {
            // Merge: update end time
            last[1] = max(last[1], current[1]);
        } else {
            // No overlap: add new interval
            merged.push_back(current);
        }
    }
    
    return merged;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if current overlaps with last merged interval
        if current[0] <= last[1]:
            # Merge: update end time
            last[1] = max(last[1], current[1])
        else:
            # No overlap: add new interval
            merged.append(current)
    
    return merged
```

</details>

### Related Problems

- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)
- [Insert Interval](https://leetcode.com/problems/insert-interval/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)

---

## Template 2: Insert Interval

**Key Points**:
- Find position to insert new interval
- Merge overlapping intervals
- Handle edge cases (before all, after all, in between)
- **Time Complexity**: O(n) - single pass through intervals
- **Space Complexity**: O(n) for result array

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> insertInterval(vector<vector<int>>& intervals, vector<int>& newInterval) {
    vector<vector<int>> result;
    int i = 0;
    int n = intervals.size();
    
    // Add all intervals before newInterval
    while (i < n && intervals[i][1] < newInterval[0]) {
        result.push_back(intervals[i]);
        i++;
    }
    
    // Merge overlapping intervals
    while (i < n && intervals[i][0] <= newInterval[1]) {
        newInterval[0] = min(newInterval[0], intervals[i][0]);
        newInterval[1] = max(newInterval[1], intervals[i][1]);
        i++;
    }
    
    // Add merged interval
    result.push_back(newInterval);
    
    // Add remaining intervals
    while (i < n) {
        result.push_back(intervals[i]);
        i++;
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def insert_interval(intervals, new_interval):
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals before newInterval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    # Add merged interval
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result
```

</details>

### Related Problems

- [Insert Interval](https://leetcode.com/problems/insert-interval/)
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

---

## Template 3: Find Interval Intersections

**Key Points**:
- Use two pointers for two interval lists
- Find intersection of current intervals
- Move pointer of interval that ends earlier
- **Time Complexity**: O(m + n) where m and n are sizes of two lists
- **Space Complexity**: O(k) where k is number of intersections

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
vector<vector<int>> intervalIntersection(vector<vector<int>>& firstList, 
                                         vector<vector<int>>& secondList) {
    vector<vector<int>> result;
    int i = 0, j = 0;
    
    while (i < firstList.size() && j < secondList.size()) {
        vector<int>& interval1 = firstList[i];
        vector<int>& interval2 = secondList[j];
        
        // Find intersection
        int start = max(interval1[0], interval2[0]);
        int end = min(interval1[1], interval2[1]);
        
        // If valid intersection, add to result
        if (start <= end) {
            result.push_back({start, end});
        }
        
        // Move pointer of interval that ends earlier
        if (interval1[1] < interval2[1]) {
            i++;
        } else {
            j++;
        }
    }
    
    return result;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def interval_intersection(first_list, second_list):
    result = []
    i = j = 0
    
    while i < len(first_list) and j < len(second_list):
        interval1 = first_list[i]
        interval2 = second_list[j]
        
        # Find intersection
        start = max(interval1[0], interval2[0])
        end = min(interval1[1], interval2[1])
        
        # If valid intersection, add to result
        if start <= end:
            result.append([start, end])
        
        # Move pointer of interval that ends earlier
        if interval1[1] < interval2[1]:
            i += 1
        else:
            j += 1
    
    return result
```

</details>

### Related Problems

- [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
- [Employee Free Time](https://leetcode.com/problems/employee-free-time/)

---

## Template 4: Meeting Rooms / Scheduling

**Key Points**:
- Sort intervals by start time
- Use priority queue (min-heap) for end times
- Track number of rooms needed
- **Time Complexity**: O(n log n) - sorting + heap operations
- **Space Complexity**: O(n) for heap

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
// Meeting Rooms II: Minimum rooms needed
int minMeetingRooms(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    
    // Sort by start time
    sort(intervals.begin(), intervals.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[0] < b[0];
         });
    
    // Min-heap to track end times
    priority_queue<int, vector<int>, greater<int>> minHeap;
    
    for (auto& interval : intervals) {
        // If earliest ending meeting has ended, reuse that room
        if (!minHeap.empty() && minHeap.top() <= interval[0]) {
            minHeap.pop();
        }
        
        // Add current meeting's end time
        minHeap.push(interval[1]);
    }
    
    return minHeap.size();
}

// Meeting Rooms: Check if can attend all
bool canAttendMeetings(vector<vector<int>>& intervals) {
    if (intervals.empty()) return true;
    
    sort(intervals.begin(), intervals.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[0] < b[0];
         });
    
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] < intervals[i-1][1]) {
            return false; // Overlapping intervals
        }
    }
    
    return true;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
import heapq

# Meeting Rooms II: Minimum rooms needed
def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min-heap to track end times
    min_heap = []
    
    for interval in intervals:
        # If earliest ending meeting has ended, reuse that room
        if min_heap and min_heap[0] <= interval[0]:
            heapq.heappop(min_heap)
        
        # Add current meeting's end time
        heapq.heappush(min_heap, interval[1])
    
    return len(min_heap)

# Meeting Rooms: Check if can attend all
def can_attend_meetings(intervals):
    if not intervals:
        return True
    
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False  # Overlapping intervals
    
    return True
```

</details>

### Related Problems

- [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
- [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

---

## Template 5: Remove Overlapping Intervals

**Key Points**:
- Sort intervals by end time (greedy: keep intervals ending earliest)
- Track last end time
- Remove intervals that start before last end
- **Time Complexity**: O(n log n) - sorting dominates
- **Space Complexity**: O(1) - only tracking variables

<details open>
<summary><strong>📋 C++ Template</strong></summary>

```cpp
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    if (intervals.empty()) return 0;
    
    // Sort by end time (greedy: keep intervals ending earliest)
    sort(intervals.begin(), intervals.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[1] < b[1];
         });
    
    int count = 0;
    int lastEnd = intervals[0][1];
    
    for (int i = 1; i < intervals.size(); i++) {
        if (intervals[i][0] < lastEnd) {
            // Overlapping: remove this interval
            count++;
        } else {
            // Non-overlapping: update last end
            lastEnd = intervals[i][1];
        }
    }
    
    return count;
}
```

</details>

<details>
<summary><strong>🐍 Python Template</strong></summary>

```python
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    
    # Sort by end time (greedy: keep intervals ending earliest)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < last_end:
            # Overlapping: remove this interval
            count += 1
        else:
            # Non-overlapping: update last end
            last_end = intervals[i][1]
    
    return count
```

</details>

### Related Problems

- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)
- [Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/)

---

## Key Takeaways

1. **Sort First**: Almost always sort intervals by start or end time
2. **Greedy Approach**: For scheduling, keep intervals ending earliest
3. **Overlap Check**: `interval1[0] <= interval2[1] && interval2[0] <= interval1[1]`
4. **Merge Condition**: `current[0] <= last[1]` means overlap
5. **Two Pointers**: Use for finding intersections between two interval lists
6. **Heap for Scheduling**: Use min-heap to track end times for room allocation

---

## Common Mistakes

1. **Not sorting**: Forgetting to sort intervals first
2. **Wrong sort key**: Sorting by start vs end matters for different problems
3. **Overlap condition**: Incorrect overlap checking logic
4. **Index errors**: Off-by-one errors when accessing interval elements
5. **Edge cases**: Not handling empty intervals, single interval

---

## Practice Problems by Difficulty

### Easy
- [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/)
- [Merge Intervals](https://leetcode.com/problems/merge-intervals/)

### Medium
- [Insert Interval](https://leetcode.com/problems/insert-interval/)
- [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
- [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/)
- [Interval List Intersections](https://leetcode.com/problems/interval-list-intersections/)
- [Remove Covered Intervals](https://leetcode.com/problems/remove-covered-intervals/)
- [Minimum Number of Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

### Hard
- [Employee Free Time](https://leetcode.com/problems/employee-free-time/)

---

## References

* [LeetCode Array Tag](https://leetcode.com/tag/array/) - Many interval problems
* [Interval Scheduling (Wikipedia)](https://en.wikipedia.org/wiki/Interval_scheduling)

