+++
title = "Data Science Interview Math Problems"
date = 2025-12-06T10:00:00+05:30
draft = false
weight = 8
description = "Comprehensive collection of mathematical problems commonly asked in Data Science interviews. Covers probability, permutations & combinations, coin toss, dice problems, conditional probability, Bayes' theorem, expected value, and statistical reasoning."
+++

---

## Introduction

Data Science interviews often include mathematical problems to test your understanding of probability, statistics, and logical reasoning. This guide covers common problem types with detailed solutions.

**Key Topics**:
- Probability fundamentals
- Permutations & Combinations
- Coin toss problems
- Dice problems
- Conditional probability
- Bayes' theorem
- Expected value
- Sampling problems

---

## 1. Probability Fundamentals

### Problem 1: Basic Probability

**Question**: A bag contains 5 red balls, 3 blue balls, and 2 green balls. What is the probability of drawing a red ball?

**Solution**:
```
Total balls = 5 + 3 + 2 = 10
Red balls = 5
P(Red) = 5/10 = 0.5 = 50%
```

### Problem 2: Multiple Events (AND)

**Question**: What is the probability of drawing 2 red balls in a row (without replacement)?

**Solution**:
```
P(Red1 and Red2) = P(Red1) × P(Red2 | Red1)
                 = (5/10) × (4/9)
                 = 20/90 = 2/9 ≈ 0.222
```

### Problem 3: Multiple Events (OR)

**Question**: What is the probability of drawing a red OR blue ball?

**Solution**:
```
P(Red or Blue) = P(Red) + P(Blue) - P(Red and Blue)
               = 5/10 + 3/10 - 0  (mutually exclusive)
               = 8/10 = 0.8
```

---

## 2. Permutations & Combinations

### Permutations (Order Matters)

**Formula**: P(n, r) = n! / (n - r)!

**Question**: How many ways can you arrange 5 books on a shelf?

**Solution**:
```
P(5, 5) = 5! = 5 × 4 × 3 × 2 × 1 = 120 ways
```

### Combinations (Order Doesn't Matter)

**Formula**: C(n, r) = n! / (r! × (n - r)!)

**Question**: How many ways can you choose 3 books from 5?

**Solution**:
```
C(5, 3) = 5! / (3! × 2!) = 120 / (6 × 2) = 10 ways
```

### Problem: Committee Selection

**Question**: From 10 people, how many ways can you form a committee of 4?

**Solution**:
```
C(10, 4) = 10! / (4! × 6!) = 210 ways
```

### Problem: Arrangements with Restrictions

**Question**: How many ways can 5 people sit in a row if 2 specific people must sit together?

**Solution**:
```
Treat the 2 people as 1 unit: 4 units total
Ways to arrange 4 units: 4! = 24
Ways to arrange the 2 people within their unit: 2! = 2
Total: 24 × 2 = 48 ways
```

---

## 3. Coin Toss Problems

### Problem 1: Single Coin Toss

**Question**: What is the probability of getting heads in a fair coin toss?

**Solution**:
```
P(Heads) = 1/2 = 0.5
```

### Problem 2: Multiple Coin Tosses

**Question**: What is the probability of getting exactly 2 heads in 3 coin tosses?

**Solution**:
```
Using Binomial Distribution:
P(X = k) = C(n, k) × p^k × (1-p)^(n-k)

n = 3, k = 2, p = 0.5
P(X = 2) = C(3, 2) × (0.5)^2 × (0.5)^1
         = 3 × 0.25 × 0.5
         = 0.375 = 37.5%
```

### Problem 3: At Least K Heads

**Question**: What is the probability of getting at least 2 heads in 4 coin tosses?

**Solution**:
```
P(X ≥ 2) = P(X = 2) + P(X = 3) + P(X = 4)

P(X = 2) = C(4, 2) × (0.5)^4 = 6 × 0.0625 = 0.375
P(X = 3) = C(4, 3) × (0.5)^4 = 4 × 0.0625 = 0.25
P(X = 4) = C(4, 4) × (0.5)^4 = 1 × 0.0625 = 0.0625

P(X ≥ 2) = 0.375 + 0.25 + 0.0625 = 0.6875
```

### Problem 4: Consecutive Heads

**Question**: What is the probability of getting 3 consecutive heads in 5 tosses?

**Solution**:
```
Possible positions for 3 consecutive heads:
- Positions 1-3: HHHXX (2^2 = 4 ways for remaining)
- Positions 2-4: THHHT (1 way)
- Positions 3-5: XXHHH (2^2 = 4 ways)

But we need to subtract overlaps:
- HHHHX: counted in both 1-3 and 2-4
- XHHHH: counted in both 2-4 and 3-5
- HHHHH: counted in all three

Total: 4 + 1 + 4 - 2 - 2 + 1 = 6 ways
Probability: 6 / 32 = 0.1875
```

### Problem 5: First Head on K-th Toss

**Question**: What is the probability that the first head occurs on the 3rd toss?

**Solution**:
```
Sequence: T, T, H
P = (1/2) × (1/2) × (1/2) = 1/8 = 0.125

This follows Geometric Distribution:
P(X = k) = (1-p)^(k-1) × p
```

---

## 4. Dice Problems

### Problem 1: Single Die

**Question**: What is the probability of rolling a 6 on a fair die?

**Solution**:
```
P(6) = 1/6 ≈ 0.167
```

### Problem 2: Sum of Two Dice

**Question**: What is the probability of rolling a sum of 7 with two dice?

**Solution**:
```
Favorable outcomes: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 ways
Total outcomes: 6 × 6 = 36
P(Sum = 7) = 6/36 = 1/6 ≈ 0.167
```

### Problem 3: At Least One Six

**Question**: What is the probability of getting at least one 6 when rolling two dice?

**Solution**:
```
P(At least one 6) = 1 - P(No 6s)
                  = 1 - (5/6) × (5/6)
                  = 1 - 25/36
                  = 11/36 ≈ 0.306
```

### Problem 4: Specific Sequence

**Question**: What is the probability of rolling (1, 2, 3) in that order with three dice?

**Solution**:
```
P = (1/6) × (1/6) × (1/6) = 1/216 ≈ 0.0046
```

---

## 5. Conditional Probability

### Problem 1: Basic Conditional Probability

**Question**: In a deck of 52 cards, what is the probability of drawing a king given that you've drawn a face card?

**Solution**:
```
P(King | Face Card) = P(King and Face Card) / P(Face Card)
                    = P(King) / P(Face Card)
                    = (4/52) / (12/52)
                    = 4/12 = 1/3 ≈ 0.333
```

### Problem 2: Disease Testing

**Question**: A disease affects 1% of the population. A test is 99% accurate (99% true positive, 99% true negative). If someone tests positive, what is the probability they have the disease?

**Solution**:
```
Using Bayes' Theorem:
P(Disease | Positive) = P(Positive | Disease) × P(Disease) / P(Positive)

P(Disease) = 0.01
P(Positive | Disease) = 0.99
P(Positive | No Disease) = 0.01

P(Positive) = P(Positive | Disease) × P(Disease) + 
              P(Positive | No Disease) × P(No Disease)
           = 0.99 × 0.01 + 0.01 × 0.99
           = 0.0099 + 0.0099 = 0.0198

P(Disease | Positive) = (0.99 × 0.01) / 0.0198
                       = 0.0099 / 0.0198
                       ≈ 0.5 = 50%
```

### Problem 3: Monty Hall Problem

**Question**: You're on a game show with 3 doors. Behind one is a car, behind the other two are goats. You pick door 1. The host opens door 3, revealing a goat. Should you switch to door 2?

**Solution**:
```
Initial probability: P(Car behind door 1) = 1/3

If you don't switch:
P(Win) = 1/3

If you switch:
P(Win) = P(Car behind door 2 | Host opened door 3)
       = P(Car behind door 2) = 2/3

You should switch! Probability increases from 1/3 to 2/3.
```

---

## 6. Bayes' Theorem Problems

### Problem 1: Spam Detection

**Question**: 10% of emails are spam. Spam emails contain "free" 80% of the time. Non-spam emails contain "free" 10% of the time. If an email contains "free", what's the probability it's spam?

**Solution**:
```
P(Spam) = 0.1
P(Free | Spam) = 0.8
P(Free | Not Spam) = 0.1

P(Free) = P(Free | Spam) × P(Spam) + P(Free | Not Spam) × P(Not Spam)
        = 0.8 × 0.1 + 0.1 × 0.9
        = 0.08 + 0.09 = 0.17

P(Spam | Free) = P(Free | Spam) × P(Spam) / P(Free)
               = (0.8 × 0.1) / 0.17
               = 0.08 / 0.17
               ≈ 0.471 = 47.1%
```

### Problem 2: Multiple Tests

**Question**: Same disease test as before, but now the person tests positive twice (independent tests). What's the probability they have the disease?

**Solution**:
```
P(Disease) = 0.01
P(Positive | Disease) = 0.99
P(Positive | No Disease) = 0.01

P(Two Positives | Disease) = 0.99² = 0.9801
P(Two Positives | No Disease) = 0.01² = 0.0001

P(Two Positives) = 0.9801 × 0.01 + 0.0001 × 0.99
                 = 0.009801 + 0.000099
                 = 0.0099

P(Disease | Two Positives) = (0.9801 × 0.01) / 0.0099
                            ≈ 0.99 = 99%
```

---

## 7. Expected Value

### Problem 1: Dice Expected Value

**Question**: What is the expected value of a single die roll?

**Solution**:
```
E[X] = Σ(x × P(x))
     = 1×(1/6) + 2×(1/6) + 3×(1/6) + 4×(1/6) + 5×(1/6) + 6×(1/6)
     = (1+2+3+4+5+6)/6
     = 21/6 = 3.5
```

### Problem 2: Game Winnings

**Question**: You roll a die. If you get 1-3, you lose $2. If you get 4-5, you win $3. If you get 6, you win $10. What's your expected winnings?

**Solution**:
```
E[Winnings] = (-2)×(3/6) + 3×(2/6) + 10×(1/6)
            = -1 + 1 + 1.67
            = 1.67
```

### Problem 3: Expected Number of Tosses

**Question**: How many coin tosses are expected to get the first head?

**Solution**:
```
Geometric Distribution:
E[X] = 1/p = 1/0.5 = 2 tosses
```

---

## 8. Sampling Problems

### Problem 1: Sampling Without Replacement

**Question**: From a deck of 52 cards, you draw 5 cards. What's the probability of getting exactly 2 aces?

**Solution**:
```
Using Hypergeometric Distribution:
P(X = k) = C(K, k) × C(N-K, n-k) / C(N, n)

N = 52 (total cards)
K = 4 (aces)
n = 5 (cards drawn)
k = 2 (aces wanted)

P(X = 2) = C(4, 2) × C(48, 3) / C(52, 5)
         = 6 × 17296 / 2598960
         ≈ 0.0399
```

### Problem 2: Sample Size Calculation

**Question**: How many samples do you need to be 95% confident that the sample mean is within 0.1 of the population mean (σ = 1)?

**Solution**:
```
For 95% confidence, z = 1.96
Margin of error = z × (σ / √n)
0.1 = 1.96 × (1 / √n)
√n = 1.96 / 0.1 = 19.6
n = 19.6² ≈ 384
```

---

## 9. Combinatorial Problems

### Problem 1: Birthday Paradox

**Question**: What's the probability that at least 2 people in a room of 23 share the same birthday?

**Solution**:
```
P(At least 2 share birthday) = 1 - P(All different birthdays)

P(All different) = (365/365) × (364/365) × ... × (343/365)
                 = 365! / (365^23 × 342!)

P(At least 2 share) ≈ 0.507 = 50.7%
```

### Problem 2: Card Problems

**Question**: What's the probability of getting a flush (5 cards of same suit) in poker?

**Solution**:
```
Total ways to choose 5 cards: C(52, 5) = 2,598,960

Ways to get flush:
- Choose suit: 4 ways
- Choose 5 cards from that suit: C(13, 5) = 1,287
- Total: 4 × 1,287 = 5,148

But subtract straight flushes: 40
Flushes: 5,148 - 40 = 5,108

P(Flush) = 5,108 / 2,598,960 ≈ 0.00197
```

---

## 10. Advanced Problems

### Problem 1: Random Walk

**Question**: Starting at position 0, you take steps of +1 or -1 with equal probability. What's the probability of returning to 0 after 10 steps?

**Solution**:
```
For return to 0, need equal +1 and -1 steps.
Need 5 +1 and 5 -1 steps.

P = C(10, 5) × (0.5)^10
  = 252 × 0.000977
  ≈ 0.246
```

### Problem 2: Coupon Collector

**Question**: How many boxes of cereal do you need to buy on average to collect all 5 different toys?

**Solution**:
```
Expected value for coupon collector:
E[X] = n × (1 + 1/2 + 1/3 + ... + 1/n)
     = n × H_n (harmonic number)

For n = 5:
E[X] = 5 × (1 + 1/2 + 1/3 + 1/4 + 1/5)
     = 5 × 2.283
     ≈ 11.42 boxes
```

---

## 11. Interview Tips

### Common Problem Types

1. **Basic Probability**: Simple fractions, ratios
2. **Conditional Probability**: Given some information, what's the probability?
3. **Bayes' Theorem**: Updating probabilities with new evidence
4. **Expected Value**: Average outcome over many trials
5. **Permutations/Combinations**: Counting problems
6. **Sampling**: With/without replacement, sample size
7. **Distributions**: Binomial, Geometric, Hypergeometric

### Problem-Solving Strategy

1. **Understand the Problem**: Read carefully, identify what's being asked
2. **Define Events**: Clearly define what events you're calculating
3. **Choose Method**: Probability rules, Bayes', counting, etc.
4. **Calculate Step-by-Step**: Show your work
5. **Verify**: Check if answer makes sense

### Common Mistakes to Avoid

1. **Confusing AND/OR**: 
   - AND: Multiply probabilities
   - OR: Add (if mutually exclusive) or use inclusion-exclusion
2. **Forgetting Conditional**: P(A|B) ≠ P(A)
3. **Sampling**: With vs without replacement
4. **Independence**: Assuming independence when not true
5. **Order Matters**: Permutations vs combinations

---

## Key Takeaways

1. **Practice**: Work through many problems to build intuition
2. **Understand Concepts**: Don't just memorize formulas
3. **Visualize**: Draw diagrams, use tree diagrams
4. **Verify**: Check your calculations
5. **Think Step-by-Step**: Break complex problems into parts
6. **Common Distributions**: Know when to use Binomial, Geometric, Hypergeometric
7. **Bayes' Theorem**: Very common in interviews, practice it

---

## Practice Problems

### Easy
1. Probability of drawing a red card from a deck
2. Probability of getting heads in 3 coin tosses
3. Expected value of a die roll

### Medium
4. Probability of at least one 6 in 3 dice rolls
5. Disease testing with Bayes' theorem
6. Expected number of coin tosses for first head

### Hard
7. Monty Hall problem
8. Birthday paradox
9. Coupon collector problem
10. Random walk problems

---

## References

- "Introduction to Probability" by Blitzstein & Hwang
- "Think Bayes" by Allen Downey
- Khan Academy: Probability and Statistics
- LeetCode: Math problems section

