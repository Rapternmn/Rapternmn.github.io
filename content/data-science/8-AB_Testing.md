+++
title = "A/B Testing & Experimental Design: Interview Q&A Guide"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 8
description = "Comprehensive guide to A/B testing, hypothesis testing, and experimental design for data science interviews. Covers experimental design, hypothesis testing, sample size calculation, statistical power, and multi-armed bandits."
+++


## 1. A/B Testing Overview

### What is A/B Testing?

**Definition**: Randomized controlled experiment comparing two variants (A and B) to determine which performs better.

**Purpose**: Make data-driven decisions about changes to products, features, or processes.

**Applications**:
- Website design changes
- Email subject lines
- Pricing strategies
- Algorithm improvements
- UI/UX changes

### Key Components

1. **Control Group (A)**: Current version
2. **Treatment Group (B)**: New version
3. **Metric**: What we're measuring (CTR, conversion rate, revenue)
4. **Hypothesis**: What we expect to happen
5. **Randomization**: Users randomly assigned to groups

---

## 2. Experimental Design

### Steps in A/B Testing

1. **Define Objective**: What are we trying to improve?
2. **Formulate Hypothesis**: Null and alternative hypotheses
3. **Choose Metric**: Primary and secondary metrics
4. **Calculate Sample Size**: How many users needed?
5. **Randomize**: Assign users to groups
6. **Run Experiment**: Collect data
7. **Analyze Results**: Statistical tests
8. **Make Decision**: Launch or iterate

### Randomization

**Why Randomize?**
- Eliminates selection bias
- Ensures groups are comparable
- Allows causal inference

**Methods**:
- **Simple Randomization**: Each user randomly assigned
- **Stratified Randomization**: Randomize within strata (e.g., by country)
- **Block Randomization**: Randomize in blocks

**Important**: Randomization must be consistent (same user always in same group).

---

### Control vs Treatment

**Control Group (A)**:
- Current version
- Baseline for comparison
- No changes

**Treatment Group (B)**:
- New version
- Contains the change being tested
- Should differ only in the variable being tested

**Isolation**: Only one variable should differ between groups.

---

## 3. Hypothesis Testing in A/B Tests

### Setting Up Hypotheses

**Null Hypothesis ($H_0$)**: No difference between A and B
```
H_0: μ_A = μ_B  or  H_0: μ_B - μ_A = 0
```

**Alternative Hypothesis ($H_1$)**: There is a difference
```
H_1: μ_A ≠ μ_B  (two-tailed)
H_1: μ_B > μ_A  (one-tailed)
```

### Statistical Tests

#### T-Test (Two-Sample)

**Use Case**: Comparing means of two groups.

**Assumptions**:
- Normally distributed (or large sample)
- Independent samples
- Equal variances (or use Welch's t-test)

**Test Statistic**:
```
t = (X̄_B - X̄_A) / √((s_A²/n_A) + (s_B²/n_B))
```

#### Z-Test

**Use Case**: Large samples (n > 30) or known population variance.

**Test Statistic**:
```
z = (X̄_B - X̄_A) / √((σ_A²/n_A) + (σ_B²/n_B))
```

#### Chi-Square Test

**Use Case**: Comparing proportions (conversion rates).

**Example**: Conversion rate A vs B

#### Mann-Whitney U Test

**Use Case**: Non-parametric alternative to t-test.

**When to Use**:
- Non-normal data
- Small samples
- Ordinal data

---

### P-Value Interpretation

**P-value**: Probability of observing data as extreme (or more extreme) if $H_0$ is true.

**Decision Rule**:
- **p < α (0.05)**: Reject $H_0$ → Significant difference
- **p ≥ α (0.05)**: Fail to reject $H_0$ → No significant difference

**Common Misconceptions**:
- ❌ P-value = probability $H_0$ is true
- ❌ P-value = effect size
- ✅ P-value = probability of data given $H_0$

---

## 4. Sample Size Calculation

### Why Calculate Sample Size?

- **Too small**: Low power, may miss real effects
- **Too large**: Wastes resources, takes longer

### Factors Affecting Sample Size

1. **Effect Size**: Minimum detectable difference
2. **Significance Level (α)**: Type I error rate (typically 0.05)
3. **Power (1-β)**: Probability of detecting effect (typically 0.80)
4. **Variance**: Higher variance → larger sample needed
5. **Baseline Rate**: For proportion metrics

### Formula for Two Proportions

**Sample size per group**:
```
n = ((Z_(α/2) + Z_β)² × (p_A(1-p_A) + p_B(1-p_B))) / (p_B - p_A)²
```

Where:
- $p_A$: Baseline conversion rate
- $p_B$: Expected conversion rate
- $Z_{\alpha/2}$: Critical value (1.96 for α=0.05)
- $Z_{\beta}$: Power value (0.84 for 80% power)

### Example

**Scenario**: 
- Baseline conversion: 2% ($p_A = 0.02$)
- Minimum lift: 20% ($p_B = 0.024$)
- α = 0.05, Power = 80%

**Calculation**: ~15,000 users per group

---

## 5. Statistical Power

### Definition

**Power (1-β)**: Probability of correctly rejecting $H_0$ when it's false.

**Interpretation**: If there's a real effect, power is the chance we'll detect it.

### Power Analysis

**Components**:
- **Effect Size**: How big is the difference?
- **Sample Size**: How many observations?
- **Significance Level**: α (Type I error)
- **Power**: 1-β (typically 0.80)

**Relationship**: 
- Larger effect → Higher power
- Larger sample → Higher power
- Lower α → Lower power

### Increasing Power

1. **Increase sample size**: Most direct
2. **Increase effect size**: Make change more impactful
3. **Increase α**: Trade-off (more false positives)
4. **Reduce variance**: Better measurement, stratification

---

## 6. Common Pitfalls

### 1. Multiple Testing Problem

**Problem**: Testing multiple hypotheses increases false positive rate.

**Example**: Testing 20 metrics at α=0.05 → ~64% chance of at least one false positive.

**Solutions**:
- **Bonferroni Correction**: Divide α by number of tests
- **False Discovery Rate (FDR)**: Control expected proportion of false discoveries
- **Pre-specify primary metric**: Only one primary hypothesis

### 2. Peeking / Early Stopping

**Problem**: Checking results early and stopping when significant.

**Why Bad**: Inflates Type I error rate.

**Solution**: Pre-specify sample size and duration, don't peek.

### 3. Selection Bias

**Problem**: Non-random assignment or selection.

**Examples**:
- Users self-select into groups
- Different user segments in groups
- Time-based bias

**Solution**: Proper randomization, stratification.

### 4. Novelty Effect

**Problem**: Users react to novelty, not actual improvement.

**Solution**: Run test long enough for novelty to wear off.

### 5. Simpson's Paradox

**Problem**: Aggregated results show opposite trend from subgroups.

**Example**: Overall conversion higher in A, but higher in B for each segment.

**Solution**: Stratified analysis, check subgroups.

### 6. Insufficient Sample Size

**Problem**: Too few users → low power → miss real effects.

**Solution**: Calculate sample size beforehand.

### 7. Metric Selection

**Problem**: Optimizing wrong metric.

**Example**: Optimizing clicks instead of conversions.

**Solution**: Align metrics with business goals.

---

## 7. Advanced Topics

### Sequential Testing

**Definition**: Analyze data as it arrives, stop when significant.

**Methods**:
- **Sequential Probability Ratio Test (SPRT)**
- **Group Sequential Design**

**Advantages**: Can stop early, save resources.

**Challenges**: More complex, requires careful design.

### Multi-Variant Testing (A/B/n)

**Definition**: Testing more than two variants simultaneously.

**Example**: A/B/C test (control + 2 treatments).

**Considerations**:
- Larger sample size needed
- Multiple comparisons correction
- More complex analysis

### Factorial Design

**Definition**: Test multiple factors simultaneously.

**Example**: Test button color AND button text.

**Advantages**: 
- Test interactions
- More efficient
- Understand factor effects

**Challenges**: More complex, larger sample size.

---

## 8. Multi-Armed Bandits

### What are Multi-Armed Bandits?

**Definition**: Adaptive experimentation that balances exploration and exploitation.

**Key Difference from A/B Testing**:
- **A/B Testing**: Fixed allocation (50/50)
- **Bandits**: Dynamic allocation (more traffic to better variant)

### Algorithms

#### 1. Epsilon-Greedy

- **ε% of time**: Explore (random)
- **(1-ε)% of time**: Exploit (best variant)

#### 2. Upper Confidence Bound (UCB)

- Choose variant with highest upper confidence bound
- Balances exploration and exploitation

#### 3. Thompson Sampling

- Bayesian approach
- Sample from posterior, choose best

### When to Use Bandits

**Advantages**:
- Faster convergence
- Less wasted traffic
- Adaptive

**Disadvantages**:
- More complex
- Harder to analyze
- May converge to suboptimal

**Use Cases**:
- When exploration cost is high
- When you need quick decisions
- When variants are similar

---

## 9. Best Practices

### Before the Test

1. **Clear hypothesis**: What are you testing?
2. **Right metric**: Aligned with business goals
3. **Sample size**: Calculate beforehand
4. **Duration**: Long enough for statistical significance
5. **One variable**: Only test one change at a time

### During the Test

1. **No peeking**: Don't check results early
2. **Monitor**: Check for technical issues
3. **Consistency**: Keep test conditions stable
4. **Documentation**: Record any issues

### After the Test

1. **Statistical significance**: Check p-value
2. **Effect size**: Is it practically significant?
3. **Subgroup analysis**: Check different segments
4. **Documentation**: Record learnings
5. **Replication**: Validate findings

### Reporting Results

**Include**:
- Hypothesis
- Sample sizes
- Duration
- Statistical test used
- P-value
- Effect size
- Confidence intervals
- Business impact

---

## Quick Reference

### Sample Size Calculator Inputs

- Baseline conversion rate
- Minimum detectable effect
- Significance level (α = 0.05)
- Power (1-β = 0.80)

### Decision Framework

1. **p < 0.05 AND effect size meaningful**: Launch
2. **p < 0.05 BUT effect size small**: Consider business value
3. **p ≥ 0.05**: No significant difference, iterate or abandon

### Common Metrics

- **Conversion Rate**: Purchases / Visitors
- **Click-Through Rate (CTR)**: Clicks / Impressions
- **Revenue per User**: Total Revenue / Users
- **Engagement**: Time spent, actions taken

---

*Last Updated: 2024*

