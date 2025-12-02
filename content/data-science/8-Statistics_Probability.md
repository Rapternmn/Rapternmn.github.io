+++
title = "Statistics & Probability"
date = 2025-11-22T10:00:00+05:30
draft = false
weight = 8
description = "Essential statistics and probability concepts. Covers probability fundamentals, distributions, central limit theorem, hypothesis testing, confidence intervals, and Bayesian statistics."
+++


## 1. Probability Fundamentals

### What is Probability?

**Probability** measures the likelihood of an event occurring, ranging from 0 (impossible) to 1 (certain).

### Key Concepts

#### Conditional Probability

**Definition**: Probability of event A given that event B has occurred.

**Formula**:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Example**: Probability of rain given it's cloudy.

#### Bayes' Theorem

**Formula**:
```
P(A|B) = (P(B|A) × P(A)) / P(B)
```

**Components**:
- $P(A|B)$: Posterior probability
- $P(B|A)$: Likelihood
- $P(A)$: Prior probability
- $P(B)$: Evidence (normalizing constant)

**Use Case**: Updating beliefs with new evidence (e.g., spam detection, medical diagnosis).

#### Independence

Events A and B are independent if:
```
P(A ∩ B) = P(A) × P(B)
```

Or equivalently:
```
P(A|B) = P(A)
```

---

## 2. Distributions

### Normal Distribution (Gaussian)

**Definition**: Bell-shaped, symmetric distribution.

**Probability Density Function**:
```
f(x) = (1 / (σ√(2π))) × e^(-½((x-μ)/σ)²)
```

**Parameters**:
- $\mu$: Mean (center)
- $\sigma$: Standard deviation (spread)

**Properties**:
- 68% of data within 1σ
- 95% of data within 2σ
- 99.7% of data within 3σ

**Use Cases**: Natural phenomena, measurement errors, many ML assumptions.

---

### Binomial Distribution

**Definition**: Number of successes in n independent trials.

**Probability Mass Function**:
```
P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
```

**Parameters**:
- n: Number of trials
- p: Probability of success

**Mean**: $\mu = np$
**Variance**: $\sigma^2 = np(1-p)$

**Use Cases**: Coin flips, A/B testing, quality control.

---

### Poisson Distribution

**Definition**: Number of events in a fixed interval of time/space.

**Probability Mass Function**:
```
P(X = k) = (λ^k × e^(-λ)) / k!
```

**Parameters**:
- $\lambda$: Average rate of events

**Properties**:
- Mean = Variance = $\lambda$
- Memoryless

**Use Cases**: Number of emails per hour, website visits, defects in manufacturing.

---

### Exponential Distribution

**Definition**: Time between events in a Poisson process.

**Probability Density Function**:
```
f(x) = λ × e^(-λx), for x ≥ 0
```

**Parameters**:
- $\lambda$: Rate parameter

**Mean**: $\mu = \frac{1}{\lambda}$
**Variance**: $\sigma^2 = \frac{1}{\lambda^2}$

**Use Cases**: Time between arrivals, failure times, waiting times.

---

## 3. Central Limit Theorem

### Statement

**CLT**: As sample size increases, the distribution of sample means approaches a normal distribution, regardless of the original population distribution.

**Mathematical Formulation**:
```
X̄ ~ N(μ, σ²/n)
```

As $n \to \infty$, where:
- $\bar{X}$: Sample mean
- $\mu$: Population mean
- $\sigma^2$: Population variance
- n: Sample size

### Key Points

1. **Works for any distribution**: Original distribution doesn't matter
2. **Requires large n**: Typically n ≥ 30
3. **Sample means are normally distributed**: Even if original data isn't
4. **Standard error decreases**: $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$

### Applications

- Confidence intervals
- Hypothesis testing
- Sampling distributions
- Justifies use of normal distribution in many statistical tests

---

## 4. Hypothesis Testing

### Framework

**Null Hypothesis ($H_0$)**: Default assumption (e.g., no effect, no difference)
**Alternative Hypothesis ($H_1$)**: What we want to prove

### Steps

1. **State hypotheses**: $H_0$ and $H_1$
2. **Choose significance level**: $\alpha$ (typically 0.05)
3. **Collect data**: Sample from population
4. **Calculate test statistic**: Based on data
5. **Calculate p-value**: Probability of observing data if $H_0$ is true
6. **Make decision**: Reject $H_0$ if p-value < $\alpha$

### Type I and Type II Errors

| | Reject $H_0$ | Fail to Reject $H_0$ |
|---|---|---|
| **$H_0$ True** | Type I Error (α) | Correct |
| **$H_0$ False** | Correct | Type II Error (β) |

- **Type I Error (α)**: False positive - reject true $H_0$
- **Type II Error (β)**: False negative - fail to reject false $H_0$
- **Power (1-β)**: Probability of correctly rejecting false $H_0$

---

## 5. Confidence Intervals

### Definition

**Confidence Interval**: Range of values that likely contains the true population parameter.

**Interpretation**: "We are 95% confident that the true mean lies between [lower, upper]."

**Note**: Does NOT mean 95% probability that parameter is in interval (frequentist interpretation).

### Formula for Mean (Known σ)

```
x̄ ± z_(α/2) × (σ / √n)
```

### Formula for Mean (Unknown σ)

```
x̄ ± t_(α/2, n-1) × (s / √n)
```

Where:
- $\bar{x}$: Sample mean
- $z_{\alpha/2}$ or $t_{\alpha/2}$: Critical value
- $s$: Sample standard deviation
- $n$: Sample size

### Common Confidence Levels

- 90%: $\alpha = 0.10$
- 95%: $\alpha = 0.05$ (most common)
- 99%: $\alpha = 0.01$

---

## 6. Correlation vs Causation

### Correlation

**Definition**: Statistical relationship between two variables.

**Correlation Coefficient (r)**:
```
r = Σ((x_i - x̄)(y_i - ȳ)) / √(Σ(x_i - x̄)² × Σ(y_i - ȳ)²)
```

**Range**: -1 to +1
- **+1**: Perfect positive correlation
- **0**: No correlation
- **-1**: Perfect negative correlation

**Properties**:
- Measures linear relationship
- Symmetric: $r_{xy} = r_{yx}$
- Does NOT imply causation

### Causation

**Definition**: One variable directly causes change in another.

**Requirements for Causation**:
1. **Association**: Variables are correlated
2. **Temporal order**: Cause precedes effect
3. **No confounding**: No third variable explaining relationship
4. **Mechanism**: Plausible explanation

### Common Confounders

- **Third variable**: Hidden factor affecting both
- **Reverse causation**: Effect causes cause
- **Selection bias**: Non-random sampling
- **Simpson's paradox**: Aggregated data shows opposite trend

**Example**: Ice cream sales and drowning deaths are correlated (both increase in summer), but ice cream doesn't cause drowning.

---

## 7. Bayesian Statistics

### Bayesian vs Frequentist

| Aspect | Frequentist | Bayesian |
|---|---|---|
| **Probability** | Long-run frequency | Degree of belief |
| **Parameters** | Fixed, unknown | Random variables |
| **Prior Knowledge** | Not used | Incorporated via priors |
| **Inference** | Point estimates, CIs | Posterior distributions |

### Bayesian Inference

**Posterior** = (Likelihood × Prior) / Evidence

```
P(θ|X) = (P(X|θ) × P(θ)) / P(X)
```

Where:
- $P(\theta|X)$: Posterior (updated belief)
- $P(X|\theta)$: Likelihood (data given parameter)
- $P(\theta)$: Prior (initial belief)
- $P(X)$: Evidence (normalizing constant)

### Advantages

- Incorporates prior knowledge
- Provides probability distributions, not just point estimates
- Handles small samples better
- Natural for updating beliefs

### Use Cases

- Medical diagnosis
- Spam filtering
- Recommendation systems
- A/B testing

---

## 8. Common Statistical Tests

### T-Test

**Purpose**: Compare means of two groups.

**Types**:
1. **One-sample t-test**: Compare sample mean to known value
2. **Two-sample t-test**: Compare means of two groups
3. **Paired t-test**: Compare means of paired samples

**Assumptions**:
- Data is normally distributed (or large sample)
- Equal variances (for two-sample)
- Independent samples

### Chi-Square Test

**Purpose**: Test independence between categorical variables.

**Formula**:
```
χ² = Σ((O_i - E_i)² / E_i)
```

Where:
- $O_i$: Observed frequency
- $E_i$: Expected frequency

**Use Cases**: Contingency tables, goodness of fit.

### ANOVA (Analysis of Variance)

**Purpose**: Compare means of three or more groups.

**Null Hypothesis**: All group means are equal
**Alternative**: At least one mean differs

**F-statistic**:
```
F = Between-group variance / Within-group variance
```

**Use Cases**: Comparing multiple treatments, experimental design.

### Mann-Whitney U Test

**Purpose**: Non-parametric alternative to t-test.

**Use When**:
- Data not normally distributed
- Small sample sizes
- Ordinal data

---

## 9. P-values and Significance

### What is a P-value?

**Definition**: Probability of observing data as extreme (or more extreme) than what we observed, assuming the null hypothesis is true.

**Interpretation**:
- **Small p-value (< 0.05)**: Unlikely data if $H_0$ is true → reject $H_0$
- **Large p-value (> 0.05)**: Likely data if $H_0$ is true → fail to reject $H_0$

### Common Misconceptions

❌ **P-value is probability $H_0$ is true**: No, it's probability of data given $H_0$
❌ **P-value = 0.05 means 5% chance of error**: No, it's the significance level
❌ **P-value measures effect size**: No, it measures evidence against $H_0$

### Significance Levels

- **α = 0.05**: Standard threshold (5% chance of Type I error)
- **α = 0.01**: Stricter (1% chance)
- **α = 0.10**: More lenient (10% chance)

### Multiple Testing Problem

**Problem**: Testing multiple hypotheses increases chance of false positives.

**Example**: Testing 20 hypotheses at α=0.05 → ~64% chance of at least one false positive.

**Solutions**:
- **Bonferroni correction**: Divide α by number of tests
- **False Discovery Rate (FDR)**: Control expected proportion of false discoveries
- **Benjamini-Hochberg procedure**: FDR control method

---

## Quick Reference

### Key Formulas

**Mean**: $\bar{x} = \frac{1}{n}\sum x_i$

**Variance**: $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$

**Standard Deviation**: $s = \sqrt{s^2}$

**Standard Error**: $SE = \frac{s}{\sqrt{n}}$

**Z-score**: $z = \frac{x - \mu}{\sigma}$

**Correlation**: $r = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$

### Decision Rules

- **Reject $H_0$**: p-value < α
- **Fail to reject $H_0$**: p-value ≥ α
- **95% CI doesn't contain null value**: Reject $H_0$ at α=0.05

---

*Last Updated: 2024*

