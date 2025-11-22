+++
title = "LLM Evaluation Metrics"
date = 2025-11-22T13:00:00+05:30
draft = false
weight = 3
description = "Comprehensive guide to LLM evaluation metrics covering knowledge & reasoning benchmarks (MMLU, ARC), code generation metrics, translation & summarization metrics, language modeling metrics, and classification metrics."
+++

## 1. Knowledge & Reasoning Benchmarks

### MMLU (Massive Multitask Language Understanding)

- **Purpose**: Test broad world knowledge and reasoning across domains.
- **Format**: Multiple-choice questions (4 options).
- **Evaluation Metric**: Accuracy (% correct answers).

---

### ARC (AI2 Reasoning Challenge)

- **Purpose**: Test commonsense science reasoning at the middle-school level.
- **Format**: Multiple-choice (science questions).
- **Metric**: Accuracy.

---

### BIG-Bench & BIG-Bench Hard (BBH)

- **Purpose**: Stress-test LLM reasoning on very challenging tasks.
- **Dataset**: 200+ tasks (linguistics, logic puzzles, math, code).
- **BBH**: Subset of hardest tasks.
- **Format**: Varies — multiple-choice, free-form.
- **Metric**: Task-specific accuracy or F1.

---

## 2. Code Generation Metrics

### HumanEval

- **Purpose**: Test code generation & functional correctness.
- **Dataset**: Python programming problems with hidden unit tests.
- **Metric**: Pass@1 (percentage of solutions passing all tests).

---

## 3. Translation & Summarization Metrics

### BLEU (Bilingual Evaluation Understudy)

- **Purpose**: Evaluate translation quality.
- **Method**: N-gram overlap between generated and reference text.
- **Range**: 0-1 (higher is better).

---

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **Purpose**: Evaluate summarization quality.
- **Variants**: 
  - ROUGE-N: N-gram overlap
  - ROUGE-L: Longest common subsequence
  - ROUGE-W: Weighted longest common subsequence
- **Common**: ROUGE-1, ROUGE-2, ROUGE-L

---

### METEOR

- **Purpose**: Alternative to BLEU with better correlation to human judgment.
- **Method**: Considers synonyms and word order.

---

### COMET

- **Purpose**: Context-aware metric for translation evaluation.
- **Method**: Uses neural models to assess translation quality.

---

## 4. Language Modeling Metrics

### Perplexity

- **Purpose**: Measures how accurately the model predicts the exact sequence of tokens present in text data.
- **Interpretation**: Lower perplexity = better model performance.
- **Formula**: Perplexity = exp(cross-entropy loss)

---

## 5. Classification Metrics

### Exact Match at N

- **Purpose**: Measure exact match accuracy for classification tasks.
- **Method**: Percentage of predictions that exactly match the reference.

---

### AUC-ROC (Area Under the ROC Curve)

- **Purpose**: Evaluate binary classification performance.
- **Range**: 0-1 (higher is better, 0.5 = random).

---
