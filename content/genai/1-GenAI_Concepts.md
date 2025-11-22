+++
title = "GenAI Concepts"
date = 2025-11-22T13:00:00+05:30
draft = false
weight = 1
description = "Comprehensive guide to Generative AI concepts covering core concepts, parameters & generation control, training paradigms, architecture, evaluation, RAG, system design, optimization, safety, and latest trends."
+++

## 1. Core Concepts

**Q: What is Generative AI? How is it different from traditional AI?**  
A: Generative AI creates new content (text, images, audio, code) based on learned patterns. Traditional AI focuses on classification/regression (discriminative tasks), while Generative AI produces data.

**Q: What is the difference between discriminative and generative models?**  
- **Discriminative**: Learn P(y|x) → classify/predict (e.g., Logistic Regression, BERT)
- **Generative**: Learn P(x,y) or P(x) → generate new data (e.g., GPT, GANs)

**Q: Explain how LLMs work at a high level.**  
A: Large Language Models are trained on vast text corpora using Transformer architecture. They predict the next token in a sequence, learning language patterns, semantics, and world knowledge. Pre-training builds general knowledge, while fine-tuning and alignment (e.g., RLHF) adapt models for specific tasks.

**Q: What is the difference between pre-training, fine-tuning, RLHF, and RAG?**  
- **Pre-training**: Train on large general corpus with next-token prediction
- **Fine-tuning**: Further train on task-specific labeled data
- **RLHF**: Use human preference data to align outputs with human values
- **RAG**: Combine LLMs with external knowledge retrieval (vector DBs) for factual, updated responses

**Q: What are embeddings and why are they useful?**  
A: Embeddings are vector representations of text that capture semantic meaning. They enable similarity search, clustering, recommendations, and retrieval in RAG pipelines.

**Q: What is the difference between encoder-only, decoder-only, and encoder-decoder architectures?**  
- **Encoder-only (e.g., BERT)**: Good for classification, embeddings, understanding tasks
- **Decoder-only (e.g., GPT)**: Good for generative tasks (text completion)
- **Encoder-decoder (e.g., T5, BART)**: Good for seq-to-seq tasks (translation, summarization)

**Q: What is prompt engineering? Why is it important?**  
A: Prompt engineering is designing effective input instructions to guide LLM behavior. It impacts model performance significantly without retraining. Techniques: add context, use few-shot examples, add constraints.

**Q: How to improve an LLM's performance without fine-tuning?**  
1. Prompt refinement
2. Few-shot prompting
3. Use RAG (Retrieval-Augmented Generation)
4. Output format constraints
5. External tools (calculators, search)

---

## 2. Parameters & Generation Control

**Q: What does Temperature do?**  
A: Scales token probabilities before sampling.
- Low (0.0–0.3): Deterministic, factual tasks
- Medium (~0.7): Balanced creativity
- High (>1): More random/creative

**Q: Explain Top-k and Top-p.**  
- **Top-k**: Sample only from the top k most likely tokens
- **Top-p (nucleus)**: Sample from the smallest set of tokens whose cumulative probability ≥ p
- Both control randomness and diversity
- Common setup: `temperature=0.7, top_p=0.9`

**Q: How to reduce repetition in LLM outputs?**  
A: Use **repetition_penalty**, **presence_penalty**, or adjust decoding strategy.

---

## 3. Training Paradigms

**Q: Difference between pretraining, fine-tuning, and instruction-tuning?**  
- **Pretraining**: Train on general web corpus (next token prediction)
- **Fine-tuning**: Train on task-specific dataset (e.g., customer support)
- **Instruction-tuning**: Train to follow user instructions (prompt → response)

**Q: What is Reinforcement Learning from Human Feedback (RLHF)?**  
A: RLHF aligns LLM behavior with human preferences:
1. Pretrain with next-token prediction
2. Collect human preference scores
3. Train a reward model
4. Optimize LLM with reinforcement learning

Used in ChatGPT for safe, helpful responses.

---

## 4. Architecture & Components

**Q: How does the Transformer architecture work?**  
A: Transformer uses self-attention to capture relationships between all tokens in a sequence — parallelizable and scalable.

**Variants:**
- **Encoder**: Used in BERT
- **Decoder**: Used in GPT
- **Encoder-Decoder**: Used in T5

**Q: What is self-attention?**  
A: A mechanism where each token attends to all other tokens in a sequence and assigns weights based on relevance. It allows the model to capture context efficiently.

**Q: What are positional encodings?**  
A: Since transformers have no recurrence, positional encodings are added to input embeddings to give the model a sense of order.

---

## 5. Evaluation

**Q: How do you evaluate an LLM response?**  
- **Offline metrics**: BLEU, ROUGE, METEOR, perplexity, BERTScore
- **Human evaluation**: correctness, fluency, factuality, safety
- **For RAG**: retrieval precision/recall, grounding, faithfulness

**Q: What is hallucination in LLMs? How to mitigate it?**  
A: Hallucination = confident but factually wrong output.

**Mitigations:**
- Use RAG (external knowledge)
- Prompt grounding with references
- Post-processing with fact-checkers
- Penalize hallucinations during training

---

## 6. RAG (Retrieval-Augmented Generation)

**Q: Why use RAG instead of fine-tuning?**  
A: Fine-tuning is costly, static, and not easily updated. RAG allows dynamic knowledge updates without retraining and reduces hallucinations.

**Q: What is chunking strategy in RAG?**  
A: Splitting documents into smaller chunks for embedding/retrieval.
- **Fixed size**: Simple, but may cut semantic boundaries
- **Semantic chunking**: Splits at natural boundaries (paragraphs, headings)
- **Sliding window**: Adds overlap to preserve context

---

## 7. System Design for GenAI

**Q: Design a Document Q&A system with RAG.**  
1. Preprocess documents → chunk → create embeddings → store in vector DB
2. At query time: embed query → retrieve top-k chunks → feed chunks + query to LLM
3. Generate answer with citations
4. Evaluate using retrieval metrics + answer grounding

**Q: How to scale a real-time GenAI system for millions of users?**  
- Use **caching** for frequent queries
- **Batching** requests on GPU
- **Model quantization** for faster inference
- **Speculative decoding** or **smaller distilled models** for speed
- Load balance across inference servers

---

## 8. Optimization & Deployment

**Q: What are LoRA and PEFT?**  
- **LoRA (Low-Rank Adaptation)**: Fine-tunes only small low-rank matrices, not the full model
- **PEFT (Parameter Efficient Fine-Tuning)**: General techniques to fine-tune small subsets of parameters
- Both save compute and memory
- Use cases: Domain adaptation with limited data

**Q: What is quantization?**  
A: Reducing precision (e.g., 16-bit → 8-bit → 4-bit) to shrink model size and speed up inference, with minimal accuracy loss.

**Q: How to reduce latency in LLM serving?**  
- Use model distillation, quantization
- Use GPU/TPU batching
- Use caching
- Speculative decoding
- Stream responses instead of waiting for full completion

---

## 9. Safety & Ethics

**Q: What risks exist in deploying LLMs?**  
- Hallucinations
- Bias & toxicity
- Prompt injection & data leakage
- Security misuse (phishing, malware)

**Mitigations**: Guardrails, content filtering, RAG grounding, red-teaming.

**Q: What are guardrails?**  
A: Systems (filters, validators, policies) that ensure model outputs remain safe, relevant, and compliant.

---

## 10. Latest Trends

**Q: What is function calling in LLMs?**  
A: The model outputs structured JSON-like arguments to call external APIs/tools. Enables LLMs as controllers in agent systems.

**Q: How do long-context models work?**  
A: They use techniques like **attention optimizations (FlashAttention, sliding-window attention, recurrence)** or **memory tokens** to handle >100k tokens.

**Q: What is structured output generation?**  
A: Forcing model to output JSON, XML, or conform to a schema using constrained decoding or grammar-based generation.

---

## Quick Tips

- For **factual tasks** → low temperature, RAG grounding
- For **creative tasks** → higher temperature/top-p
- Always explain **trade-offs**: coherence vs. diversity, accuracy vs. creativity
- Mention **latency, scalability, safety** in system design answers

