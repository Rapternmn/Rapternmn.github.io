+++
title = "Prompts: Chat Format and Templates"
date = 2025-11-22T13:00:00+05:30
draft = false
description = "Comprehensive guide to prompt engineering covering chat format APIs (OpenAI, Anthropic, Claude), prompt templates, deployment formats, and best practices for effective prompt design."
+++

## 1. Prompt as a Dict (Chat Format API – OpenAI / Anthropic / Claude)

In OpenAI's Chat API (e.g., GPT-4 Turbo), the prompt is passed as a list of dictionaries representing a conversation.

```json
[
  {
    "role": "system",
    "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately."
  },
  {
    "role": "user",
    "content": "Context:\n<retrieved context chunks>\n\nQuestion: <user query>"
  }
]
```

This is the chat format used by:

- OpenAI (GPT-3.5, GPT-4)
- Anthropic Claude (uses system + messages)
- Gemini / Mistral (via wrappers like LangChain, LlamaIndex)

---

## 2. Why We Still Show Prompt Templates as Strings

When you're working with:

- Prompt engineering / chaining (LangChain, LlamaIndex)
- RAG pipelines
- Prompt tuning or fine-tuning

...you often first define the prompt as a template string, like:

```python
template = """
You are a helpful assistant.

Context:
{context}

Question:
{question}

Answer:
"""
```

---

## 3. Final Format for Deployment

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages
)
```

---

## 4. Optional Roles in OpenAI API

- **"system"** – Sets behavior/tone of assistant (e.g. domain expert, formal, casual).
- **"user"** – User question or input.
- **"assistant"** – (Optional) Insert previous assistant replies to maintain conversation.
- **"tool"** – Used in tool-calling.

---

## Summary

| Format        | When Used                            | Example                                    |
| ------------- | ------------------------------------ | ------------------------------------------ |
| Prompt String | For templates and chaining           | `f"Context: {context}\nQuestion: {query}"` |
| Dict Format   | For Chat APIs (OpenAI, Claude, etc.) | `[{"role": "user", "content": "..."}]`     |
