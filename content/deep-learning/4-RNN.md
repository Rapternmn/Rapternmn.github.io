+++
title = "Recurrent Neural Networks (RNN) & LSTM Architectures"
date = 2025-11-22T11:00:00+05:30
draft = false
description = "Comprehensive guide to Recurrent Neural Networks covering vanilla RNN, LSTM, GRU, bidirectional RNNs, sequence-to-sequence architectures, and their applications in NLP, time series, and sequential data processing."
+++

# 🔄 Recurrent Neural Networks (RNN) & LSTM Architectures

Recurrent Neural Networks (RNNs) are a class of neural networks designed to process sequential data by maintaining hidden states that capture information from previous time steps. This document covers fundamental RNN architectures, their mathematical formulations, variants, and applications.

**Key Concepts:**
- Sequential data processing
- Hidden state propagation
- Vanishing/exploding gradient problems
- Gating mechanisms (LSTM, GRU)
- Bidirectional and deep architectures

---

## 🌳 1. Basic RNN (Vanilla RNN)

The basic RNN is the simplest recurrent architecture that processes sequences by maintaining a hidden state that gets updated at each time step.

### Mathematical Formulation

**Forward Pass:**

At each time step `t`:
```
hₜ = tanh(Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + bₕ)
yₜ = Wₕᵧ · hₜ + bᵧ
```

Where:
- `hₜ` = hidden state at time `t` (shape: `[hidden_size]`)
- `hₜ₋₁` = hidden state at previous time step
- `xₜ` = input at time `t` (shape: `[input_size]`)
- `yₜ` = output at time `t` (shape: `[output_size]`)
- `Wₕₕ` = weight matrix for hidden-to-hidden (shape: `[hidden_size × hidden_size]`)
- `Wₓₕ` = weight matrix for input-to-hidden (shape: `[hidden_size × input_size]`)
- `Wₕᵧ` = weight matrix for hidden-to-output (shape: `[output_size × hidden_size]`)
- `bₕ`, `bᵧ` = bias vectors
- `tanh` = hyperbolic tangent activation function

**Initial Hidden State:**
```
h₀ = 0  (typically initialized to zeros)
```

**Vectorized Form (for batch of size B):**
```
Hₜ = tanh(Hₜ₋₁ · Wₕₕᵀ + Xₜ · Wₓₕᵀ + bₕ)
Yₜ = Hₜ · Wₕᵧᵀ + bᵧ
```

Where:
- `Hₜ` = `[B × hidden_size]`
- `Xₜ` = `[B × input_size]`
- `Yₜ` = `[B × output_size]`

### Loss Function

**For Sequence-to-Sequence Tasks:**
```
L = (1/T) * Σₜ₌₁ᵀ Lₜ(yₜ, ŷₜ)
```

Where `Lₜ` is the loss at time step `t` (e.g., cross-entropy for classification, MSE for regression).

**For Sequence-to-One Tasks:**
```
L = L(yₜ, ŷₜ)  (only final output)
```

### Backpropagation Through Time (BPTT)

**Gradient w.r.t. Hidden State:**
```
∂L/∂hₜ = ∂L/∂yₜ · Wₕᵧ + ∂L/∂hₜ₊₁ · Wₕₕ · (1 - tanh²(zₜ₊₁))
```

Where `zₜ = Wₕₕ · hₜ₋₁ + Wₓₕ · xₜ + bₕ`.

**Gradient w.r.t. Weights:**
```
∂L/∂Wₕₕ = Σₜ₌₁ᵀ ∂L/∂hₜ · (1 - tanh²(zₜ)) · hₜ₋₁ᵀ
∂L/∂Wₓₕ = Σₜ₌₁ᵀ ∂L/∂hₜ · (1 - tanh²(zₜ)) · xₜᵀ
∂L/∂Wₕᵧ = Σₜ₌₁ᵀ ∂L/∂yₜ · hₜᵀ
```

### Properties

**Advantages:**
- **Sequential processing:** Can handle variable-length sequences
- **Parameter sharing:** Same weights across all time steps
- **Memory:** Hidden state captures information from previous steps
- **Flexible:** Can be used for sequence-to-sequence, sequence-to-one, one-to-sequence tasks

**Limitations:**
- **Vanishing gradients:** Gradients decay exponentially over time steps
- **Exploding gradients:** Gradients can grow exponentially (less common)
- **Short-term memory:** Struggles with long-range dependencies
- **Computational bottleneck:** Sequential processing prevents parallelization

### Vanishing Gradient Problem

**Root Cause:**
The gradient flows through time via repeated multiplication by `Wₕₕ`:
```
∂L/∂h₀ = ∂L/∂hₜ · Wₕₕᵀ · Wₕₕᵀ · ... · Wₕₕᵀ · (1 - tanh²(z₁)) · ... · (1 - tanh²(zₜ))
```

If eigenvalues of `Wₕₕ` are < 1, gradients vanish. If > 1, gradients explode.

**Impact:**
- Early time steps receive very small gradients
- Network cannot learn long-range dependencies
- Training becomes very slow or ineffective

---

## 🟦 2. Long Short-Term Memory (LSTM)

LSTM was designed to solve the vanishing gradient problem by introducing gating mechanisms that allow the network to selectively remember or forget information.

### Architecture Overview

An LSTM cell has three gates:
1. **Forget Gate:** Decides what information to discard from cell state
2. **Input Gate:** Decides what new information to store in cell state
3. **Output Gate:** Decides what parts of cell state to output

### Mathematical Formulation

**Cell Components:**

**Forget Gate:**
```
fₜ = σ(Wₓf · xₜ + Wₕf · hₜ₋₁ + bf)
```

**Input Gate:**
```
iₜ = σ(Wₓᵢ · xₜ + Wₕᵢ · hₜ₋₁ + bi)
C̃ₜ = tanh(WₓC · xₜ + WₕC · hₜ₋₁ + bC)
```

**Cell State Update:**
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

**Output Gate:**
```
oₜ = σ(Wₓₒ · xₜ + Wₕₒ · hₜ₋₁ + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

Where:
- `σ` = sigmoid activation function
- `⊙` = element-wise multiplication (Hadamard product)
- `Cₜ` = cell state at time `t`
- `Cₜ₋₁` = cell state at previous time step
- `hₜ` = hidden state (output) at time `t`
- `fₜ`, `iₜ`, `oₜ` = forget, input, output gates (all in range [0, 1])
- `C̃ₜ` = candidate cell state values

**Initial States:**
```
h₀ = 0
C₀ = 0
```

### Parameter Count

For an LSTM with `input_size = d` and `hidden_size = h`:
- Forget gate: `4h² + 4hd + 4h` parameters
- Input gate: `4h² + 4hd + 4h` parameters
- Cell state: `4h² + 4hd + 4h` parameters
- Output gate: `4h² + 4hd + 4h` parameters

**Total:** `16h² + 16hd + 16h = 16h(h + d + 1)` parameters

### Why LSTM Solves Vanishing Gradients

**Key Insight:** The cell state `Cₜ` has a linear path through time:
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

The gradient flows through this path with minimal decay:
```
∂Cₜ/∂Cₜ₋₁ = fₜ  (element-wise)
```

Since `fₜ` is learned and can be close to 1, gradients can flow through many time steps without vanishing.

### Properties

**Advantages:**
- **Long-term memory:** Can remember information for many time steps
- **Selective memory:** Gates allow fine-grained control over what to remember/forget
- **Gradient flow:** Better gradient propagation than vanilla RNN
- **Flexible:** Works well for various sequence tasks

**Limitations:**
- **Computational cost:** More parameters and operations than vanilla RNN
- **Complexity:** More hyperparameters to tune
- **Still sequential:** Cannot parallelize across time steps
- **Memory intensive:** Stores both hidden state and cell state

### Variants

#### Peephole Connections
Allows gates to see the cell state:
```
fₜ = σ(Wₓf · xₜ + Wₕf · hₜ₋₁ + Wₐf · Cₜ₋₁ + bf)
```

#### Coupled Input-Forget Gate
Combines input and forget gates:
```
Cₜ = (1 - iₜ) ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

---

## 🟩 3. Gated Recurrent Unit (GRU)

GRU is a simplified variant of LSTM that combines the forget and input gates into a single "update gate" and merges the cell state and hidden state.

### Mathematical Formulation

**Update Gate:**
```
zₜ = σ(Wₓz · xₜ + Wₕz · hₜ₋₁ + bz)
```

**Reset Gate:**
```
rₜ = σ(Wₓr · xₜ + Wₕr · hₜ₋₁ + br)
```

**Candidate Hidden State:**
```
h̃ₜ = tanh(Wₓh · xₜ + Wₕh · (rₜ ⊙ hₜ₋₁) + bh)
```

**Hidden State Update:**
```
hₜ = (1 - zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
```

Where:
- `zₜ` = update gate (controls how much of previous state to keep)
- `rₜ` = reset gate (controls how much of previous state to forget)
- `h̃ₜ` = candidate hidden state
- `hₜ` = final hidden state

**Initial State:**
```
h₀ = 0
```

### Parameter Count

For a GRU with `input_size = d` and `hidden_size = h`:
- Update gate: `h² + hd + h` parameters
- Reset gate: `h² + hd + h` parameters
- Candidate state: `h² + hd + h` parameters

**Total:** `3h² + 3hd + 3h = 3h(h + d + 1)` parameters

### Properties

**Advantages:**
- **Simpler than LSTM:** Fewer parameters (3 gates vs 4 in LSTM)
- **Faster training:** Less computation per time step
- **Often comparable performance:** Works as well as LSTM on many tasks
- **Better gradient flow:** Still solves vanishing gradient problem

**Limitations:**
- **Less expressive:** May struggle with very long sequences compared to LSTM
- **Still sequential:** Cannot parallelize across time steps
- **Memory:** Still needs to store hidden state

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| **Gates** | 3 (forget, input, output) | 2 (update, reset) |
| **States** | Hidden + Cell | Hidden only |
| **Parameters** | ~4× more | Fewer |
| **Complexity** | Higher | Lower |
| **Performance** | Better on long sequences | Often comparable |
| **Training Speed** | Slower | Faster |

**When to use GRU:**
- Limited computational resources
- Shorter sequences
- When LSTM performance is similar

**When to use LSTM:**
- Very long sequences
- Complex long-range dependencies
- When maximum performance is needed

---

## 🔀 4. Bidirectional RNN (BiRNN)

Bidirectional RNNs process sequences in both forward and backward directions, allowing the network to use information from both past and future contexts.

### Architecture

**Forward Pass:**
```
hₜ→ = f(Wₓₕ→ · xₜ + Wₕₕ→ · hₜ₋₁→ + b→)
```

**Backward Pass:**
```
hₜ← = f(Wₓₕ← · xₜ + Wₕₕ← · hₜ₊₁← + b←)
```

**Combined Output:**
```
hₜ = [hₜ→; hₜ←]  (concatenation)
yₜ = Wₕᵧ · hₜ + bᵧ
```

Where:
- `hₜ→` = forward hidden state
- `hₜ←` = backward hidden state
- `[;]` = concatenation operator

### Variants

#### Bidirectional LSTM (BiLSTM)
Uses LSTM cells in both directions:
```
hₜ→ = LSTM→(xₜ, hₜ₋₁→, Cₜ₋₁→)
hₜ← = LSTM←(xₜ, hₜ₊₁←, Cₜ₊₁←)
hₜ = [hₜ→; hₜ←]
```

#### Bidirectional GRU (BiGRU)
Uses GRU cells in both directions:
```
hₜ→ = GRU→(xₜ, hₜ₋₁→)
hₜ← = GRU←(xₜ, hₜ₊₁←)
hₜ = [hₜ→; hₜ←]
```

### Properties

**Advantages:**
- **Context awareness:** Can use both past and future information
- **Better representations:** Often produces richer feature representations
- **Useful for:** Named entity recognition, sentiment analysis, machine translation

**Limitations:**
- **Requires full sequence:** Cannot be used in online/streaming scenarios
- **More parameters:** Approximately 2× parameters of unidirectional RNN
- **Slower inference:** Must process entire sequence before output

### Applications

- **Named Entity Recognition (NER):** Context from both sides helps identify entities
- **Sentiment Analysis:** Future words can clarify sentiment of current words
- **Machine Translation:** Better understanding of source sentence structure

---

## 📚 5. Deep RNN (Stacked RNN)

Deep RNNs stack multiple RNN layers on top of each other, allowing the network to learn hierarchical representations of sequential data.

### Architecture

**Multi-Layer RNN:**
```
hₜ⁽¹⁾ = RNN₁(xₜ, hₜ₋₁⁽¹⁾)
hₜ⁽²⁾ = RNN₂(hₜ⁽¹⁾, hₜ₋₁⁽²⁾)
...
hₜ⁽ᴸ⁾ = RNNₗ(hₜ⁽ᴸ⁻¹⁾, hₜ₋₁⁽ᴸ⁾)
yₜ = W · hₜ⁽ᴸ⁾ + b
```

Where:
- `L` = number of layers
- `hₜ⁽ˡ⁾` = hidden state of layer `l` at time `t`
- Each layer can be vanilla RNN, LSTM, or GRU

### Properties

**Advantages:**
- **Hierarchical features:** Lower layers capture local patterns, higher layers capture abstract patterns
- **Increased capacity:** More parameters allow modeling complex sequences
- **Better representations:** Often improves performance on complex tasks

**Limitations:**
- **Training difficulty:** Deeper networks are harder to train
- **Computational cost:** More layers = more computation
- **Overfitting risk:** More parameters can lead to overfitting
- **Gradient issues:** Still susceptible to vanishing gradients (though LSTM/GRU help)

### Best Practices

- **Layer normalization:** Apply layer norm between RNN layers
- **Residual connections:** Add skip connections to help gradient flow
- **Dropout:** Apply dropout between layers (not in recurrent connections)
- **Gradual depth:** Start with 2-3 layers, increase if needed

---

## 🔄 6. Sequence-to-Sequence (Seq2Seq) Architecture

Seq2Seq models use an encoder-decoder architecture to map variable-length input sequences to variable-length output sequences.

### Architecture

**Encoder:**
```
hₜᵉⁿᶜ = RNN(xₜ, hₜ₋₁ᵉⁿᶜ)
c = hₜᵉⁿᶜ  (context vector from final hidden state)
```

**Decoder:**
```
hₜᵈᵉᶜ = RNN(yₜ₋₁, hₜ₋₁ᵈᵉᶜ, c)
yₜ = softmax(W · hₜᵈᵉᶜ + b)
```

Where:
- Encoder processes input sequence `x₁, ..., xₜ`
- Context vector `c` summarizes entire input
- Decoder generates output sequence `y₁, ..., yₜ'` one token at a time

### Attention Mechanism

**Problem with Basic Seq2Seq:**
- Single context vector `c` must encode entire input sequence
- Information bottleneck for long sequences
- All input positions treated equally

**Solution: Attention**
```
Attention weights: αₜᵢ = softmax(score(hₜᵈᵉᶜ, hᵢᵉⁿᶜ))
Context vector: cₜ = Σᵢ αₜᵢ · hᵢᵉⁿᶜ
```

**Attention Variants:**

**Dot Product Attention:**
```
score(hₜᵈᵉᶜ, hᵢᵉⁿᶜ) = hₜᵈᵉᶜᵀ · hᵢᵉⁿᶜ
```

**General Attention:**
```
score(hₜᵈᵉᶜ, hᵢᵉⁿᶜ) = hₜᵈᵉᶜᵀ · Wₐ · hᵢᵉⁿᶜ
```

**Additive Attention (Bahdanau):**
```
score(hₜᵈᵉᶜ, hᵢᵉⁿᶜ) = vᵀ · tanh(W₁ · hₜᵈᵉᶜ + W₂ · hᵢᵉⁿᶜ)
```

### Properties

**Advantages:**
- **Variable length:** Handles sequences of different lengths
- **Flexible:** Can be used for translation, summarization, dialogue
- **Attention:** Allows focusing on relevant parts of input

**Limitations:**
- **Sequential decoding:** Cannot parallelize output generation
- **Slow inference:** Must generate tokens one at a time
- **Context limitation:** Fixed-size context vector in basic version

---

## 🎯 7. Applications and Use Cases

### Natural Language Processing (NLP)
- **Machine Translation:** Seq2Seq with attention
- **Text Generation:** Language modeling with RNN/LSTM
- **Sentiment Analysis:** Classification using BiLSTM
- **Named Entity Recognition:** Sequence labeling with BiLSTM + CRF

### Speech Recognition
- **Speech-to-Text:** Acoustic modeling with LSTM
- **Voice Activity Detection:** Sequence classification

### Time Series Forecasting
- **Stock Price Prediction:** LSTM for financial time series
- **Weather Forecasting:** RNN for temporal patterns
- **Demand Forecasting:** Sequence prediction

### Other Applications
- **Video Analysis:** Frame-by-frame processing
- **Music Generation:** Sequential pattern learning
- **Protein Structure Prediction:** Biological sequence analysis

---

## 🔧 8. Training Techniques

### Gradient Clipping

**Problem:** Exploding gradients in RNNs

**Solution:** Clip gradients to a maximum norm:
```
if ||g|| > threshold:
    g = g · (threshold / ||g||)
```

Where `g` is the gradient vector.

### Truncated BPTT

**Problem:** Full BPTT is computationally expensive for long sequences

**Solution:** Only backpropagate through a fixed window:
```
Backpropagate through last K time steps only
```

### Teacher Forcing

**Problem:** During training, decoder uses its own (potentially wrong) predictions

**Solution:** Use ground truth labels during training:
```
Training: hₜ = RNN(yₜ₋₁ᵗʳᵘᵉ, hₜ₋₁)
Inference: hₜ = RNN(yₜ₋₁ᵖʳᵉᵈ, hₜ₋₁)
```

### Scheduled Sampling

Gradually transition from teacher forcing to using predictions:
```
Use ground truth with probability p, prediction with (1-p)
p decreases during training
```

---

## 📊 9. Comparison Summary

| Architecture | Parameters | Memory | Training Speed | Long Sequences | Use Case |
|--------------|------------|--------|----------------|----------------|----------|
| **Vanilla RNN** | Low | Low | Fast | Poor | Simple sequences |
| **LSTM** | High | High | Slow | Excellent | Long sequences, complex dependencies |
| **GRU** | Medium | Medium | Medium | Good | Balanced performance/speed |
| **BiRNN** | 2× | 2× | 2× slower | Depends on cell | Context-aware tasks |
| **Deep RNN** | L× | L× | L× slower | Better | Complex hierarchical patterns |
| **Seq2Seq** | Very High | Very High | Very Slow | Excellent | Translation, summarization |

---

## 🎓 10. Key Takeaways

1. **RNNs are designed for sequential data** but suffer from vanishing gradients
2. **LSTM solves vanishing gradients** through gating mechanisms and cell state
3. **GRU is a simpler alternative** to LSTM with comparable performance
4. **Bidirectional RNNs** use both past and future context
5. **Deep RNNs** learn hierarchical representations
6. **Seq2Seq with attention** enables variable-length sequence mapping
7. **Training techniques** like gradient clipping and teacher forcing are essential
8. **Choose architecture based on:** sequence length, computational budget, task requirements

---

## 📚 References & Further Reading

- **Original LSTM Paper:** Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **GRU Paper:** Cho et al. (2014) - "Learning Phrase Representations using RNN Encoder-Decoder"
- **Attention Mechanism:** Bahdanau et al. (2014) - "Neural Machine Translation by Jointly Learning to Align and Translate"
- **BPTT:** Werbos (1990) - "Backpropagation Through Time"
- **Modern Applications:** See Transformer architecture for attention-only models

---

*This document covers fundamental RNN architectures. For attention-only models (Transformers), see separate documentation.*

