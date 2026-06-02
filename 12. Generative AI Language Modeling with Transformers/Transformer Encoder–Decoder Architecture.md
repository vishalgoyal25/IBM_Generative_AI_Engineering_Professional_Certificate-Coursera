# Transformer Encoder–Decoder Architecture: Complete Deep Learning Notes

---

# 1. Introduction

The Transformer architecture is the foundation of modern Generative AI.

Before Transformers, most Natural Language Processing (NLP) systems relied on:

* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory Networks (LSTMs)
* Sequence-to-Sequence Models

These architectures processed text sequentially.

Example:

```text
Word 1 → Word 2 → Word 3 → Word 4
```

This caused:

* Slow training
* Difficulty handling long-range dependencies
* Limited parallelization

Transformers solved these problems using:

## Attention Mechanism

Instead of processing tokens one at a time, Transformers process all tokens simultaneously and learn which words are important to one another.

This innovation became the foundation of:

* BERT
* GPT
* T5
* LLaMA
* Claude
* Gemini
* Modern Agentic AI Systems

---

# 2. Original Transformer Architecture

The original Transformer contains two major components:

```text
Encoder
   ↓
Memory
   ↓
Decoder
```

The encoder understands the input sequence.

The decoder generates the output sequence.

Example:

```text
Input:
"I love machine learning"

↓

Output:
"J'aime l'apprentissage automatique"
```

---

# 3. Complete High-Level Pipeline

```text
Input Sentence
      ↓
Tokenization
      ↓
Embedding
      ↓
Positional Encoding
      ↓
Encoder
      ↓
Memory
      ↓
Decoder
      ↓
Linear Layer
      ↓
Softmax
      ↓
Generated Tokens
```

---

# 4. Tokenization

Neural networks cannot understand raw text.

Text must first be broken into tokens.

## Example

Input:

```text
I love machine learning
```

Tokenized:

```text
[I]
[love]
[machine]
[learning]
```

These tokens become the input to the embedding layer.

---

# 5. Token Embedding

The model converts tokens into vectors.

Example:

```text
love
```

becomes

```text
[0.23, 0.91, 0.18, ...]
```

Every token receives its own vector representation.

The embedding layer learns semantic relationships.

Example:

```text
King
Queen
Prince
Princess
```

are represented close to one another in vector space.

---

# 6. Why Embeddings Matter

Without embeddings:

```text
cat = 15
dog = 38
```

These numbers have no meaning.

With embeddings:

```text
cat → vector
dog → vector
lion → vector
```

The model learns:

```text
cat ≈ dog
cat ≈ lion
```

Semantic relationships emerge automatically.

---

# 7. Positional Encoding

Transformers process all tokens simultaneously.

Therefore:

```text
I love AI
```

and

```text
AI love I
```

would appear identical.

To solve this problem, positional information is added.

---

## Example

| Token | Position |
| ----- | -------- |
| I     | 1        |
| love  | 2        |
| AI    | 3        |

Position vectors are added to token embeddings.

This allows the model to learn word order.

---

# 8. Encoder Overview

The encoder's job is:

## Understand The Entire Input

The encoder:

* Reads all words
* Understands context
* Learns relationships
* Produces contextual representations

Output:

```text
Memory
```

---

# 9. Encoder Block

Each encoder layer contains:

## Multi-Head Self-Attention

followed by

## Feed Forward Network

with

## Residual Connections

and

## Layer Normalization

---

# 10. Self-Attention

The core question:

```text
Which words matter to me?
```

Every word looks at every other word.

---

## Example

Sentence:

```text
The animal didn't cross the road because it was tired.
```

The word:

```text
it
```

attends strongly to:

```text
animal
```

instead of

```text
road
```

The model learns this automatically.

---

# 11. Multi-Head Attention

Instead of one attention mechanism:

```text
1 attention head
```

the model uses:

```text
Head 1
Head 2
Head 3
Head 4
...
```

Each head learns different relationships.

---

## Example

Head 1:

```text
Grammar
```

Head 2:

```text
Meaning
```

Head 3:

```text
Long-range dependencies
```

Head 4:

```text
Phrase structure
```

---

# 12. Feed Forward Network

After attention:

```text
Communication
```

comes

```text
Processing
```

Each token passes through:

```text
Linear
↓
Activation
↓
Linear
```

This refines the learned representation.

---

# 13. Encoder Output

After several encoder layers:

```text
Input Sentence
↓
Encoder
↓
Memory
```

This memory contains:

* Meaning
* Context
* Relationships
* Semantic information

The entire sentence is now represented numerically.

---

# 14. Decoder Overview

The decoder's job is:

## Generate Output Tokens

Unlike the encoder:

```text
Encoder → Understand
Decoder → Generate
```

---

# 15. Decoder Inputs

The decoder receives:

## Input 1

Previously generated tokens.

Example:

```text
J'
```

then

```text
J'aime
```

then

```text
J'aime l'
```

---

## Input 2

Encoder Memory

The decoder continuously consults encoder output.

---

# 16. Masked Self-Attention

Decoder generation must be causal.

Future tokens cannot be visible.

---

## Example

Predict:

```text
Word #3
```

Allowed:

```text
Word #1
Word #2
```

Not Allowed:

```text
Word #4
Word #5
```

This is called:

## Causal Masking

---

# 17. Why Causal Masking Exists

Without masking:

The model would cheat.

Training would become meaningless.

Masking forces genuine next-token prediction.

---

# 18. Cross Attention

Cross-attention connects:

```text
Decoder
↓
Encoder Memory
```

This allows translation.

---

## Example

Source:

```text
I love machine learning
```

Generating:

```text
apprentissage
```

The decoder attends strongly to:

```text
machine learning
```

inside encoder memory.

---

# 19. Why Cross-Attention Matters

Without cross-attention:

The decoder would have no knowledge of the source sentence.

Translation would be impossible.

---

# 20. Linear Layer

Decoder output is still a vector.

Humans need words.

The linear layer maps:

```text
Hidden Vector
↓
Vocabulary Space
```

Example:

```text
Paris      0.92
London     0.03
Berlin     0.02
```

---

# 21. Softmax Layer

Softmax converts scores into probabilities.

Example:

```text
Paris      92%
London      5%
Berlin      3%
```

Highest probability wins.

---

# 22. Autoregressive Generation

The process repeats:

```text
J'
↓
J'aime
↓
J'aime l'
↓
J'aime l'apprentissage
↓
J'aime l'apprentissage automatique
```

until:

```text
<EOS>
```

appears.

---

# 23. Encoder vs Decoder

| Encoder             | Decoder                    |
| ------------------- | -------------------------- |
| Understands         | Generates                  |
| Bidirectional       | Autoregressive             |
| Uses Self-Attention | Uses Masked Self-Attention |
| Produces Memory     | Produces Tokens            |
| BERT Foundation     | GPT Foundation             |

---

# 24. BERT

Architecture:

```text
Encoder Only
```

Purpose:

```text
Language Understanding
```

Tasks:

* MLM
* NSP
* Classification
* Search
* Embeddings

---

# 25. GPT

Architecture:

```text
Decoder Only
```

Purpose:

```text
Language Generation
```

Tasks:

* Chatbots
* Content Generation
* Coding
* Reasoning
* Agents

---

# 26. Key Insight

Transformer introduced:

## Attention

BERT introduced:

## Deep Language Understanding

GPT introduced:

## Autoregressive Generation

Modern Generative AI combines these concepts to build:

* LLMs
* RAG Systems
* Tool Calling
* AI Agents
* Agentic Workflows

---

# 27. Final Summary

Complete Flow:

```text
Text
↓
Tokens
↓
Embeddings
↓
Positional Encoding
↓
Encoder Attention
↓
Feed Forward
↓
Memory
↓
Decoder Masked Attention
↓
Cross Attention
↓
Feed Forward
↓
Linear Layer
↓
Softmax
↓
Generated Token
↓
Final Output
```

This Encoder–Decoder Transformer architecture is the conceptual foundation behind BERT, GPT, Fine-Tuning, RAG, and Agentic AI.
