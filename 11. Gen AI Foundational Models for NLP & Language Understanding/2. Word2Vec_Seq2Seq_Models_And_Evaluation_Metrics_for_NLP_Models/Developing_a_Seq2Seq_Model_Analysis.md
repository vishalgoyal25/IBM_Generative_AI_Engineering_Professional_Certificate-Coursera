# Developing a Sequence-to-Sequence (Seq2Seq) Model - Complete Lab Analysis & Learning Notes

---

# 1. Lab Goal

The purpose of this lab was **NOT** to build a production-grade translator.

The real purpose was to understand:

- How Neural Machine Translation (NMT) worked before Transformers.
- How Encoder-Decoder architectures were designed.
- How Seq2Seq models generate language.
- Why Attention was invented.
- Why Transformers eventually replaced Seq2Seq RNN models.

This lab is essentially a historical bridge between:

Traditional NLP
→ Seq2Seq
→ Attention
→ Transformers
→ GPT/BERT/Modern LLMs

---

# 2. Complete Bird's-Eye Flow of the Lab

Dataset

↓

German Sentences + English Sentences

↓

Data Preprocessing

↓

Tokenization

↓

Vocabulary Creation

↓

Numerical Encoding

↓

BOS/EOS/PAD Handling

↓

Batch Creation

↓

Encoder Construction

↓

Decoder Construction

↓

Manual Encoder Testing

↓

Manual Decoder Testing

↓

Teacher Forcing Demonstration

↓

Seq2Seq Construction

↓

Training Function Definition

↓

Evaluation Function Definition

↓

Model Initialization

↓

Weight Initialization

↓

Optimizer Definition

↓

Loss Function Definition

↓

Model Training

↓

Save Model

↓

Load Pretrained Model

↓

Inference Function

↓

German → English Translation

↓

BLEU Score Evaluation

↓

Exercise Translation

↓

Observe Model Limitations

↓

Understand Need for Attention

↓

Understand Need for Transformers

---

# 3. Data Preprocessing Stage

Before any neural network can process language, text must be converted into numbers.

The lab first prepares the dataset.

Example:

German:

Ein asiatischer Mann kehrt den Gehweg.

English:

An Asian man is sweeping the sidewalk.

---

## Step 1: Tokenization

Sentence:

Ein asiatischer Mann kehrt den Gehweg

↓

Tokens:

["Ein",
 "asiatischer",
 "Mann",
 "kehrt",
 "den",
 "Gehweg"]

---

## Step 2: Add Special Tokens

The model must know:

- Sentence Start
- Sentence End

Therefore:

<bos>
Ein
asiatischer
Mann
kehrt
den
Gehweg
<eos>

---

## Step 3: Vocabulary Creation

Every unique word receives an ID.

Example:

<bos> → 0

Ein → 12

asiatischer → 45

Mann → 71

kehrt → 89

Gehweg → 130

<eos> → 1

---

## Step 4: Numericalization

Convert words to IDs.

Example:

[0,12,45,71,89,130,1]

---

## Step 5: Padding

Sentences have different lengths.

Neural networks require equal-sized batches.

Therefore:

Shorter sentences get:

<pad>

tokens added.

Example:

<bos>
People
walk
.
<eos>
<pad>
<pad>

---

## What You Observed

The lab displayed batches such as:

German:

<bos> . Innenstadt der in Hüten schwarzen mit Personen <eos>

English:

<bos> People in black hats gathered together downtown . <eos>

This was simply showing:

- Tokenization completed
- Vocabulary mapping completed
- BOS/EOS inserted
- Padding inserted

Dataset ready for training

---

# 4. Encoder Construction

Now the actual neural network begins.

The encoder's job:

Convert an entire German sentence into a numerical representation.

---

## Encoder Inputs

The encoder receives:

German IDs

↓

[0,12,45,71,89,130,1]

---

## Embedding Layer

Words cannot directly enter an LSTM.

First they become dense vectors.

Example:

Word ID:

71

↓

Embedding Vector:

[0.23,
 -0.41,
 0.55,
 ...
]

Dimension:

128 in final model

---

## Why Embeddings?

Because:

Words with similar meaning become numerically closer.

Example:

king ≈ queen

man ≈ woman

dog ≈ puppy

---

## LSTM Encoder

Embedded words enter the encoder LSTM one word at a time.

Word 1
↓

Word 2
↓

Word 3
↓

...

↓

Word N

---

For each step LSTM updates:

- Hidden State
- Cell State

---

## Encoder Output

Final output:

Hidden State

+

Cell State

These two vectors represent the encoder's memory of the sentence.

The actual LSTM outputs are ignored.

Only:

hidden

and

cell

are forwarded to the decoder.

---

# 5. Manual Encoder Test

The lab manually tested the encoder.

Input:

src_batch

↓

Embedding

↓

LSTM

↓

hidden_t

cell_t

---

Purpose:

Verify:

Encoder works correctly before building the complete Seq2Seq architecture.

---

# 6. Decoder Construction

Now the decoder is created.

Purpose:

Generate English words one-by-one.

---

## Decoder Inputs

Decoder receives:

1. Previous English token

2. Hidden State

3. Cell State

---

## Decoder Layers

Embedding
↓

LSTM
↓

Linear Layer
↓

Softmax

↓

Probability Distribution

---

Example Output

The decoder predicts:

P(people)=0.52

P(man)=0.21

P(dog)=0.04

P(street)=0.02

...

Highest probability wins.

---

# 7. Manual Decoder Test

The lab tested decoder separately.

Input:

<bos>

↓

Decoder

↓

Prediction

↓

Hidden Update

↓

Cell Update

---

Purpose:

Verify decoder works before connecting it to encoder.

---

# 8. Teacher Forcing Demonstration

One of the most important concepts.

---

Normally:

Decoder predicts:

Word1

↓

Word2

↓

Word3

↓

Word4

---

But if Word2 is wrong:

Everything after becomes worse.

---

Teacher Forcing solves this.

Instead of feeding:

Predicted Word

You feed:

Actual Correct Word

during training.

---

Example

Actual:

An Asian man is sweeping

Model predicts:

An Asian dog ...

Teacher forcing forces:

man

into next step.

This stabilizes training.

---

# 9. Building Seq2Seq Architecture

Now encoder and decoder are connected.

---

Flow:

German Sentence

↓

Encoder

↓

Hidden State

+

Cell State

↓

Decoder

↓

English Sentence

---

This is the first complete translation system.

---

# 10. Seq2Seq Forward Pass

For every training batch:

Step 1:

Encoder processes source sentence.

↓

Produces:

hidden

cell

---

Step 2:

Decoder starts with:

<bos>

---

Step 3:

Predict next English word.

---

Step 4:

Use Teacher Forcing decision.

Either:

Actual next word

or

Predicted word

becomes next decoder input.

---

Step 5:

Repeat until:

<eos>

or

sequence ends.

---

# 11. Model Initialization

Final model dimensions:

Input Vocabulary:

~19,214 German words

Output Vocabulary:

~10,837 English words

Embedding Size:

128

Hidden Size:

256

Layers:

1

Dropout:

0.3

---

# 12. Weight Initialization

Every neural network begins with random weights.

Lab used:

Uniform Distribution

between:

-0.08

and

0.08

---

Purpose:

Avoid bad starting conditions.

Ensure stable learning.

---

# 13. Model Size

The lab calculated:

7,422,165 Trainable Parameters

---

These parameters exist inside:

- Encoder Embeddings
- Decoder Embeddings
- Encoder LSTM
- Decoder LSTM
- Output Layer

---

# 14. Optimizer

Optimizer:

Adam

Purpose:

Update parameters after backpropagation.

---

Flow:

Loss

↓

Gradients

↓

Adam

↓

Weight Updates

---

# 15. Loss Function

CrossEntropyLoss

Purpose:

Compare:

Predicted Word

vs

Actual Word

---

Smaller Loss

=

Better Translation

---

# 16. Training Stage

Actual training loop:

Batch

↓

Forward Pass

↓

Prediction

↓

Loss

↓

Backward Pass

↓

Gradient Calculation

↓

Gradient Clipping

↓

Adam Update

↓

Next Batch

---

Repeated for:

Multiple Epochs

---

# 17. Gradient Clipping

Why?

LSTMs suffer from:

Exploding Gradients

Huge gradient values can destroy learning.

Therefore:

Gradient Norm is clipped.

---

# 18. Evaluation Stage

After training:

No optimization.

Only testing.

---

Teacher Forcing:

OFF

because real-world translation won't know correct answers.

---

Model must generate words entirely on its own.

---

# 19. Pretrained Model

Very important observation.

The lab mostly skipped real training.

Instead:

IBM provided:

RNN-TR-model.pt

---

Why?

Because:

Training takes a long time.

Especially on CPU.

---

Therefore:

Training section is mainly educational.

The actual translation demonstrations used:

Pretrained Weights

---

# 20. What Was Actually Executed?

Conceptually:

You built:

Encoder
Decoder
Seq2Seq
Training Logic

---

Practically:

You loaded:

Pretrained Model

↓

Inference

↓

Translation

---

Meaning:

The architecture was built.

The heavy learning had already happened earlier.

---

# 21. Inference Stage

Now translation begins.

Input:

German Sentence

↓

Encoder

↓

Context Vector

↓

Decoder

↓

English Sentence

---

Example

Input:

Menschen gehen auf der Straße

Output:

A people are walking on the street in a city.

---

# 22. BLEU Score Evaluation

BLEU compares:

Generated Translation

vs

Reference Translations

---

Higher BLEU

=

Closer to human translation

---

Purpose:

Automatic translation quality measurement.

---

# 23. Critical Observation

Input:

Ein asiatischer Mann kehrt den Gehweg.

Expected:

An Asian man is sweeping the sidewalk.

Model Output:

This is five workers are working on a bench.

---

Clearly incorrect.

---

# 24. Why Did It Fail?

Because classic Seq2Seq compresses:

Entire German sentence

↓

Single Hidden State

↓

Single Cell State

---

Information Bottleneck occurs.

Important details get lost.

---

Longer sentences become difficult.

Complex relationships disappear.

---

# 25. Birth of Attention

Researchers realized:

One fixed context vector is insufficient.

---

Instead:

Allow decoder to look back at encoder outputs.

At every decoding step.

---

Now decoder can focus on:

Mann

↓

man

and later

Gehweg

↓

sidewalk

when needed.

---

Translation quality improves dramatically.

---

# 26. Birth of Transformers

Attention worked so well that researchers asked:

Why keep RNNs at all?

---

Transformers removed:

- RNN
- LSTM
- Sequential Processing

and kept:

Attention

only.

---

Benefits:

Better Context

Parallel Training

Long-Range Understanding

Higher Accuracy

Much Faster Training

---

# 27. Historical Evolution

Rule-Based Translation

↓

Statistical Machine Translation

↓

RNN

↓

LSTM

↓

Seq2Seq

↓

Seq2Seq + Attention

↓

Transformer

↓

BERT

↓

GPT

↓


Modern LLMs

---

# Final Takeaway

This lab is one of the most important historical NLP labs.

The real lesson is NOT:

"How to build a translator."

The real lesson is:

"How modern language models evolved."

You learned:

- Tokenization
- Vocabulary Mapping
- Embeddings
- Encoder
- Decoder
- Hidden State
- Cell State
- Teacher Forcing
- Seq2Seq
- Training Loop
- Inference
- BLEU Evaluation
- Context Bottleneck Problem
- Motivation for Attention
- Motivation for Transformers

Seq2Seq represents the final major milestone before the Transformer revolution that eventually led to BERT, GPT, ChatGPT, Claude, Gemini, and modern Generative AI systems.
