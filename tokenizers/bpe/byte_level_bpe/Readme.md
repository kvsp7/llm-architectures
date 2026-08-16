# Byte-Level BPE

A **from-scratch implementation of Byte-Level Byte Pair Encoding (BPE) in Python**, built to understand how modern subword tokenizers work internally.

Instead of starting with words or characters, this implementation starts with the **256 possible byte values** and repeatedly merges the most frequent adjacent pair to learn larger tokens.

> **Training data:** This implementation is currently trained on an **English-only corpus**.  
> **Tokenization:** Because it operates on UTF-8 bytes, it can still represent Unicode text such as Telugu, Hindi, Japanese, Chinese, and emojis. However, its learned merge rules are optimized for English because the training corpus is English-only.

The goal of this project is **understanding the algorithm from first principles**, rather than using an existing tokenizer library.

---

## Table of Contents

- [Overview](#overview)
- [What Is Byte-Level BPE?](#what-is-byte-level-bpe)
- [How It Works](#how-it-works)
- [Training Data](#training-data)
- [Step-by-Step Training](#step-by-step-training)
- [Example](#example)
- [Encoding](#encoding)
- [Decoding](#decoding)
- [Unicode Support](#unicode-support)
- [Implementation](#implementation)
- [Merge Representation](#merge-representation)
- [Example Usage](#example-usage)
- [Testing](#testing)
- [Word-Level BPE vs Byte-Level BPE](#word-level-bpe-vs-byte-level-bpe)
- [Byte-Level BPE vs GPT-2](#byte-level-bpe-vs-gpt-2)
- [Complexity](#complexity)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Repository Context](#repository-context)
- [Status](#status)

---

# Overview

Tokenization is the process of converting text into smaller units called **tokens** that a language model can process.

For example:

```text
"hello world"
       ↓
   Tokenizer
       ↓
[hello] [world]
```

Different tokenization algorithms use different starting representations.

A character-based tokenizer might start with:

```text
hello
  ↓
h e l l o
```

A word-based tokenizer might start with:

```text
hello world
  ↓
hello | world
```

A **Byte-Level BPE tokenizer** starts with the UTF-8 bytes of the text:

```text
hello
  ↓
UTF-8
  ↓
[104, 101, 108, 108, 111]
```

These 256 possible byte values form the initial vocabulary.

BPE then learns useful combinations of these bytes.

For example:

```text
(108, 111) → 256
```

represents:

```text
l + o → lo
```

A later merge can combine the newly created token:

```text
(256, 119) → 257
```

representing:

```text
lo + w → low
```

The tokenizer therefore gradually learns larger and more useful subword units.

---

# What Is Byte-Level BPE?

**Byte Pair Encoding (BPE)** is an iterative vocabulary-learning algorithm.

The basic idea is:

1. Start with a small vocabulary.
2. Find the most frequent adjacent pair.
3. Merge that pair into a new token.
4. Repeat.

In this implementation, the initial vocabulary contains the **256 possible byte values**:

```text
0, 1, 2, ..., 255
```

Newly learned tokens start at:

```text
256
```

For example:

```text
108 → 'l'
111 → 'o'

(108, 111) → 256
```

The next merge can use token `256`:

```text
256 + 119 → 257
```

So BPE can build tokens hierarchically:

```text
l + o
  ↓
 lo
  ↓
lo + w
  ↓
 low
```

---

# How It Works

The complete training pipeline is:

```text
                 English Training Corpus
                          │
                          ▼
                    UTF-8 Encoding
                          │
                          ▼
                     Byte Sequences
                          │
                          ▼
                 Initial Token IDs
                          │
                          ▼
                 Count Adjacent Pairs
                          │
                          ▼
              Find Most Frequent Pair
                          │
                          ▼
                     Merge Pair
                          │
                          ▼
                  Assign New Token ID
                          │
                          ▼
                    Repeat N Times
                          │
                          ▼
                 Learned Merge Rules
```

After training, the tokenizer has learned a set of rules such as:

```text
(108, 111) → 256
(256, 119) → 257
...
```

These rules are then used to tokenize new text.

---

# Training Data

The current implementation is trained using an **English-only text corpus**.

For example:

```text
The quick brown fox jumps over the lazy dog.
Machine learning is a field of artificial intelligence.
Large language models learn patterns from text.
```

The training corpus determines which byte combinations become useful tokens.

Because the current corpus is English-focused, the tokenizer will generally learn merges that are useful for:

- English letters
- English words
- common English subwords
- punctuation
- spaces
- common patterns appearing in the corpus

### Important: Byte-Level Does Not Mean Multilingual Training

The tokenizer can still process text such as:

```text
తెలుగు
नमस्ते
こんにちは
你好
🚀🤖
```

because these characters can be represented as UTF-8 bytes.

However, the tokenizer was **not trained on these languages**.

Therefore, it may not have learned efficient multi-byte merges for them.

Conceptually:

```text
English training data
        ↓
Learn English-oriented merges
        ↓
Byte-Level BPE vocabulary
        ↓
Can represent Unicode
        ↓
But non-English text may require more tokens
```

This distinction is important.

**Unicode representation is not the same thing as multilingual training.**

---

# Step-by-Step Training

## 1. Convert Text to UTF-8 Bytes

Each training line is converted into UTF-8:

```python
line.encode("utf-8")
```

For example:

```text
hello
```

becomes:

```text
[104, 101, 108, 108, 111]
```

The tokenizer therefore does not directly operate on Python characters.

It operates on byte values.

---

## 2. Start With the Initial Vocabulary

There are exactly 256 possible byte values:

```text
0
1
2
...
255
```

Therefore:

```python
self.next_id = 256
```

The first learned BPE token receives ID:

```text
256
```

The next receives:

```text
257
```

and so on.

---

## 3. Count Adjacent Pairs

Consider:

```text
[104, 101, 108, 108, 111]
```

The adjacent pairs are:

```text
(104, 101)
(101, 108)
(108, 108)
(108, 111)
```

The tokenizer counts how frequently each pair occurs throughout the training corpus.

For example:

```text
Pair              Frequency

(108, 111)           500
(104, 101)           420
(108, 108)           310
...
```

---

## 4. Select the Most Frequent Pair

The most frequent pair is selected.

Conceptually:

```python
best_pair = max(pairs, key=pairs.get)
```

Suppose:

```text
(108, 111)
```

is the most frequent pair.

---

## 5. Assign a New Token ID

The selected pair receives a new token ID:

```text
(108, 111) → 256
```

The tokenizer records this merge rule.

---

## 6. Merge the Pair

Every occurrence of:

```text
(108, 111)
```

is replaced with:

```text
256
```

So:

```text
[108, 111, 119]
```

becomes:

```text
[256, 119]
```

---

## 7. Repeat

The tokenizer continues:

```text
Count pairs
     ↓
Find most frequent pair
     ↓
Create new token
     ↓
Merge pair
     ↓
Repeat
```

until the requested number of merges has been reached or there are no useful pairs left according to the implementation's stopping condition.

---

# Example

Consider the simplified training corpus:

```text
low low lower
```

The word:

```text
low
```

is represented as:

```text
[108, 111, 119]
```

Suppose the most frequent pair is:

```text
(108, 111)
```

The tokenizer learns:

```text
(108, 111) → 256
```

Therefore:

```text
[108, 111, 119]
```

becomes:

```text
[256, 119]
```

Now suppose the next frequent pair is:

```text
(256, 119)
```

The tokenizer learns:

```text
(256, 119) → 257
```

The sequence becomes:

```text
[257]
```

So the tokenizer has learned:

```text
l + o → 256
256 + w → 257
```

Conceptually:

```text
l   o   w
│   │   │
└───┘   │
  256   │
   └────┘
    257
```

This illustrates the fundamental mechanism of BPE.

---

# Encoding

After training, the learned merge rules are used to tokenize new text.

The encoding pipeline is:

```text
Input Text
    ↓
UTF-8 Encoding
    ↓
Initial Byte IDs
    ↓
Apply Learned Merge #1
    ↓
Apply Learned Merge #2
    ↓
Apply Learned Merge #3
    ↓
...
    ↓
Final Token IDs
```

Suppose the learned rules are:

```text
(108, 111) → 256
(256, 119) → 257
```

Encoding:

```text
"low"
```

starts with:

```text
[108, 111, 119]
```

Apply merge #1:

```text
[256, 119]
```

Apply merge #2:

```text
[257]
```

Therefore:

```text
"low" → [257]
```

The token ID `257` represents the learned byte sequence corresponding to `"low"`.

---

# Decoding

Decoding reverses the learned merges.

Suppose the tokenizer has:

```text
256 → (108, 111)
257 → (256, 119)
```

Given:

```text
[257]
```

the decoder expands:

```text
257
 ↓
256, 119
 ↓
108, 111, 119
```

The byte sequence becomes:

```text
[108, 111, 119]
```

which is converted back to text:

```python
bytes(ids).decode("utf-8")
```

Result:

```text
low
```

The fundamental property is:

```python
text == tokenizer.decode(tokenizer.encode(text))
```

For valid UTF-8 text, this should reconstruct the original text.

---

# Unicode Support

Because the tokenizer starts from **bytes**, it can represent Unicode text through UTF-8.

For example:

```text
Hello 世界
```

is first converted into UTF-8 bytes.

Similarly:

```text
తెలుగు
```

```text
नमस्ते
```

```text
こんにちは
```

and:

```text
🚀🤖
```

can all be represented as UTF-8 byte sequences.

The overall process is:

```text
Unicode Text
     ↓
UTF-8
     ↓
Bytes
     ↓
BPE
     ↓
Token IDs
```

### Important

The tokenizer's ability to represent these languages comes from **UTF-8 byte representation**, not from multilingual training.

Since this implementation was trained on an English-only corpus:

```text
English
   ↓
Well-learned English merges

Other languages
   ↓
Representable
   ↓
But potentially less efficient
```

This makes the implementation useful for demonstrating why byte-level tokenization avoids the traditional "unknown character" problem while also showing the importance of the training corpus.

---

# Implementation

The implementation is divided into two main classes.

```text
Tokenizer
    │
    ├── preprocessing()
    ├── generate_vocabulary()
    ├── generate_pairs()
    ├── merge_pair()
    └── train_tokenizer()
    
BPETokenizer
    │
    ├── encode()
    └── decode()
```

---

## `Tokenizer`

The `Tokenizer` class is responsible for learning the BPE vocabulary and merge rules.

### `preprocessing()`

Converts the training text into UTF-8 byte sequences.

```text
Text
 ↓
UTF-8
 ↓
Bytes
```

---

### `generate_vocabulary()`

Builds the initial vocabulary representation and tracks the training sequences.

The initial token space is based on:

```text
0–255
```

---

### `generate_pairs()`

Finds adjacent token pairs and calculates their frequencies.

For example:

```text
[10, 20, 30]
```

produces:

```text
(10, 20)
(20, 30)
```

---

### `merge_pair()`

Replaces occurrences of the selected pair with a newly created token ID.

For example:

```text
(108, 111) → 256
```

---

### `train_tokenizer()`

Coordinates the complete BPE training process:

```text
Preprocess
   ↓
Build vocabulary
   ↓
Count pairs
   ↓
Select pair
   ↓
Merge
   ↓
Repeat
```

---

# `BPETokenizer`

The `BPETokenizer` class is responsible for applying the learned tokenizer.

It provides:

```text
encode()
decode()
```

### `encode()`

Converts input text into token IDs using the learned merge rules.

### `decode()`

Expands the learned tokens back into their byte representation and reconstructs the original UTF-8 text.

---

# Merge Representation

Learned merge rules are stored in:

```python
self.merged_pairs
```

For example:

```text
(108, 111) → 256
(256, 119) → 257
```

The order in which merges were learned is stored separately:

```python
self.merge_order
```

For example:

```text
1. (108, 111)
2. (256, 119)
3. ...
```

### Why Is Merge Order Important?

BPE is sequential.

A later merge may depend on an earlier merge.

For example:

```text
(108, 111) → 256
```

must happen before:

```text
(256, 119) → 257
```

because token `256` does not exist until the first merge has been performed.

Therefore the encoder must reproduce the learned merge order.

---

# Example Usage

A basic example using the English training corpus:

```python
with open("The-verdict.txt", "r", encoding="utf-8") as file:
    corpus = [
        line.rstrip("\n")
        for line in file
        if line.strip()
    ]

tokenizer = BPETokenizer(
    corpus=corpus,
    merges=20000
)

text = "Implementration of Byte-Level-BPE"

ids = tokenizer.encode(text)

print("Token IDs:")
print(ids)

decoded = tokenizer.decode(ids)

print("Decoded:")
print(decoded)
```

Expected output:

```text
Token IDs:
[...]

Decoded:
Implementration of Byte-Level-BPE
```

The most important correctness check is:

```python
assert text == tokenizer.decode(tokenizer.encode(text))
```

If this passes, the tokenizer successfully performs the encode/decode round trip.

---

# Testing

The tokenizer should be tested with different categories of text.

## ASCII

```python
"Hello world!"
```

## Punctuation

```python
"Hello, world!"
```

## Numbers

```python
"Machine Learning 2026"
```

## English Sentence

```python
"Byte-Level BPE is a tokenizer algorithm."
```

## Telugu

```python
"నమస్కారం"
```

## Hindi

```python
"नमस्ते"
```

## Japanese

```python
"こんにちは"
```

## Chinese

```python
"你好世界"
```

## Emoji

```python
"AI 🚀🤖"
```

## Mixed Unicode

```python
"Hello 世界! AI 🚀"
```

## Mixed Languages

```python
"Hello नमस्ते こんにちは తెలుగు 🚀"
```

For every valid UTF-8 input:

```python
decoded = tokenizer.decode(
    tokenizer.encode(text)
)

assert decoded == text
```

This checks whether encoding followed by decoding reconstructs the original text.

### Token Efficiency

Correct decoding is not the only useful test.

It is also useful to measure how many tokens are produced:

```text
Input text
    ↓
Tokenizer
    ↓
Token IDs
    ↓
Number of tokens
```

For example:

```python
ids = tokenizer.encode(text)

print("Number of tokens:", len(ids))
```

Because the current tokenizer was trained on English-only data, English text will generally be tokenized more efficiently than languages that were not represented in the training corpus.

---

# Word-Level BPE vs Byte-Level BPE

The main difference is the initial representation.

## Word-Level BPE

A simplified pipeline is:

```text
Text
 ↓
Words
 ↓
Characters / Subwords
 ↓
BPE
 ↓
Tokens
```

The tokenizer depends on how text is split into words.

---

## Byte-Level BPE

The pipeline is:

```text
Text
 ↓
UTF-8
 ↓
Bytes
 ↓
BPE
 ↓
Tokens
```

The initial representation consists of byte values rather than words.

This gives byte-level tokenization an important property:

> Any valid UTF-8 text can be represented using the initial byte vocabulary.

However, the quality and efficiency of the learned tokens still depend heavily on the training corpus.

---

# Byte-Level BPE vs GPT-2

This implementation is **conceptually similar to the core idea of byte-level BPE**, but it is **not an exact reproduction of the GPT-2 tokenizer**.

GPT-2 uses byte-level BPE together with additional tokenizer-specific mechanisms.

A production tokenizer can include:

- byte-to-Unicode mappings
- regular-expression pre-tokenization
- special token handling
- vocabulary files
- ranked merge rules
- optimized encoding
- tokenizer serialization
- model-specific conventions

This repository intentionally focuses on the fundamental learning mechanism:

```text
Bytes
  ↓
Pair frequency
  ↓
Most frequent pair
  ↓
Merge
  ↓
New token
  ↓
Repeat
```

Therefore:

```text
This project
      ≈
Core Byte-Level BPE concept

This project
      ≠
Exact GPT-2 tokenizer implementation
```

The purpose is to understand the algorithm rather than reproduce every engineering detail of a production tokenizer.

---

# Complexity

The current implementation repeatedly performs operations such as:

1. Counting adjacent pair frequencies.
2. Finding the most frequent pair.
3. Scanning sequences to apply the merge.
4. Repeating this process for each merge.

This makes the implementation suitable for:

- learning
- experimentation
- debugging
- understanding BPE
- small research experiments

It is **not optimized for very large-scale tokenizer training**.

Production tokenizers use more efficient data structures and algorithms to update pair statistics without repeatedly scanning the entire corpus.

---

# Limitations

The current implementation has several limitations.

- Training corpus is currently English-only.
- No optimized pair-frequency updates.
- Basic training implementation.
- Sequential merge application.
- No special-token framework.
- No vocabulary serialization.
- No tokenizer save/load functionality.
- No optimized inference implementation.
- No production-grade pre-tokenization.
- No extensive benchmark suite.
- Not intended for production LLM training.

The English-only training corpus is particularly important:

```text
Byte-level representation
        ≠
Multilingual training
```

The tokenizer can represent multilingual text, but it has not learned language-specific merges for those languages.

---

# Future Improvements

Planned improvements include:

- [ ] Add tokenizer save/load functionality
- [ ] Add vocabulary serialization
- [ ] Add special tokens
- [ ] Add automated unit tests
- [ ] Add merge-rank optimization
- [ ] Optimize pair-frequency updates
- [ ] Benchmark training speed
- [ ] Benchmark encoding speed
- [ ] Measure token compression
- [ ] Compare token efficiency across languages
- [ ] Train on a multilingual corpus
- [ ] Compare against existing tokenizer implementations
- [ ] Integrate the tokenizer with a Transformer
- [ ] Use the tokenizer in an end-to-end language model

---

# Repository Context

This tokenizer is part of the:

```text
llm-architectures
```

repository.

The broader goal of the repository is to implement and understand the fundamental components behind modern Large Language Models **from scratch**.

A possible tokenizer progression within the repository is:

```text
Word-Level BPE
      ↓
Byte-Level BPE
      ↓
WordPiece
      ↓
Unigram
      ↓
Production-Style Tokenizers
      ↓
Transformer / LLM
```

The objective is not simply to call an existing tokenizer library.

Instead, the goal is to understand:

- how tokenizers represent text
- how vocabularies are constructed
- how merge rules are learned
- how token IDs are assigned
- how encoding works
- how decoding reconstructs text
- why different LLM architectures use different tokenization strategies

---

# Why Build BPE From Scratch?

Modern LLMs process token IDs rather than raw text.

A simplified language-model pipeline is:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embeddings
   ↓
Transformer
   ↓
Logits
   ↓
Next Token
```

The tokenizer is therefore one of the first components in the LLM pipeline.

Implementing it from scratch makes it easier to understand what happens between:

```text
Text
```

and:

```text
Token IDs
```

before the Transformer ever sees the input.

This project is intended to make that process transparent.

---

# Status

**Implementation:** Complete — basic Byte-Level BPE

**Training Corpus:** English-only

**Encoding:** Implemented

**Decoding:** Implemented

**Unicode Representation:** Supported through UTF-8

**Multilingual Training:** Not currently implemented

**Focus:** Algorithmic understanding and from-scratch implementation

**Production Ready:** No

**Next Steps:** Optimization, testing, multilingual experiments, and Transformer integration

---

## Key Takeaway

The core idea of this implementation can be summarized in one pipeline:

```text
                Text
                 ↓
              UTF-8
                 ↓
               Bytes
                 ↓
        Find frequent pairs
                 ↓
             Merge pair
                 ↓
          Create new token
                 ↓
             Repeat
                 ↓
         Learned vocabulary
                 ↓
          Encode new text
                 ↓
             Token IDs
```

The most important concept is:

> **Byte-Level BPE starts with bytes and learns larger tokens by repeatedly merging frequent adjacent pairs.**

This implementation provides a simple way to see that process directly, without hiding the algorithm behind a tokenizer library.