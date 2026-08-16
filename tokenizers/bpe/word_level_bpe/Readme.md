# Word-Level BPE

A **from-scratch implementation of Word-Level Byte Pair Encoding (BPE) in Python**, built to understand how subword tokenization works internally.

This implementation starts with **pre-tokenized words**, decomposes each word into individual characters, and repeatedly merges the most frequent adjacent pair to learn larger subword units.

The goal is to understand the **core BPE algorithm from first principles** rather than relying on an existing tokenizer library.

---

## Table of Contents

- [Overview](#overview)
- [What Is BPE?](#what-is-bpe)
- [How It Works](#how-it-works)
- [Step-by-Step Training](#step-by-step-training)
- [Example](#example)
- [Encoding](#encoding)
- [Decoding](#decoding)
- [Implementation](#implementation)
- [Merge Representation](#merge-representation)
- [Example Usage](#example-usage)
- [Testing](#testing)
- [Word-Level BPE vs Byte-Level BPE](#word-level-bpe-vs-byte-level-bpe)
- [Difference From Production LLM Tokenizers](#difference-from-production-llm-tokenizers)
- [Complexity](#complexity)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Repository Context](#repository-context)
- [Status](#status)

---

# Overview

Tokenization converts raw text into smaller units called **tokens** that can later be converted into integer IDs and processed by a language model.

A simple character representation of:

```text
hello
```

is:

```text
h e l l o
```

BPE starts with these small symbols and learns larger, frequently occurring combinations.

For example:

```text
l + o → lo
```

Then:

```text
lo + w → low
```

This allows the tokenizer to represent text using a mixture of:

- characters
- subwords
- complete words

depending on which patterns were learned during training.

---

# What Is BPE?

**Byte Pair Encoding (BPE)** is an iterative vocabulary-learning algorithm.

The basic idea is:

```text
Start with small symbols
        ↓
Count adjacent pairs
        ↓
Find the most frequent pair
        ↓
Merge the pair
        ↓
Create a new symbol
        ↓
Repeat
```

For Word-Level BPE, the initial symbols are characters inside pre-tokenized words.

For example:

```text
lower
```

is represented as:

```text
l o w e r </w>
```

where:

```text
</w>
```

represents the **end of the word**.

The tokenizer can then learn:

```text
l + o → lo
```

followed by:

```text
lo + w → low
```

and potentially:

```text
low + e → lowe
```

depending on the training corpus and merge frequencies.

---

# How It Works

The complete pipeline is:

```text
                    Raw Corpus
                        │
                        ▼
                   Preprocessing
                        │
                        ▼
                  Word Vocabulary
                        │
                        ▼
               Character Decomposition
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
                 Assign Token
                        │
                        ▼
                   Repeat N Times
                        │
                        ▼
                Learned Merge Rules
```

After training, the learned merge rules are used to encode new text.

---

# Step-by-Step Training

## 1. Preprocessing

The raw corpus is first processed into words and punctuation.

For example:

```text
Hello, world!
```

can be separated approximately into:

```text
Hello
,
world
!
```

The exact behavior depends on the preprocessing rules implemented by the tokenizer.

This step is important because Word-Level BPE operates on **pre-tokenized units**.

---

## 2. Build the Word Vocabulary

Each word is represented as individual characters followed by an end-of-word marker.

For example:

```text
hello
```

becomes:

```text
h e l l o </w>
```

Another word:

```text
lower
```

becomes:

```text
l o w e r </w>
```

The tokenizer also tracks how frequently each word occurs in the training corpus.

Conceptually:

```text
Word                  Frequency

hello                    100
world                     80
lower                     40
...
```

The frequency information is used when calculating pair frequencies.

---

## 3. Count Adjacent Pairs

Consider:

```text
h e l l o </w>
```

The adjacent pairs are:

```text
(h, e)
(e, l)
(l, l)
(l, o)
(o, </w>)
```

The tokenizer counts these pairs across the entire training vocabulary.

For example:

```text
Pair                 Frequency

(l, o)                   500
(e, r)                   420
(h, e)                   350
...
```

---

## 4. Select the Most Frequent Pair

The tokenizer selects the pair with the highest frequency.

Conceptually:

```python
best_pair = max(pairs, key=pairs.get)
```

Suppose the most frequent pair is:

```text
(l, o)
```

---

## 5. Merge the Pair

The selected pair becomes a new symbol:

```text
l + o → lo
```

Therefore:

```text
l o w e r </w>
```

becomes:

```text
lo w e r </w>
```

The newly created symbol can participate in future merges.

For example:

```text
lo + w → low
```

producing:

```text
low e r </w>
```

This is what allows BPE to build larger subword units incrementally.

---

## 6. Repeat

The process continues:

```text
Count pairs
     ↓
Find most frequent pair
     ↓
Merge pair
     ↓
Update vocabulary
     ↓
Count pairs again
     ↓
Repeat
```

The process continues until the requested number of merges has been performed or the implementation's stopping condition is reached.

---

# Example

Consider a simplified training corpus:

```text
low low lower
```

The word:

```text
low
```

starts as:

```text
l o w </w>
```

Suppose:

```text
(l, o)
```

is the most frequent pair.

The tokenizer learns:

```text
(l, o) → lo
```

The sequence becomes:

```text
lo w </w>
```

Suppose the next frequent pair is:

```text
(lo, w)
```

The tokenizer learns:

```text
(lo, w) → low
```

Now:

```text
low </w>
```

The learned merge sequence is therefore:

```text
l + o → lo
lo + w → low
```

Conceptually:

```text
l   o   w   </w>
│   │   │
└───┘   │
 lo     │
 └──────┘
   low
```

The tokenizer has learned a larger unit from smaller units.

---

# Encoding

After training, the learned merge rules are applied to new text.

The encoding pipeline is:

```text
Input Text
    ↓
Preprocessing
    ↓
Words / Punctuation
    ↓
Characters + </w>
    ↓
Apply Learned Merge Rules
    ↓
Subword Tokens
    ↓
Integer Token IDs
```

For example, suppose the tokenizer has learned:

```text
l + o → lo
lo + w → low
```

The input:

```text
low
```

starts as:

```text
l o w </w>
```

After the first merge:

```text
lo w </w>
```

After the second merge:

```text
low </w>
```

The resulting tokens are then mapped to integer IDs.

For example:

```text
low → 421
```

The exact IDs depend on the vocabulary learned during training.

---

# Decoding

Decoding performs the reverse high-level operation.

The process is:

```text
Token IDs
    ↓
Token Strings
    ↓
Concatenate Tokens
    ↓
Interpret </w>
    ↓
Reconstruct Words
    ↓
Original Text
```

For example:

```text
[421, 52, 817]
```

might correspond to:

```text
["low</w>", ",", "hello</w>"]
```

which can be reconstructed into:

```text
low, hello
```

The exact token IDs and representations depend on the learned vocabulary.

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

Processes the raw training corpus into words and punctuation.

Conceptually:

```text
Raw Text
   ↓
Preprocessing
   ↓
Words / Punctuation
```

---

### `generate_vocabulary()`

Builds the initial word vocabulary.

Each word is decomposed into characters and receives an end-of-word marker.

For example:

```text
hello
```

becomes:

```text
h e l l o </w>
```

The frequency of each word is also tracked.

---

### `generate_pairs()`

Calculates the frequency of adjacent symbol pairs across the vocabulary.

For example:

```text
h e l l o </w>
```

produces:

```text
(h, e)
(e, l)
(l, l)
(l, o)
(o, </w>)
```

---

### `merge_pair()`

Merges the selected pair throughout the vocabulary.

For example:

```text
(l, o)
```

becomes:

```text
lo
```

---

### `train_tokenizer()`

Coordinates the complete BPE training process.

```text
Preprocess
    ↓
Build vocabulary
    ↓
Count pairs
    ↓
Select best pair
    ↓
Merge
    ↓
Repeat
```

---

# `BPETokenizer`

The `BPETokenizer` class uses the trained vocabulary and merge rules to tokenize new text.

It provides:

```text
encode()
decode()
```

### `encode()`

Converts input text into integer token IDs.

### `decode()`

Converts token IDs back into tokens and reconstructs the original text.

---

# Merge Representation

BPE is a sequential algorithm, so the order in which merge rules are learned matters.

For example:

```text
1. l + o → lo
2. lo + w → low
```

The second merge depends on the first.

Therefore the tokenizer must preserve the learned merge order when encoding new text.

Conceptually:

```text
Merge #1
(l, o) → lo

Merge #2
(lo, w) → low
```

This allows the encoder to reproduce the same sequence of transformations learned during training.

---

# Example Usage

A basic example:

```python
with open("The-verdict.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

tokenizer = BPETokenizer(
    data=corpus,
    merges=30000
)

text = "Hey, How are you..."

ids = tokenizer.encode(text)

print("Token IDs:")
print(ids)

decoded = tokenizer.decode(ids)

print("Decoded:")
print(decoded)
```

Expected output will depend on the training corpus and learned vocabulary:

```text
Token IDs:
[...]

Decoded:
Hey, How are you...
```

The most important correctness condition is:

```python
assert text == tokenizer.decode(tokenizer.encode(text))
```

This verifies the encode/decode round trip.

---

# Testing

The tokenizer should be tested using different kinds of text.

## Simple English

```python
"Hello world!"
```

## Punctuation

```python
"Hello, world!"
```

## Repeated Words

```python
"low low lower lower"
```

## Numbers

```python
"Machine Learning 2026"
```

## Unknown or Rare Words

```python
"Xylophonically"
```

This helps test the `<unk>` behavior of the tokenizer.

## Mixed Text

```python
"Hello, AI!"
```

For every supported input:

```python
decoded = tokenizer.decode(
    tokenizer.encode(text)
)

assert decoded == text
```

---

# Implementation Details

| Component | Implementation |
|---|---|
| Algorithm | BPE |
| Tokenization | Word-Level |
| Initial Representation | Characters |
| Word Boundary | `</w>` |
| Pair Selection | Highest frequency |
| Merge Strategy | Greedy |
| Unknown Token | `<unk>` |
| Encoding | Tokens → Integer IDs |
| Decoding | Integer IDs → Text |
| Training Data | Pre-tokenized corpus |
| Implementation | From scratch |

---

# Word-Level BPE vs Byte-Level BPE

Both algorithms use the same fundamental BPE idea:

```text
Find frequent pair
       ↓
Merge pair
       ↓
Repeat
```

The major difference is **what they start with**.

## Word-Level BPE

This implementation follows:

```text
Text
 ↓
Words
 ↓
Characters
 ↓
BPE
 ↓
Subword Tokens
```

For example:

```text
hello
 ↓
h e l l o </w>
 ↓
he l l o </w>
 ↓
hel l o </w>
 ↓
hello </w>
```

---

## Byte-Level BPE

Byte-Level BPE follows:

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

For example:

```text
hello
 ↓
UTF-8
 ↓
[104, 101, 108, 108, 111]
 ↓
BPE merges
 ↓
Token IDs
```

### Main Difference

| Feature | Word-Level BPE | Byte-Level BPE |
|---|---|---|
| Initial units | Characters | Bytes |
| Pre-tokenization | Required | Can operate from bytes |
| Word boundary | `</w>` | Represented through byte-level sequence / tokenizer rules |
| Initial vocabulary | Character-based | 256 bytes |
| Unknown handling | Often needs `<unk>` | Byte representation can represent arbitrary UTF-8 input |
| Unicode representation | Depends on character vocabulary | UTF-8 bytes |
| This repository | Implemented | Implemented separately |

The two implementations demonstrate how changing the initial representation changes tokenizer behavior.

---

# Difference From Production LLM Tokenizers

This project is an educational implementation of the core BPE algorithm.

It should **not** be considered an exact implementation of a production tokenizer used by GPT, BERT, or another modern LLM.

Production tokenizers may include:

- sophisticated pre-tokenization
- regular-expression rules
- special tokens
- optimized vocabulary structures
- ranked merge rules
- optimized encoding
- tokenizer serialization
- language-specific handling
- normalization
- model-specific conventions

This implementation intentionally keeps the algorithm visible and understandable.

The objective is:

```text
Understand the algorithm
        ↓
Understand the implementation
        ↓
Understand production tokenizers
```

rather than simply calling a tokenizer library.

---

# Complexity

The current implementation repeatedly performs operations such as:

1. Counting pair frequencies.
2. Finding the most frequent pair.
3. Updating the vocabulary.
4. Applying the merge.
5. Repeating for each merge.

Therefore, the implementation is primarily designed for:

- learning
- experimentation
- debugging
- small-scale research
- understanding BPE

It is not optimized for large-scale production tokenizer training.

Production implementations use more efficient data structures and algorithms for maintaining pair statistics and merge rankings.

---

# Limitations

Current limitations include:

- Word-level preprocessing.
- Character-based initial representation.
- Basic regular-expression preprocessing.
- Sequential application of learned merge rules.
- Basic `<unk>` handling.
- No special-token framework.
- No byte-level representation.
- No optimized pair-ranking structure.
- No tokenizer serialization.
- No save/load functionality.
- No production-grade preprocessing.
- Not optimized for large-scale training.

These limitations are intentional because the primary goal is **algorithmic understanding**.

---

# Future Work

Planned improvements include:

- [ ] Add tokenizer save/load functionality
- [ ] Add vocabulary serialization
- [ ] Add special tokens
- [ ] Improve preprocessing
- [ ] Add automated unit tests
- [ ] Optimize merge ranking
- [ ] Benchmark training speed
- [ ] Benchmark encoding speed
- [ ] Measure token efficiency
- [ ] Compare Word-Level and Byte-Level tokenization
- [ ] Implement WordPiece
- [ ] Implement Unigram
- [ ] Compare against existing tokenizer implementations
- [ ] Integrate tokenizers with Transformer implementations

The tokenizer progression in this repository is:

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
```

---

# Repository Context

This implementation is part of the:

```text
llm-architectures
```

repository.

The repository focuses on implementing and understanding the fundamental components behind modern Large Language Models **from scratch**.

The broader learning path is:

```text
Tokenization
     ↓
Embeddings
     ↓
Attention
     ↓
Transformer
     ↓
Language Modeling
     ↓
LLM Architecture
```

The objective is not simply to use existing libraries.

Instead, the repository aims to understand:

- the algorithms
- the mathematics
- the data structures
- the training procedures
- the implementation details
- the design decisions behind modern LLMs

---

# Why Build BPE From Scratch?

Tokenization is one of the first stages of a language-model pipeline.

A simplified pipeline is:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding
   ↓
Transformer
   ↓
Output Probabilities
```

Implementing BPE from scratch makes it possible to understand exactly what happens between:

```text
Raw Text
```

and:

```text
Token IDs
```

before the Transformer processes the input.

Instead of treating tokenization as a black box, this implementation exposes the underlying process:

```text
Characters
    ↓
Pair Frequencies
    ↓
Most Frequent Pair
    ↓
Merge
    ↓
New Symbol
    ↓
Repeat
    ↓
Learned Vocabulary
```

---

# Status

**Implementation:** Complete — basic Word-Level BPE

**Algorithm:** BPE

**Initial Units:** Characters

**Word Boundary:** `</w>`

**Encoding:** Implemented

**Decoding:** Implemented

**Unknown Token:** `<unk>`

**Focus:** Algorithmic understanding and from-scratch implementation

**Production Ready:** No

**Next Steps:** Optimization, testing, WordPiece, Unigram, and Transformer integration

---

# Key Takeaway

The core idea of this implementation can be summarized as:

```text
                Raw Text
                   ↓
              Preprocessing
                   ↓
                 Words
                   ↓
          Characters + </w>
                   ↓
          Count frequent pairs
                   ↓
             Merge pair
                   ↓
            Create symbol
                   ↓
               Repeat
                   ↓
          Learned BPE vocabulary
                   ↓
             Encode text
                   ↓
              Token IDs
```

The most important concept is:

> **Word-Level BPE starts with characters inside pre-tokenized words and learns larger subword tokens by repeatedly merging the most frequent adjacent pairs.**

This implementation provides a transparent way to understand that process before moving on to more advanced tokenization approaches used in modern LLMs.