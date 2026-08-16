# Byte-Level BPE

A **from-scratch implementation of Byte-Level Byte Pair Encoding (BPE)** in Python.

This implementation demonstrates how a tokenizer can start from the **256 possible byte values** and learn larger subword tokens by repeatedly merging the most frequent adjacent pair.

The purpose is to understand the internal mechanics of BPE rather than relying on an existing tokenizer library.

---

## Table of Contents

- [Overview](#overview)
- [How Byte-Level BPE Works](#how-byte-level-bpe-works)
- [Example](#example)
- [Training](#training)
- [Encoding](#encoding)
- [Decoding](#decoding)
- [Implementation](#implementation)
- [Architecture](#architecture)
- [Example Usage](#example-usage)
- [Testing](#testing)
- [Word-Level BPE vs Byte-Level BPE](#word-level-bpe-vs-byte-level-bpe)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Purpose](#purpose)

---

## Overview

Traditional character-based tokenization begins with characters:

```text
hello
 ↓
h e l l o
```

Byte-Level BPE instead begins with the UTF-8 representation of the text:

```text
hello
 ↓
UTF-8
 ↓
[104, 101, 108, 108, 111]
```

These byte values form the initial vocabulary.

The tokenizer then learns frequent byte/token combinations.

For example:

```text
(108, 111) → 256
```

represents:

```text
l + o → lo
```

A later merge can then combine the newly created token:

```text
(256, 119) → 257
```

representing:

```text
lo + w → low
```

The tokenizer therefore gradually builds larger and more useful subword tokens.

---

# How Byte-Level BPE Works

The complete training process is:

```text
                 Training Corpus
                       │
                       ▼
                UTF-8 Encoding
                       │
                       ▼
                 Byte Sequences
                       │
                       ▼
              Build Vocabulary
                       │
                       ▼
              Count Adjacent Pairs
                       │
                       ▼
             Select Most Frequent Pair
                       │
                       ▼
                  Merge Pair
                       │
                       ▼
                Assign New ID
                       │
                       ▼
              Repeat N Times
                       │
                       ▼
              Learned Merge Rules
```

### Initial Vocabulary

There are 256 possible byte values:

```text
0, 1, 2, ..., 255
```

Therefore:

```python
self.next_id = 256
```

Newly learned BPE tokens start at ID `256`.

For example:

```text
Byte 108 → 'l'
Byte 111 → 'o'

(108, 111) → 256
```

---

# Example

Consider the training corpus:

```text
low low lower
```

The text is converted to UTF-8 bytes.

For example:

```text
low
```

becomes:

```text
[108, 111, 119]
```

The tokenizer counts adjacent pairs:

```text
(108, 111)
(111, 119)
```

Suppose:

```text
(108, 111)
```

is the most frequent pair.

It is merged:

```text
(108, 111) → 256
```

Now:

```text
[108, 111, 119]
```

becomes:

```text
[256, 119]
```

If the next most frequent pair is:

```text
(256, 119)
```

another merge is learned:

```text
(256, 119) → 257
```

Now:

```text
[256, 119]
```

becomes:

```text
[257]
```

The tokenizer has therefore learned:

```text
l + o → 256
256 + w → 257
```

Conceptually:

```text
l o w
│ │ │
└─┘ → 256

256 w
───── → 257
```

This is the fundamental idea behind BPE.

---

# Training

The tokenizer performs the following operations.

## 1. Convert text to UTF-8 bytes

```python
line.encode("utf-8")
```

Example:

```text
"hello"
```

becomes:

```text
[104, 101, 108, 108, 111]
```

---

## 2. Build the vocabulary

The tokenizer stores each byte sequence together with its frequency.

Conceptually:

```text
Byte sequence                    Frequency

(104, 101, 108, 108, 111)           50
(104, 105)                          20
...
```

---

## 3. Count adjacent pairs

For:

```text
[104, 101, 108, 108, 111]
```

the pairs are:

```text
(104, 101)
(101, 108)
(108, 108)
(108, 111)
```

Their frequencies are accumulated across the corpus.

---

## 4. Select the most frequent pair

The pair with the highest frequency is selected:

```python
best_pair = max(pairs, key=pairs.get)
```

---

## 5. Assign a new token ID

The selected pair receives a new ID:

```text
256
257
258
...
```

For example:

```text
(104, 101) → 256
```

---

## 6. Merge the pair

Every occurrence of that pair is replaced with the new token ID.

The process repeats until:

- the requested number of merges is reached, or
- no pairs remain, or
- the most frequent pair occurs only once.

---

# Encoding

Once training is complete, new text can be encoded using the learned merge rules.

The process is:

```text
Input Text
    ↓
UTF-8 Encoding
    ↓
Initial Byte IDs
    ↓
Apply Merge #1
    ↓
Apply Merge #2
    ↓
Apply Merge #3
    ↓
...
    ↓
Final Token IDs
```

For example, if the learned rules are:

```text
(108, 111) → 256
(256, 119) → 257
```

then:

```text
"low"
```

starts as:

```text
[108, 111, 119]
```

After merge #1:

```text
[256, 119]
```

After merge #2:

```text
[257]
```

Therefore:

```text
"low" → [257]
```

---

# Decoding

Decoding performs the reverse process.

The tokenizer maintains:

```text
256 → (108, 111)
257 → (256, 119)
```

Given:

```text
[257]
```

the decoder expands it:

```text
257
 ↓
256, 119
 ↓
108, 111, 119
```

The resulting bytes are then decoded using UTF-8:

```python
bytes(ids).decode("utf-8")
```

giving:

```text
low
```

---

# Unicode Support

One major advantage of starting from bytes is that arbitrary Unicode text can be represented through UTF-8.

For example:

```text
Hello 世界 🚀
```

is converted into UTF-8 bytes before BPE is applied.

Therefore the tokenizer does not need an `<unk>` token simply because a character was not present in the training vocabulary.

The fundamental process is:

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

---

# Implementation

The implementation contains two classes.

## `Tokenizer`

Responsible for BPE training:

```text
Tokenizer
├── preprocessing()
├── generate_vocabulary()
├── generate_pairs()
├── merge_pair()
└── train_tokenizer()
```

### `preprocessing()`

Converts training text into UTF-8 byte sequences.

### `generate_vocabulary()`

Creates the initial vocabulary and stores sequence frequencies.

### `generate_pairs()`

Calculates frequencies of adjacent pairs.

### `merge_pair()`

Merges the selected pair and assigns its token ID.

### `train_tokenizer()`

Runs the complete BPE training procedure.

---

## `BPETokenizer`

Responsible for using the trained tokenizer:

```text
BPETokenizer
├── encode()
└── decode()
```

### `encode()`

Converts text into BPE token IDs.

### `decode()`

Expands learned tokens back into bytes and reconstructs the original UTF-8 text.

---

# Merge Representation

Learned merges are stored in:

```python
self.merged_pairs
```

Example:

```text
(108, 111) → 256
(256, 119) → 257
```

The order of learned merges is stored separately:

```python
self.merge_order
```

For example:

```text
1. (108, 111)
2. (256, 119)
3. ...
```

The merge order is important because BPE is a **sequential merging process**.

The same merge order must be applied during encoding.

---

# Example Usage

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

Expected result:

```text
Token IDs:
[...]

Decoded:
Implementration of Byte-Level-BPE
```

The fundamental correctness condition is:

```python
text == tokenizer.decode(tokenizer.encode(text))
```

---

# Testing

The tokenizer should be tested with different types of input.

### ASCII

```python
"Hello world!"
```

### Punctuation

```python
"Hello, world!"
```

### Numbers

```python
"Machine Learning 2026"
```

### Unicode

```python
"Hello 世界"
```

### Emoji

```python
"AI 🚀🤖"
```

### Mixed text

```python
"Hello 世界! AI 🚀"
```

For every valid UTF-8 input:

```python
decoded = tokenizer.decode(
    tokenizer.encode(text)
)

assert decoded == text
```

This verifies the fundamental encode/decode round trip.

---

# Word-Level BPE vs Byte-Level BPE

This repository contains multiple tokenizer implementations.

### Word-Level BPE

```text
Text
 ↓
Words
 ↓
Characters
 ↓
BPE
```

The initial representation depends on pre-tokenized words.

### Byte-Level BPE

```text
Text
 ↓
UTF-8 Bytes
 ↓
BPE
```

The initial representation consists of byte values.

This makes the byte-level approach capable of representing arbitrary Unicode text.

---

# Difference From Production LLM Tokenizers

This project implements the **core Byte-Level BPE algorithm** for educational and research purposes.

It should not be considered an exact reproduction of GPT-2, GPT-3, GPT-4, or another production tokenizer.

Production tokenizers may additionally use:

- specialized pre-tokenization
- regular-expression tokenization rules
- special tokens
- vocabulary files
- optimized merge lookup
- optimized encoding algorithms
- tokenizer serialization
- model-specific conventions

The implementation here intentionally focuses on understanding the fundamental BPE mechanism.

---

# Complexity

The current implementation repeatedly scans the vocabulary to:

1. Count pair frequencies.
2. Find the most frequent pair.
3. Apply the merge.
4. Repeat for every merge.

Therefore, it is primarily an **educational implementation** and is not optimized for very large corpora.

Production implementations use substantially more efficient data structures and algorithms for maintaining pair statistics.

---

# Limitations

Current limitations include:

- Basic training implementation
- No optimized pair-frequency updates
- Sequential merge application
- No special-token framework
- No vocabulary serialization
- No save/load functionality
- No optimized inference implementation
- No production-grade pre-tokenization

---

# Future Improvements

Planned improvements:

- [ ] Add tokenizer save/load functionality
- [ ] Add vocabulary serialization
- [ ] Add special tokens
- [ ] Add automated unit tests
- [ ] Add merge-rank optimization
- [ ] Benchmark training and encoding speed
- [ ] Measure compression/token efficiency
- [ ] Compare against existing BPE implementations
- [ ] Integrate with a Transformer implementation

---

# Repository Context

This implementation is part of the **`llm-architectures`** repository.

The repository focuses on implementing and understanding the fundamental components behind modern Large Language Models from scratch.

The tokenizer progression is:

```text
Word-Level BPE
      ↓
Byte-Level BPE
      ↓
WordPiece
      ↓
Unigram
      ↓
LLM Tokenizer Implementations
```

The objective is not simply to use existing libraries, but to understand **how the underlying algorithms work and why modern LLMs use them**.

---

## Status

**Implementation:** Complete — basic Byte-Level BPE

**Focus:** Algorithmic understanding and from-scratch implementation

**Production ready:** No

**Next step:** Optimization, testing, and integration with a Transformer