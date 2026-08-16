
# Word-Level BPE

A from-scratch implementation of **Byte Pair Encoding (BPE)** applied to pre-tokenized words.

This implementation is built to understand the core mechanics of BPE tokenization used in NLP and Large Language Models.

## Overview

BPE starts with individual characters and repeatedly merges the most frequent adjacent pair.

For example:

```text
l o w e r </w>
```

After learning merge rules:

```text
l o → lo
lo w → low
```

The representation can become:

```text
low e r </w>
```

The `</w>` token represents the end of a word.

## How It Works

```text
Raw Corpus
    ↓
Preprocessing
    ↓
Word Vocabulary
    ↓
Character Decomposition
    ↓
Pair Frequency Calculation
    ↓
Select Most Frequent Pair
    ↓
Merge Pair
    ↓
Repeat
    ↓
Learned Merge Rules
```

### 1. Preprocessing

The corpus is split into words and punctuation.

Example:

```text
"Hello, world!"
```

becomes approximately:

```text
["Hello", ",", "world", "!"]
```

### 2. Vocabulary Generation

Each word is converted into individual characters with an end-of-word marker.

```text
hello
```

becomes:

```text
h e l l o </w>
```

The frequency of each word is also stored.

### 3. Pair Generation

Adjacent symbol pairs are counted.

For:

```text
h e l l o </w>
```

the pairs are:

```text
(h, e)
(e, l)
(l, l)
(l, o)
(o, </w>)
```

### 4. Pair Selection

The most frequent pair is selected.

```python
best_pair = max(pairs, key=pairs.get)
```

### 5. Pair Merging

The selected pair is merged into a single symbol.

For example:

```text
l + o → lo
```

The process is repeated for the specified number of merges.

### 6. Encoding

When new text is provided, the tokenizer:

```text
Text
 ↓
Preprocessing
 ↓
Characters + </w>
 ↓
Apply learned merge rules
 ↓
Subword tokens
 ↓
Integer IDs
```

### 7. Decoding

The decoder converts integer IDs back into tokens and reconstructs the original text.

```text
Token IDs
 ↓
Tokens
 ↓
Concatenate
 ↓
Replace </w>
 ↓
Text
```

## Implementation

The tokenizer contains two main classes:

```text
Tokenizer
├── preprocessing()
├── generate_vocabulary()
├── generate_pairs()
├── merge_pair()
└── train_tokenizer()

BPETokenizer
├── encode()
└── decode()
```

## Example

```python
with open("The-verdict.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

tokenizer = BPETokenizer(
    data=corpus,
    merges=30000
)

text = "Hey, How are you..."

ids = tokenizer.encode(text)

print(ids)

decoded = tokenizer.decode(ids)

print(decoded)
```

## Implementation Details

| Component | Implementation |
|---|---|
| Algorithm | BPE |
| Tokenization | Word-Level |
| Initial symbols | Characters |
| Word boundary | `</w>` |
| Pair selection | Highest frequency |
| Merge strategy | Greedy |
| Unknown token | `<unk>` |
| Encoding | Tokens → Integer IDs |
| Decoding | Integer IDs → Tokens |

## Word-Level vs Byte-Level BPE

This implementation is **Word-Level BPE**.

It follows:

```text
Text
 ↓
Words
 ↓
Characters
 ↓
BPE
```

It is different from Byte-Level BPE:

```text
Text
 ↓
UTF-8 Bytes
 ↓
BPE
```

Byte-Level BPE is commonly used in modern LLM tokenization systems and can represent arbitrary text at the byte level.

This implementation intentionally uses characters within pre-tokenized words to provide a clear understanding of the fundamental BPE algorithm.

## Limitations

This implementation is primarily educational and is not optimized for production use.

Current limitations:

- Word-level preprocessing
- Character-based initial vocabulary
- Basic regular-expression preprocessing
- Sequential application of learned merge rules
- Basic `<unk>` handling
- No special-token system
- No byte-level encoding
- No optimized pair-ranking structure

## Future Work

Planned tokenizer implementations in this repository:

```text
Word-Level BPE
      ↓
Byte-Level BPE
      ↓
WordPiece
      ↓
Unigram
```

Future improvements include:

- Byte-Level BPE
- Special tokens
- Vocabulary serialization
- Efficient merge ranking
- Tokenization benchmarks
- Token efficiency analysis
- Comparison with existing LLM tokenizers

## Purpose

This implementation is part of the `llm-architectures` repository, which focuses on implementing and understanding the fundamental components of modern Large Language Models from scratch.

The goal is to understand the **algorithms, mathematics, and implementation details** behind these architectures rather than only using existing libraries.