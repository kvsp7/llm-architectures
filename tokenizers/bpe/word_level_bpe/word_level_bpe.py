import re


class Tokenizer:
    def __init__(self):
        # Stores the sequence of BPE merge rules learned during training.
        # Example: [('l', 'o'), ('lo', 'w'), ...]
        self.merged_pairs = []

    def preprocessing(self, corpus):
        # Split the corpus into words and punctuation.
        # Whitespace is also used as a delimiter.
        processed = re.split(r"([.,:;'_&!?()\"]|--|\s)", corpus)

        # Remove empty strings and surrounding whitespace.
        tokens = [token.strip() for token in processed if token.strip()]

        return tokens

    def generate_vocabulary(self, tokens):
        # Create the initial BPE vocabulary.
        #
        # Each word is represented as individual characters followed
        # by </w>, which marks the end of the word.
        #
        # Example:
        # "hello" -> "h e l l o </w>"
        vocabulary = {}

        for token in tokens:
            token = " ".join(list(token)) + " </w>"

            # Store the frequency of each word representation.
            vocabulary[token] = vocabulary.get(token, 0) + 1

        return vocabulary

    def generate_pairs(self, vocabulary):
        # Count the frequency of every adjacent symbol pair.
        #
        # Example:
        # "h e l l o </w>"
        #
        # Pairs:
        # ('h', 'e')
        # ('e', 'l')
        # ('l', 'l')
        # ('l', 'o')
        # ('o', '</w>')
        #
        # Word frequency is included when calculating pair frequency.
        pairs = {}

        for token, freq in vocabulary.items():
            token_list = token.split()

            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i + 1])

                pairs[pair] = pairs.get(pair, 0) + freq

        return pairs

    def merge_pair(self, pair, vocabulary):
        # Merge all occurrences of the selected pair in the vocabulary.
        new_vocabulary = {}

        for token, freq in vocabulary.items():
            token_list = token.split()
            new_tokens = []

            i = 0

            while i < len(token_list):

                # Check whether the current and next symbols
                # form the pair selected for merging.
                if (
                    i < len(token_list) - 1
                    and pair == (token_list[i], token_list[i + 1])
                ):
                    # Combine the two symbols into one new token.
                    new_merge = token_list[i] + token_list[i + 1]
                    new_tokens.append(new_merge)

                    # Skip both symbols because they have been merged.
                    i += 2

                else:
                    # Keep the current symbol unchanged.
                    new_tokens.append(token_list[i])
                    i += 1

            # Convert the merged token list back into a string.
            new_token = " ".join(new_tokens)

            # Combine frequencies if the resulting token already exists.
            new_vocabulary[new_token] = (
                new_vocabulary.get(new_token, 0) + freq
            )

        return new_vocabulary

    def train_tokenizer(self, corpus, num_merges):
        # Step 1: Preprocess the corpus into words/punctuation.
        tokens = self.preprocessing(corpus=corpus)

        # Step 2: Create the initial character-level vocabulary.
        vocabulary = self.generate_vocabulary(tokens=tokens)

        # Repeat the BPE learning process for the requested number
        # of merge operations.
        for _ in range(num_merges):

            # Count all adjacent symbol pairs.
            pairs = self.generate_pairs(vocabulary=vocabulary)

            # Stop if there are no more pairs to merge.
            if not pairs:
                break

            # Select the most frequent pair.
            best_pair = max(pairs, key=pairs.get)

            # Stop if the most frequent pair occurs only once.
            # This prevents learning insignificant merges.
            if pairs[best_pair] < 2:
                break

            # Apply the selected merge to the vocabulary.
            vocabulary = self.merge_pair(
                pair=best_pair,
                vocabulary=vocabulary
            )

            # Store the merge rule.
            #
            # The order is important because the same merge sequence
            # must later be applied during encoding.
            self.merged_pairs.append(best_pair)

        return vocabulary


class BPETokenizer(Tokenizer):

    def __init__(self, data, merges=10000):
        super().__init__()

        # Train the BPE tokenizer on the provided corpus.
        self.trained_vocabulary = self.train_tokenizer(
            corpus=data,
            num_merges=merges
        )

        # Extract all learned symbols from the trained vocabulary.
        self.vocabulary = set()

        for token in self.trained_vocabulary:
            for chars in token.split():
                self.vocabulary.add(chars)

        # Add an unknown token for symbols that are not in the vocabulary.
        self.vocabulary.add("<unk>")

        # Create token <-> integer mappings.
        self.str_to_int = {
            key: i for i, key in enumerate(sorted(self.vocabulary))
        }

        self.int_to_str = {
            value: key for key, value in self.str_to_int.items()
        }

    def encode(self, text):
        # Convert input text into token IDs.
        tokens = self.preprocessing(text)
        final_ids = []

        for token in tokens:

            # Start with individual characters and mark the
            # end of the word using </w>.
            #
            # Example:
            # "hello" -> ['h', 'e', 'l', 'l', 'o', '</w>']
            char_list = list(token) + ["</w>"]

            # Apply all learned BPE merges in the same order
            # in which they were learned during training.
            for pair in self.merged_pairs:

                i = 0

                while i < len(char_list) - 1:

                    # If the current adjacent symbols match
                    # the learned merge rule, combine them.
                    if pair == (char_list[i], char_list[i + 1]):
                        char_list[i:i + 2] = ["".join(pair)]

                    else:
                        i += 1

            # Convert the resulting subword tokens into integer IDs.
            for char in char_list:

                if char in self.str_to_int:
                    final_ids.append(self.str_to_int[char])

                else:
                    # Use <unk> if the symbol does not exist
                    # in the vocabulary.
                    final_ids.append(self.str_to_int["<unk>"])

        return final_ids

    def decode(self, ids):
        # Convert integer IDs back into tokens.
        tokens = [self.int_to_str[i] for i in ids]

        # Join all tokens together.
        text = "".join(tokens)

        # Replace the word-end marker with spaces.
        return text.replace("</w>", " ").strip()


# Load the training corpus.
with open(
    r"tokenizers\The-verdict.txt",
    "r",
    encoding="utf-8"
) as file:
    corpus = file.read()



# Train the Word-Level BPE tokenizer.
tokenizer = BPETokenizer(
    data=corpus,
    merges=30000
)

'''
# Example text used to test the tokenizer.
test = "Hey, How are you..."

# Encode the text into integer token IDs.
ids = tokenizer.encode(test)
print(ids)


# Decode the token IDs back into text.
text = tokenizer.decode(ids)
print(text)
'''