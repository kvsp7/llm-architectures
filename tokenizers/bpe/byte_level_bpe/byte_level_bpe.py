class Tokenizer:
    def __init__(self):
        # Maps each learned byte pair to its newly assigned token ID.
        #
        # Example:
        # (108, 111) -> 256
        # (256, 119) -> 257
        self.merged_pairs = {}

        # The initial vocabulary consists of 256 possible byte values:
        # 0 through 255.
        # Learned BPE tokens therefore start at ID 256.
        self.next_id = 256

        # Stores the learned merge rules in the order they were created.
        # The order is important because the same sequence must be
        # applied during encoding.
        self.merge_order = []

    def preprocessing(self, corpus):
        # Convert each training sequence from text into UTF-8 bytes.
        #
        # Example:
        # "low" -> [108, 111, 119]
        #
        # Each sequence is stored as a tuple so that it can be used
        # as a dictionary key.
        return [tuple(line.encode("utf-8")) for line in corpus]

    def generate_vocabulary(self, byte_seqs):
        # Build the initial vocabulary from the byte sequences.
        #
        # The dictionary stores:
        #
        # byte sequence -> frequency
        #
        # Example:
        # (108, 111, 119) -> 10
        #
        # If the same sequence occurs multiple times, its frequency
        # is increased.
        vocabulary = {}

        for seq in byte_seqs:
            vocabulary[seq] = vocabulary.get(seq, 0) + 1

        return vocabulary

    def generate_pairs(self, vocabulary):
        # Count the frequency of every adjacent byte/token pair.
        #
        # Example:
        #
        # (108, 111, 119)
        #
        # produces:
        # (108, 111)
        # (111, 119)
        #
        # The frequency of the complete sequence is added to each pair.
        pairs = {}

        for byte_seq, freq in vocabulary.items():
            for i in range(len(byte_seq) - 1):
                pair = (byte_seq[i], byte_seq[i + 1])

                pairs[pair] = pairs.get(pair, 0) + freq

        return pairs

    def merge_pair(self, pair, vocabulary):
        # Merge every occurrence of the selected pair in the vocabulary.
        #
        # The selected pair receives a new token ID starting from 256.
        new_vocabulary = {}

        for byte_seq, freq in vocabulary.items():

            # Convert the tuple into a list so that elements can
            # be replaced during the merge operation.
            byte_seq = list(byte_seq)

            new_bytes = []
            i = 0

            while i < len(byte_seq):

                # Check whether the current two symbols match
                # the pair selected for merging.
                if (
                    i < len(byte_seq) - 1
                    and pair == (byte_seq[i], byte_seq[i + 1])
                ):

                    # Assign an ID to the pair the first time
                    # it is encountered.
                    #
                    # Example:
                    # (108, 111) -> 256
                    if pair not in self.merged_pairs:
                        self.merged_pairs[pair] = self.next_id
                        self.next_id += 1

                        # Preserve the order in which the merge
                        # was learned.
                        self.merge_order.append(pair)

                    merge_id = self.merged_pairs[pair]

                    # Replace the two symbols with their new token ID.
                    new_bytes.append(merge_id)

                    # Skip both symbols because they have been merged.
                    i += 2

                else:
                    # No merge at this position, so keep the
                    # current symbol unchanged.
                    new_bytes.append(byte_seq[i])
                    i += 1

            # Convert the merged sequence back to a tuple so it can
            # be used as a dictionary key.
            new_sequence = tuple(new_bytes)

            # Preserve the frequency of the original sequence.
            new_vocabulary[new_sequence] = (
                new_vocabulary.get(new_sequence, 0) + freq
            )

        return new_vocabulary

    def train_tokenizer(self, corpus, num_merges):
        # Convert the training corpus into UTF-8 byte sequences.
        byte_seqs = self.preprocessing(corpus=corpus)

        # Create the initial byte-level vocabulary.
        vocabulary = self.generate_vocabulary(
            byte_seqs=byte_seqs
        )

        # Repeatedly learn BPE merge rules.
        for _ in range(num_merges):

            # Count all adjacent pairs in the current vocabulary.
            pairs = self.generate_pairs(
                vocabulary=vocabulary
            )

            # Stop if there are no more pairs to merge.
            if not pairs:
                break

            # Select the most frequently occurring pair.
            best_pair = max(
                pairs,
                key=pairs.get
            )

            # Only learn pairs that occur at least twice.
            # This prevents learning a merge that occurs only once.
            if pairs[best_pair] < 2:
                break

            # Apply the selected merge to the vocabulary.
            vocabulary = self.merge_pair(
                pair=best_pair,
                vocabulary=vocabulary
            )

        return vocabulary


class BPETokenizer(Tokenizer):

    def __init__(self, corpus=None, merges=20000):
        super().__init__()

        # Train the BPE tokenizer on the provided corpus.
        #
        # This learns the merge rules and produces the final
        # trained vocabulary.
        self.trained_vocabulary = self.train_tokenizer(
            corpus=corpus,
            num_merges=merges
        )

        # Create the reverse mapping required for decoding.
        #
        # Example:
        # 256 -> (108, 111)
        # 257 -> (256, 119)
        self.id_to_pair = {
            merge_id: pair
            for pair, merge_id in self.merged_pairs.items()
        }

    def encode(self, text):
        # Convert the input text into UTF-8 bytes.
        #
        # Unlike Word-Level BPE, this allows arbitrary Unicode text
        # to be represented using bytes.
        byte_seq = list(text.encode("utf-8"))

        # Apply all learned merge rules in the same order in which
        # they were learned during training.
        for pair in self.merge_order:

            i = 0

            while i < len(byte_seq) - 1:

                # Check whether the current adjacent symbols
                # match the learned merge rule.
                if (
                    byte_seq[i],
                    byte_seq[i + 1]
                ) == pair:

                    # Replace the pair with its learned token ID.
                    merge_id = self.merged_pairs[pair]

                    byte_seq[i:i + 2] = [merge_id]

                    # Do not increment i here.
                    # The newly created token may participate in
                    # another merge later.
                else:
                    i += 1

        # The resulting sequence contains both original byte IDs
        # (0-255) and learned BPE token IDs (256+).
        return byte_seq

    def decode(self, ids):
        # Make a copy so that the original list of IDs is not modified.
        ids = ids.copy()

        changed = True

        # Continue expanding learned tokens until only original
        # byte values remain.
        while changed:
            changed = False
            i = 0

            while i < len(ids):

                # Check whether this ID represents a learned merge.
                if ids[i] in self.id_to_pair:

                    # Replace the merged token with the two symbols
                    # that originally created it.
                    replace_seq = list(
                        self.id_to_pair[ids[i]]
                    )

                    ids[i:i + 1] = replace_seq

                    changed = True

                else:
                    i += 1

            # Stop when there are no more learned tokens to expand.
            if not changed:
                break

        # At this point all IDs should be original byte values.
        # Convert them back into bytes and decode using UTF-8.
        return bytes(ids).decode("utf-8")


# Load the training corpus.
with open(
    r"tokenizers\The-verdict.txt",
    "r",
    encoding="utf-8"
) as file:
    # Each non-empty line is treated as a separate training sequence.
    lines = [
        line.rstrip("\n")
        for line in file
        if line.strip()
    ]


# Train the Byte-Level BPE tokenizer.
#
# The tokenizer starts with 256 byte values and learns up to
# 20,000 additional merge rules.
tokenizer = BPETokenizer(
    corpus=lines,
    merges=20000
)


'''
# Example text to encode.

text = "Implementation of Byte-Level-BPE"


# Convert the text into BPE token IDs.
ids = tokenizer.encode(text)

print("Token IDs:")
print(ids)


# Convert the token IDs back into the original text.
decoded = tokenizer.decode(ids)

print("Decoded text:")
print(decoded)
'''