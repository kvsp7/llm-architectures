from word_level_bpe import tokenizer

text = input('Enter you text:')

ids = tokenizer.encode(text=text)
decoded = tokenizer.decode(ids)


print(f"Encoded ids : {ids}")
print(f"Decoded ids : {decoded}")