from byte_level_bpe import tokenizer

text = input('Enter you text:')

ids = tokenizer.encode(text=text)
decoded = tokenizer.decode(ids)


print(f"Encoded bytes : {ids}")
print(f"Decoded ids : {decoded}")