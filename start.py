import re
import TokernizerV1
import TokenizerV2
#read text file
with open ("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text=f.read()
#print("Total number of characters", len(raw_text))
#print(raw_text[:99])

#Basic tokenizer
preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed))

# Create a vocabulary

all_words = sorted(list(set(preprocessed)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_words)
#print(vocab_size)
vocab = {token:integer for integer, token in enumerate(all_words)}
# print(len(vocab.items()))
# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i>50:
#         break

#instantiate a new tokenizer object from the SimpleTokenizerV1 class
tokenizer = TokernizerV1.TokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn
said with pardonable pride."""
ids = tokenizer.encode(text)
words = tokenizer.decode(ids)
#print(words)

#instantiate a new tokenizer object from the SimpleTokenizerV2 class
tokenizer2 = TokenizerV2.TokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
#print(text)
print(tokenizer2.decode(tokenizer2.encode(text)))