from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-mid", use_fast=True)
print(tokenizer("2003年夏季", return_offsets_mapping=True)["offset_mapping"])
print(tokenizer.tokenize("2003年夏季"))