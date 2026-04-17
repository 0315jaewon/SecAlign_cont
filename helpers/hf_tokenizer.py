from transformers import AutoTokenizer

class HFLlamaTokenizer:
    def __init__(self, path: str, max_seq_len: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.max_seq_len = max_seq_len

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def encode(self, text: str):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def add_tokens(self, tokens):
        return self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokens)}
        )

    def token_to_id(self, token: str):
        return self.tokenizer.convert_tokens_to_ids(token)

    def id_to_token(self, token_id: int):
        return self.tokenizer.convert_ids_to_tokens(token_id)
