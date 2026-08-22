"""Character-level tokenizer shared by all tasks."""
from typing import List


class Tokenizer:
    def __init__(self):
        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      '+', '-', '*', '=',
                      '[', ']', 'E', 'M', 'H',
                      '<PAD>', '<EOS>', '<BOS>']
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.pad_id = self.stoi['<PAD>']
        self.eos_id = self.stoi['<EOS>']
        self.bos_id = self.stoi['<BOS>']
        self.vocab_size = len(self.chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids)


TOKENIZER = Tokenizer()
