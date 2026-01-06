class SimpleTokenizer:
    def __init__(self, text):
        self.unk_token = "<UNK>"
        unique_chars = sorted(list(set(text)))
        self.chars = unique_chars + [self.unk_token]
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        self.unk_id = self.stoi[self.unk_token]

    def encode(self, s):
        return [self.stoi.get(c, self.unk_id) for c in s]

    def decode(self, idxs):
        return ''.join([self.itos[i] for i in idxs])