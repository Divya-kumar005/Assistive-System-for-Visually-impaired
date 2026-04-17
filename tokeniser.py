from collections import Counter

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def fit(self, captions):
        words = []
        for cap in captions:
            words.extend(cap.split())

        vocab = sorted(set(words))
        self.word2idx = {w: i+1 for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def texts_to_sequences(self, captions):
        sequences = []
        for cap in captions:
            seq = [self.word2idx.get(word, 0) for word in cap.split()]
            sequences.append(seq)
        return sequences