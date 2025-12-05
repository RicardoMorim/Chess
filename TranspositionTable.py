import chess.polyglot

class TranspositionTable:
    def __init__(self, size=2**20):  # ~1 million entries
        self.size = size
        self.table = [None] * size

    def store(self, zobrist_hash, depth, flag, value, move):
        index = zobrist_hash % self.size
        self.table[index] = (zobrist_hash, depth, flag, value, move)

    def probe(self, zobrist_hash):
        index = zobrist_hash % self.size
        entry = self.table[index]
        if entry and entry[0] == zobrist_hash:
            return entry  # (hash, depth, flag, value, move)
        return None