#!/usr/bin/env python3
"""Small whitespace-pre-split BPE tokenizer.

The merge file is a plain list of ``left<TAB>right`` lines. Replaying it in
Java is straightforward because every merged symbol is just the concatenation
of its two parents. A leading ``▁`` marks a word that follows whitespace, and
``<p>`` marks a paragraph break, so decoding does not need extra metadata.

Usage:
    python3 train_py/bpe.py --corpus data/pretrain.txt --vocab-size 8192 \
        --out out/ultron-py
"""
import argparse
import collections
import functools
import heapq
import os
import re
import sys

UNK, EOS, PAR = "<unk>", "<eos>", "<p>"
SPACE = "\u2581"


def normalize(text):
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # The pretrain corpus concatenates several books, so strip every
    # Gutenberg wrapper. Truncating at the first END marker would silently
    # train on only the first book.
    text = re.sub(r"(?im)^\*\*\* start of the project gutenberg.*\n?", "", text)
    text = re.sub(r"(?im)^\*\*\* end of the project gutenberg.*\n?", "", text)
    text = re.sub(r"\[illustration[^\]]*\]", " ", text)
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    return text


def is_word_char(c):
    return ("a" <= c <= "z") or ("0" <= c <= "9") or c == "'"


def pretokenize(text):
    """Return pieces; each word piece carries a leading space marker."""
    text = normalize(text)
    out = []
    i, n = 0, len(text)
    pending_space = True
    while i < n:
        c = text[i]
        if c in " \t\n":
            j = i
            newlines = 0
            while j < n and text[j] in " \t\n":
                if text[j] == "\n":
                    newlines += 1
                j += 1
            if newlines >= 2:
                out.append(PAR)
            pending_space = True
            i = j
        elif is_word_char(c):
            j = i
            while j < n and is_word_char(text[j]):
                j += 1
            w = text[i:j]
            while len(w) > 1 and w.endswith("'"):
                w = w[:-1]
            while len(w) > 1 and w.startswith("'"):
                w = w[1:]
            if w:
                out.append((SPACE if pending_space else "") + w)
            pending_space = False
            i = j
        else:
            out.append((SPACE if pending_space else "") + c)
            pending_space = False
            i += 1
    return out


class BPETokenizer:
    def __init__(self, itos, merges):
        self.itos = list(itos)
        self.V = len(self.itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.merges = list(merges)
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._encode_piece = functools.lru_cache(maxsize=200000)(self._encode_piece_uncached)

    @staticmethod
    def train(corpus_text, vocab_size=8192, max_pieces=None):
        pieces = pretokenize(corpus_text)
        if max_pieces:
            pieces = pieces[:max_pieces]
        counts = collections.Counter(p for p in pieces if p != PAR)
        words = [tuple(p) for p in counts]
        freqs = [counts[p] for p in counts]
        chars = sorted({c for w in words for c in w})
        merges = []

        pair_counts = collections.Counter()
        pair_words = collections.defaultdict(set)
        for wi, w in enumerate(words):
            f = freqs[wi]
            for a, b in zip(w, w[1:]):
                pair_counts[(a, b)] += f
                pair_words[(a, b)].add(wi)

        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)
        while len(merges) < vocab_size - 3 - len(chars) and heap:
            negc, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -negc or -negc < 2:
                continue
            left, right = pair
            merged = left + right
            merges.append(pair)
            affected = list(pair_words.get(pair, ()))
            for wi in affected:
                old = words[wi]
                if not any(old[k] == left and old[k + 1] == right
                           for k in range(len(old) - 1)):
                    continue
                f = freqs[wi]
                new = []
                k = 0
                while k < len(old):
                    if k + 1 < len(old) and old[k] == left and old[k + 1] == right:
                        new.append(merged)
                        k += 2
                    else:
                        new.append(old[k])
                        k += 1
                new = tuple(new)
                # Remove this word from all old pair buckets.
                for a, b in zip(old, old[1:]):
                    pair_counts[(a, b)] -= f
                    pair_words[(a, b)].discard(wi)
                    heapq.heappush(heap, (-pair_counts[(a, b)], (a, b)))
                words[wi] = new
                for a, b in zip(new, new[1:]):
                    pair_counts[(a, b)] += f
                    pair_words[(a, b)].add(wi)
                    heapq.heappush(heap, (-pair_counts[(a, b)], (a, b)))
            pair_words.pop(pair, None)

        vocab = [UNK, EOS, PAR] + chars
        seen = set(vocab)
        for left, right in merges:
            tok = left + right
            if tok not in seen:
                vocab.append(tok)
                seen.add(tok)
            if len(vocab) >= vocab_size:
                break
        return BPETokenizer(vocab, merges[:max(0, len(vocab) - 3 - len(chars))])

    def _encode_piece_uncached(self, piece):
        if piece == PAR:
            return (2,)
        syms = list(piece)
        while len(syms) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(syms) - 1):
                rank = self.merge_ranks.get((syms[i], syms[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            left, right = syms[best_idx], syms[best_idx + 1]
            merged = left + right
            out = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == left and syms[i + 1] == right:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        ids = []
        for s in syms:
            idx = self.stoi.get(s)
            if idx is not None:
                ids.append(idx)
            else:
                for c in s:
                    ids.append(self.stoi.get(c, 0))
        return tuple(ids)

    def encode(self, text):
        out = []
        for piece in pretokenize(text):
            out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids):
        parts = []
        for i in ids:
            if i < 0 or i >= self.V:
                continue
            tok = self.itos[i]
            if tok == EOS or tok == UNK:
                continue
            if tok == PAR:
                parts.append("\n\n")
            else:
                parts.append(tok.replace(SPACE, " "))
        return "".join(parts).lstrip()

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        vocab_path = os.path.join(out_dir, "ultron-vocab.txt")
        merges_path = os.path.join(out_dir, "ultron-merges.txt")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for i, tok in enumerate(self.itos):
                f.write("%d\t%s\n" % (i, tok))
        with open(merges_path, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write("%s\t%s\n" % (left, right))
        return vocab_path, merges_path

    @classmethod
    def load(cls, vocab_path, merges_path):
        itos = []
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx, tok = line.split("\t", 1)
                idx = int(idx)
                while len(itos) <= idx:
                    itos.append(UNK)
                itos[idx] = tok
        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                left, right = line.split("\t", 1)
                merges.append((left, right))
        return cls(itos, merges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/pretrain.txt")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-pieces", type=int, default=0)
    ap.add_argument("--out", default="out/ultron-py")
    args = ap.parse_args()
    text = open(args.corpus, encoding="utf-8").read()
    tok = BPETokenizer.train(text, args.vocab_size, args.max_pieces or None)
    vocab_path, merges_path = tok.save(args.out)
    sample = "User: what is obsidian?\nUltron: Obsidian is a hard block."
    ids = tok.encode(sample)
    print("vocab=%d merges=%d" % (tok.V, len(tok.merges)))
    print("sample ids=%s" % ids[:30])
    print("roundtrip=%r" % tok.decode(ids))
    print("wrote %s and %s" % (vocab_path, merges_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
