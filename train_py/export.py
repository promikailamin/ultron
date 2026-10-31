#!/usr/bin/env python3
"""Export a PyTorch Ultron checkpoint into the Java binary format.

Java stores every matrix as [in][out]; PyTorch nn.Linear stores [out][in], so
each linear weight is transposed here. The per-layer parameter order is:

  LN1 gain/bias, Wq/bq, Wk/bk, Wv/bv, Wo/bo, LN2 gain/bias, W1/b1, W2/b2

followed by the final LayerNorm gain/bias. The output head is tied to wte.

fp16 is the default (about 34 MB for 17M parameters). Use --fp32 only for
debugging or parity tests.
"""
import argparse
import os
import shutil
import struct

import numpy as np
import torch

from model import build_model

MAGIC_F32 = 0x554C5431
MAGIC_F16 = 0x554C5432


def flatten(t):
    return t.detach().cpu().contiguous().view(-1)


def export_tensors(model, fp16=True):
    out = []
    out.append(flatten(model.wte.weight))       # [V][D]
    out.append(flatten(model.wpe.weight))       # [T][D]
    for block in model.blocks:
        out.append(flatten(block.ln1.weight))
        out.append(flatten(block.ln1.bias))
        out.append(flatten(block.q.weight.t()))  # [in][out]
        out.append(flatten(block.q.bias))
        out.append(flatten(block.k.weight.t()))
        out.append(flatten(block.k.bias))
        out.append(flatten(block.v.weight.t()))
        out.append(flatten(block.v.bias))
        out.append(flatten(block.o.weight.t()))
        out.append(flatten(block.o.bias))
        out.append(flatten(block.ln2.weight))
        out.append(flatten(block.ln2.bias))
        out.append(flatten(block.w1.weight.t()))  # [D][F]
        out.append(flatten(block.w1.bias))
        out.append(flatten(block.w2.weight.t()))  # [F][D]
        out.append(flatten(block.w2.bias))
    out.append(flatten(model.ln_f.weight))
    out.append(flatten(model.ln_f.bias))
    return out


def write_model(model, path, fp16=True):
    tensors = export_tensors(model, fp16)
    with open(path, "wb") as f:
        f.write(struct.pack(">i", MAGIC_F16 if fp16 else MAGIC_F32))
        f.write(struct.pack(">6i", model.vocab, model.width, model.layers,
                            model.ffn, model.ctx, model.heads))
        for t in tensors:
            arr = t.numpy()
            if fp16:
                f.write(arr.astype(">f2").tobytes())
            else:
                f.write(arr.astype(">f4").tobytes())
    return sum(t.numel() for t in tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/ultron-py/best.pt")
    ap.add_argument("--out", default="out/ultron.bin")
    ap.add_argument("--vocab-out", default="data",
                    help="copy ultron-vocab.txt and ultron-merges.txt here")
    ap.add_argument("--tokenizer-dir", default="out/ultron-py")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = build_model(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_model(model, args.out, fp16=not args.fp32)
    print("exported %d parameters to %s (%s)" %
          (n, args.out, "fp32" if args.fp32 else "fp16"))
    print("size: %.2f MB" % (os.path.getsize(args.out) / 1e6))

    if args.vocab_out:
        os.makedirs(args.vocab_out, exist_ok=True)
        for name in ("ultron-vocab.txt", "ultron-merges.txt"):
            src = os.path.join(args.tokenizer_dir, name)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(args.vocab_out, name))
                print("copied %s -> %s" % (src, args.vocab_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
