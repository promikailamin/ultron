#!/usr/bin/env python3
"""Train Ultron from scratch: Stage A pretraining, Stage B instruction tuning.

The default configuration is the 17M model from ULTRON_FIX_PLAN.md. It is
intended for a free Colab/Kaggle GPU; the same script also has a tiny --smoke
mode so CI can check that the pipeline runs on CPU.

Example (GPU):
    python3 train_py/train.py \
        --pretrain data/pretrain.txt \
        --qa data/qa_train.jsonl \
        --eval data/qa_eval.jsonl \
        --out out/ultron-py \
        --steps-a 20000 --epochs-b 4

Then export for the Java jar:
    python3 train_py/export.py --checkpoint out/ultron-py/best.pt \
        --out out/ultron.bin --vocab-out data
"""
import argparse
import json
import math
import os
import random
import time
from array import array

import torch
import torch.nn.functional as F

from bpe import BPETokenizer
from data import load_records, make_example
from model import UltronModel, build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", default="data/pretrain.txt")
    ap.add_argument("--qa", default="data/qa_train.jsonl")
    ap.add_argument("--eval", default="data/qa_eval.jsonl")
    ap.add_argument("--out", default="out/ultron-py")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps-a", type=int, default=20000)
    ap.add_argument("--epochs-b", type=float, default=4.0)
    ap.add_argument("--lr-a", type=float, default=6e-4)
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-pretrain-tokens", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sanity run used by CI")
    return ap.parse_args()


def get_tokenizer(args):
    vocab_path = os.path.join(args.out, "ultron-vocab.txt")
    merges_path = os.path.join(args.out, "ultron-merges.txt")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        tok = BPETokenizer.load(vocab_path, merges_path)
        print("loaded tokenizer: vocab=%d merges=%d" % (tok.V, len(tok.merges)))
        return tok
    print("training BPE tokenizer...")
    text = open(args.pretrain, encoding="utf-8").read()
    if args.max_pretrain_tokens:
        text = text[:args.max_pretrain_tokens * 4]
    tok = BPETokenizer.train(text, args.vocab_size)
    os.makedirs(args.out, exist_ok=True)
    tok.save(args.out)
    print("trained tokenizer: vocab=%d merges=%d" % (tok.V, len(tok.merges)))
    return tok


def encode_pretrain(tok, path, limit=0):
    text = open(path, encoding="utf-8").read()
    if limit:
        text = text[:limit * 4]
    ids = array("I", tok.encode(text))
    print("pretrain tokens: %d" % len(ids))
    return ids


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


def save_checkpoint(path, model, config, step, best_eval, stage):
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "step": step,
        "best_eval": best_eval,
        "stage": stage,
    }, path)


def pretrain_batch(data, ctx, batch, device):
    n = len(data)
    xs, ys = [], []
    for _ in range(batch):
        if n <= ctx + 1:
            start = 0
        else:
            start = random.randrange(0, n - ctx - 1)
        win = data[start:start + ctx + 1]
        if len(win) < ctx + 1:
            win = list(win) + [0] * (ctx + 1 - len(win))
        xs.append(list(win[:-1]))
        ys.append(list(win[1:]))
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device))


def collate(examples, device):
    max_s = max(len(ids) - 1 for ids, _ in examples)
    xs, ys, ms = [], [], []
    for ids, mask in examples:
        s = len(ids) - 1
        pad = max_s - s
        xs.append(ids[:-1] + [0] * pad)
        ys.append(ids[1:] + [0] * pad)
        ms.append(mask + [0] * pad)
    return (torch.tensor(xs, dtype=torch.long, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
            torch.tensor(ms, dtype=torch.float32, device=device))


def evaluate(model, examples, ctx, device, batch_size):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            x, y, m = collate(examples[i:i + batch_size], device)
            _, loss = model(x, y, m)
            total += float(loss) * x.shape[0]
            count += x.shape[0]
    model.train()
    return total / max(1, count)


def main():
    args = parse_args()
    if args.smoke:
        args.width, args.layers, args.heads, args.ffn, args.ctx = 64, 2, 2, 128, 64
        args.vocab_size = 512
        args.batch_size = 2
        args.steps_a = 2
        args.epochs_b = 0.05
        args.eval_every = 1
        args.max_pretrain_tokens = 20000
        args.patience = 100
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = device.type == "cuda"
    print("device=%s amp=%s" % (device, use_amp))

    tok = get_tokenizer(args)
    data = encode_pretrain(tok, args.pretrain, args.max_pretrain_tokens)
    records = load_records(args.qa)
    eval_records = load_records(args.eval)
    if args.smoke:
        records = records[:40]
        eval_records = eval_records[:20]
    train_examples = [ex for ex in (make_example(tok, r, args.ctx) for r in records) if ex]
    eval_examples = [ex for ex in (make_example(tok, r, args.ctx) for r in eval_records) if ex]
    print("instruction examples: train=%d eval=%d" % (len(train_examples), len(eval_examples)))

    config = {
        "vocab": tok.V,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "ffn": args.ffn,
        "ctx": args.ctx,
    }
    model = build_model(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print("parameters: %d" % params)
    print(json.dumps(config))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_eval = float("inf")
    bad_evals = 0
    global_step = 0
    t0 = time.time()

    # ---------------- Stage A: plain next-token pretraining ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_a, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(args.steps_a):
        lr = lr_at(step, args.steps_a, args.lr_a, args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = pretrain_batch(data, args.ctx, args.batch_size, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        global_step += 1
        if step % 20 == 0 or step == args.steps_a - 1:
            elapsed = time.time() - t0
            print("A step %d/%d loss %.4f ppl %.1f lr %.2e %.0f tok/s %.0fs" %
                  (step, args.steps_a, float(loss.detach()), math.exp(min(20, float(loss.detach()))),
                   lr, args.batch_size * args.ctx / max(elapsed, 1e-9), elapsed))
        if step > 0 and step % args.eval_every == 0:
            save_checkpoint(os.path.join(args.out, "stage_a.pt"), model, config,
                            global_step, best_eval, "a")
    save_checkpoint(os.path.join(args.out, "stage_a.pt"), model, config,
                    global_step, best_eval, "a")

    # ---------------- Stage B: answer-only instruction tuning ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_b, betas=(0.9, 0.95),
                            weight_decay=0.1)
    steps_per_epoch = max(1, len(train_examples) // args.batch_size)
    total_steps = max(1, int(steps_per_epoch * args.epochs_b))
    step_in_b = 0
    stop = False
    for epoch in range(max(1, int(math.ceil(args.epochs_b)))):
        if stop:
            break
        random.shuffle(train_examples)
        for i in range(0, len(train_examples), args.batch_size):
            if step_in_b >= total_steps:
                stop = True
                break
            lr = lr_at(step_in_b, total_steps, args.lr_b, min(args.warmup, total_steps // 10 + 1))
            for group in opt.param_groups:
                group["lr"] = lr
            batch = train_examples[i:i + args.batch_size]
            if len(batch) < 2 and len(train_examples) > 2:
                continue
            x, y, m = collate(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, loss = model(x, y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step_in_b += 1
            global_step += 1
            if step_in_b % 20 == 0:
                print("B step %d/%d epoch %d loss %.4f lr %.2e" %
                      (step_in_b, total_steps, epoch + 1, float(loss.detach()), lr))
            if step_in_b % args.eval_every == 0 or step_in_b == total_steps:
                ev = evaluate(model, eval_examples, args.ctx, device, args.batch_size)
                print("  eval loss %.4f ppl %.1f (best %.4f)" %
                      (ev, math.exp(min(20, ev)), best_eval))
                save_checkpoint(os.path.join(args.out, "latest.pt"), model, config,
                                global_step, best_eval, "b")
                if ev < best_eval - 1e-4:
                    best_eval = ev
                    bad_evals = 0
                    save_checkpoint(os.path.join(args.out, "best.pt"), model, config,
                                    global_step, best_eval, "b")
                else:
                    bad_evals += 1
                    if bad_evals >= args.patience:
                        print("early stop: eval loss stopped improving")
                        stop = True
                        break

    save_checkpoint(os.path.join(args.out, "final.pt"), model, config,
                    global_step, best_eval, "b")
    print("done in %.1fs; best eval loss %.4f" % (time.time() - t0, best_eval))
    print("checkpoints in %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
