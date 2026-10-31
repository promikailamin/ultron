# Train the 17M Ultron model on a free GPU

The coding agent cannot train the 17M model on the CPU runner. Run these cells
on Colab or Kaggle, then commit the three output files back to the repository.

```bash
git clone https://github.com/promikailamin/ultron.git
cd ultron
pip install -r train_py/requirements.txt
python3 .agent-work/tools/gen_qa.py
python3 train_py/bpe.py --corpus data/pretrain.txt --vocab-size 8192 --out out/ultron-py
python3 train_py/train.py --pretrain data/pretrain.txt --qa data/qa_train.jsonl \
    --eval data/qa_eval.jsonl --out out/ultron-py --steps-a 20000 --epochs-b 4
python3 train_py/export.py --checkpoint out/ultron-py/best.pt --out out/ultron.bin --vocab-out data
python3 train_py/parity_test.py --checkpoint out/ultron-py/best.pt --tokenizer-dir out/ultron-py
```

If the GPU session cannot run the whole job in one cell, run the steps one at a
time. The tokenizer and checkpoints are written under `out/ultron-py/`, so a
later cell can resume from them.

## What to send back

Download and commit these files:

- `out/ultron.bin`
- `data/ultron-vocab.txt`
- `data/ultron-merges.txt`

Keep the training log too. Before the model is trusted, Stage B eval loss must
go down and then flatten, and the parity test must report a maximum logit
difference below `1e-3`.

If training is too slow or eval loss stops improving, report the log and the
last checkpoint instead of hiding a bad run. Useful fallbacks are more Stage A
steps, more data, or a smaller width/layer count; do not loosen the eval
thresholds.

## After the trained files are committed

```bash
bash eval/run_eval.sh
```

The eval now measures fact checks, exact-copy rate, unusable replies, in-domain
abstains and out-of-domain abstains. Until the 17M checkpoint replaces the
legacy one, the grounded composition path keeps chat working, but it is not a
substitute for a properly trained transformer.
