"""Natural-language data: stream a domain corpus, tokenize, pack to memmap.

Per domain we produce, under <out>/<domain>/:
  train_tokens.bin / heldout_tokens.bin  (uint32 memmaps of token ids)
  meta.json                              (counts, tokenizer info, seq_len)

Domains (all difficulty-latent; no external labels anywhere):
  webtext : FineWeb-Edu (streaming sample of educational web text)
  code    : CodeSearchNet python (docstrings + python source)
  math    : MetaMathQA (chain-of-thought math solutions)

Downloads happen on the GPU instance (see pyproject `[nl]` extra); the Mac
never imports this package.
"""
import argparse
import json
import os

import numpy as np

SEQ_LEN = 2048
HELDOUT_FRACTION = 0.02


def _stream_docs(domain: str, max_docs: int):
    from datasets import load_dataset
    if domain == "webtext":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train",
                          streaming=True)
        for i, ex in enumerate(ds):
            if i >= max_docs:
                break
            t = ex.get("text") or ""
            if len(t) > 200:
                yield t
    elif domain == "code":
        ds = load_dataset("code_search_net", "python", split="train",
                          streaming=True)
        for i, ex in enumerate(ds):
            if i >= max_docs:
                break
            t = ((ex.get("func_documentation_string") or "") + "\n"
                 + (ex.get("func_code_string") or ""))
            if len(t) > 200:
                yield t
    elif domain == "math":
        ds = load_dataset("meta-math/MetaMathQA", split="train",
                          streaming=True)
        for i, ex in enumerate(ds):
            if i >= max_docs:
                break
            t = (ex.get("query") or "") + "\n" + (ex.get("response") or "")
            if len(t) > 50:
                yield t
    else:
        raise ValueError(f"unknown domain: {domain}")


def prepare(domain: str, tokenizer_name: str, out_root: str,
            max_tokens: int, max_docs: int = 2_000_000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out_dir = os.path.join(out_root, domain)
    os.makedirs(out_dir, exist_ok=True)

    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    doc_sep = tok.eos_token_id

    def tokenize_stream():
        buf = []
        n_tok = 0
        for doc in _stream_docs(domain, max_docs):
            ids = tok(doc, add_special_tokens=False)["input_ids"]
            buf.extend([bos] + ids + [doc_sep])
            n_tok += len(ids) + 2
            if n_tok >= max_tokens:
                break
            if len(buf) > 5_000_000:
                yield buf
                buf = []
        if buf:
            yield buf

    all_ids = np.empty(max_tokens + SEQ_LEN + 1_000_000, dtype=np.uint32)
    n = 0
    for chunk in tokenize_stream():
        c = np.asarray(chunk, dtype=np.uint32)
        take = min(len(c), len(all_ids) - n)
        all_ids[n:n + take] = c[:take]
        n += take
        if n >= max_tokens:
            break
    all_ids = all_ids[:n]

    n_held = int(len(all_ids) * HELDOUT_FRACTION)
    held, train = all_ids[:n_held], all_ids[n_held:]

    def pack(arr, path):
        n_seq = len(arr) // (SEQ_LEN + 1)
        arr = arr[:n_seq * (SEQ_LEN + 1)].reshape(n_seq, SEQ_LEN + 1)
        np.save(path, arr)
        return n_seq

    n_tr = pack(train, os.path.join(out_dir, "train_tokens.npy"))
    n_he = pack(held, os.path.join(out_dir, "heldout_tokens.npy"))
    meta = {"domain": domain, "tokenizer": tokenizer_name,
            "seq_len": SEQ_LEN, "train_seqs": n_tr, "heldout_seqs": n_he,
            "tokens": int(n)}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


def get_batches(out_root: str, domain: str, split: str, batch_seqs: int):
    """Yield (x, y) long tensors of (B, SEQ_LEN); x = seq[:-1], y = seq[1:]."""
    path = os.path.join(out_root, domain, f"{split}_tokens.npy")
    arr = np.load(path, mmap_mode="r")
    n = arr.shape[0]
    order = np.random.permutation(n) if split == "train" else np.arange(n)
    i = 0
    while True:
        idx = order[i:i + batch_seqs]
        if len(idx) < batch_seqs:
            if split != "train":
                break
            order = np.random.permutation(n)
            i = 0
            idx = order[i:i + batch_seqs]
        i += batch_seqs
        chunk = np.asarray(arr[idx])
        yield (torch_from(chunk[:, :-1]), torch_from(chunk[:, 1:]))


def torch_from(arr):
    import torch
    return torch.from_numpy(arr.astype(np.int64, copy=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True,
                   choices=["webtext", "code", "math"])
    p.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    p.add_argument("--out-root", default="data/nl")
    p.add_argument("--max-tokens", type=int, default=400_000_000)
    args = p.parse_args()
    prepare(args.domain, args.tokenizer, args.out_root, args.max_tokens)
