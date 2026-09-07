"""Stage 2 — extract frozen image (ViT) and text (MiniLM) embeddings.

Row alignment
-------------
Embeddings are aligned by *position* to ``artifacts/prepared.{parquet,csv}``
(sorted by sample_id — see ``prepare_data.py``). Row ``i`` of ``emb_image.npy``
/ ``emb_text.npy`` corresponds to row ``i`` of the prepared table.

Chunked extraction (needed for long CPU-bound image inference)
----------------------------------------------------------------
    python src/extract_embeddings.py --modality image --start 0    --end 2000
    python src/extract_embeddings.py --modality image --start 2000 --end 4000
    ...
    python src/extract_embeddings.py --modality image --merge

Each ``--start/--end`` call writes a partial chunk
``artifacts/emb_image_{start}_{end}.npy``. ``--merge`` scans for all such
chunks, sorts them by start, verifies they contiguously cover
``[0, n_rows)`` with no gaps/overlaps, concatenates them in order, and writes
the final ``artifacts/emb_image.npy``.

Calling with neither ``--start`` nor ``--end`` processes the *entire* table
in one call and writes the final ``.npy`` directly (no chunking needed) --
this is what ``scripts/run_pipeline.*`` uses for text (fast) and can be used
for images too if the benchmark says it will comfortably finish.

Before committing to a full image run, use ``--benchmark-only`` to time
inference on a small sample and get an images/sec + ETA estimate::

    python src/extract_embeddings.py --modality image --benchmark-only --benchmark-n 64
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from utils import load_config, load_prepared_table, resolve_path, seed_everything

_CHUNK_RE = re.compile(r"emb_(image|text)_(\d+)_(\d+)\.npy$")


def _chunk_path(artifacts_dir: Path, modality: str, start: int, end: int) -> Path:
    return artifacts_dir / f"emb_{modality}_{start}_{end}.npy"


def _final_path(artifacts_dir: Path, modality: str) -> Path:
    return artifacts_dir / f"emb_{modality}.npy"


# ---------------------------------------------------------------------------
# Image embeddings (timm ViT, frozen, pooled 384-d output)
# ---------------------------------------------------------------------------

def _build_image_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class _ImageRowDataset:
    """Dataset over a list of image paths. Missing/corrupt -> zero tensor, ok=False."""

    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image

        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
            ok = True
        except Exception:
            tensor = torch.zeros(3, 224, 224)
            ok = False
        return tensor, ok, idx


def _load_image_model(model_name: str):
    import timm

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    return model


def extract_image_embeddings(
    cfg: dict,
    df_slice,
    benchmark_only: bool = False,
    benchmark_n: int = 64,
    error_log_path: Optional[Path] = None,
):
    """Run the frozen ViT over `df_slice` rows. Returns (array, n_bad, bad_local_indices)
    in full mode, or (rate_imgs_per_sec, elapsed_sec) in benchmark mode.
    """
    import torch
    from torch.utils.data import DataLoader

    images_dir = resolve_path(cfg, "images_dir")
    batch_size = cfg["image_batch_size"]
    num_workers = cfg.get("image_num_workers", 0)

    print(f"[extract_embeddings] loading image model: {cfg['image_model']}")
    t_load0 = time.time()
    model = _load_image_model(cfg["image_model"])
    print(f"[extract_embeddings] model loaded in {time.time() - t_load0:.1f}s "
          f"(num_features={model.num_features})")

    transform = _build_image_transform()
    image_paths = [images_dir / fn for fn in df_slice["image_filename"].tolist()]

    if benchmark_only:
        n = min(benchmark_n, len(image_paths))
        ds = _ImageRowDataset(image_paths[:n], transform)
        loader = DataLoader(ds, batch_size=min(batch_size, n), shuffle=False, num_workers=num_workers)
        t0 = time.time()
        n_done = 0
        with torch.no_grad():
            for tensors, oks, idxs in loader:
                feats = model(tensors)
                n_done += tensors.shape[0]
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else float("nan")
        print(f"[benchmark] image: {n_done} images in {elapsed:.2f}s -> {rate:.2f} imgs/sec "
              f"(embedding shape per batch: {tuple(feats.shape)})")
        subset_size = cfg["subset_size"]
        est_min = subset_size / rate / 60.0 if rate > 0 else float("inf")
        print(f"[benchmark] ETA for full subset_size={subset_size}: {est_min:.1f} minutes "
              f"(steady-state throughput only, excludes {time.time() - t0 - elapsed:.1f}s model load)")
        return rate, elapsed

    ds = _ImageRowDataset(image_paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dim = model.num_features
    out = np.zeros((len(image_paths), dim), dtype=np.float32)
    n_bad = 0
    bad_local_indices: list[int] = []
    t0 = time.time()
    with torch.no_grad():
        for tensors, oks, idxs in loader:
            feats = model(tensors).numpy().astype(np.float32)
            idx_np = idxs.numpy()
            out[idx_np] = feats
            oks_list = oks.tolist() if hasattr(oks, "tolist") else list(oks)
            for ok, idx in zip(oks_list, idx_np.tolist()):
                if not ok:
                    n_bad += 1
                    bad_local_indices.append(idx)
                    out[idx] = 0.0  # explicit zero-vector for missing/corrupt
    elapsed = time.time() - t0
    rate = len(image_paths) / elapsed if elapsed > 0 else float("nan")
    print(
        f"[extract_embeddings] image: {len(image_paths)} rows in {elapsed:.1f}s -> "
        f"{rate:.2f} imgs/sec, {n_bad} missing/corrupt -> zero-vector"
    )
    if bad_local_indices:
        sample_ids = df_slice["sample_id"].iloc[bad_local_indices].tolist()
        filenames = df_slice["image_filename"].iloc[bad_local_indices].tolist()
        msg = "\n".join(
            f"  local_idx={i} sample_id={sid} filename={fn}"
            for i, sid, fn in zip(bad_local_indices, sample_ids, filenames)
        )
        print(f"[extract_embeddings] WARNING: missing/corrupt images (zero-vector):\n{msg}")
        if error_log_path is not None:
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"# run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{msg}\n")
    return out, n_bad, bad_local_indices


# ---------------------------------------------------------------------------
# Text embeddings (MiniLM, mean-pooled, attention-mask weighted)
# ---------------------------------------------------------------------------

def _load_text_model(model_name: str):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_text_input(item_name, details) -> str:
    """`item_name + '. ' + details`, with a safe fallback for fully-empty rows."""
    item_name = (item_name or "").strip() if isinstance(item_name, str) else ""
    details = (details or "").strip() if isinstance(details, str) else ""
    if item_name and details:
        text = f"{item_name}. {details}"
    else:
        text = f"{item_name}{details}"
    text = text.strip(" .")
    return text if text else "unknown product"


def extract_text_embeddings(cfg: dict, df_slice, benchmark_only: bool = False, benchmark_n: int = 64):
    import torch

    print(f"[extract_embeddings] loading text model: {cfg['text_model']}")
    t_load0 = time.time()
    tokenizer, model = _load_text_model(cfg["text_model"])
    print(f"[extract_embeddings] model loaded in {time.time() - t_load0:.1f}s "
          f"(hidden_size={model.config.hidden_size})")

    texts = [build_text_input(r.item_name, r.details) for r in df_slice.itertuples()]
    batch_size = cfg["text_batch_size"]
    max_length = cfg["text_max_length"]

    if benchmark_only:
        texts = texts[: min(benchmark_n, len(texts))]

    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            outputs = model(**enc)
            pooled = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
            out[start : start + len(batch_texts)] = pooled.numpy().astype(np.float32)
    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else float("nan")
    print(f"[extract_embeddings] text: {len(texts)} rows in {elapsed:.1f}s -> {rate:.2f} texts/sec")

    if benchmark_only:
        subset_size = cfg["subset_size"]
        est_min = subset_size / rate / 60.0 if rate > 0 else float("inf")
        print(f"[benchmark] ETA for full subset_size={subset_size}: {est_min:.1f} minutes")
        return rate, elapsed
    return out, 0, []


# ---------------------------------------------------------------------------
# Chunk merging
# ---------------------------------------------------------------------------

def merge_chunks(cfg: dict, modality: str) -> Path:
    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    df = load_prepared_table(cfg)
    n_total = len(df)

    chunks = []
    for p in sorted(artifacts_dir.glob(f"emb_{modality}_*_*.npy")):
        m = _CHUNK_RE.match(p.name)
        if not m:
            continue
        start, end = int(m.group(2)), int(m.group(3))
        chunks.append((start, end, p))
    chunks.sort(key=lambda t: t[0])

    if not chunks:
        raise FileNotFoundError(f"No chunk files found for modality={modality} in {artifacts_dir}")

    cursor = 0
    arrays = []
    for start, end, p in chunks:
        if start != cursor:
            raise ValueError(
                f"Gap or overlap in chunks: expected next chunk to start at {cursor}, "
                f"but found {p.name} starting at {start}."
            )
        arr = np.load(p)
        if arr.shape[0] != (end - start):
            raise ValueError(f"Chunk {p.name} has {arr.shape[0]} rows, expected {end - start}.")
        arrays.append(arr)
        cursor = end

    if cursor != n_total:
        raise ValueError(
            f"Chunks cover rows [0, {cursor}) but the prepared table has {n_total} rows -- "
            f"merge is incomplete. Missing range: [{cursor}, {n_total})."
        )

    merged = np.concatenate(arrays, axis=0).astype(np.float32)
    final_path = _final_path(artifacts_dir, modality)
    np.save(final_path, merged)
    print(
        f"[extract_embeddings] merged {len(chunks)} chunk(s) covering [0,{n_total}) -> "
        f"{final_path} shape={merged.shape}"
    )
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract image or text embeddings.")
    parser.add_argument("--modality", choices=["image", "text"], required=True)
    parser.add_argument("--start", type=int, default=None, help="Row start (inclusive, 0-based position)")
    parser.add_argument("--end", type=int, default=None, help="Row end (exclusive)")
    parser.add_argument("--merge", action="store_true", help="Merge previously written chunks and exit")
    parser.add_argument("--benchmark-only", action="store_true", help="Time a small sample and exit; write nothing")
    parser.add_argument("--benchmark-n", type=int, default=64)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    import torch

    n_threads = os.cpu_count() or 1
    torch.set_num_threads(n_threads)
    print(f"[extract_embeddings] torch.set_num_threads({n_threads})")

    artifacts_dir = resolve_path(cfg, "artifacts_dir")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_chunks(cfg, args.modality)
        return

    df = load_prepared_table(cfg)
    n_total = len(df)

    is_full_range = args.start is None and args.end is None
    start = args.start if args.start is not None else 0
    end = min(args.end, n_total) if args.end is not None else n_total

    df_slice = df.iloc[start:end].reset_index(drop=True)
    print(f"[extract_embeddings] modality={args.modality} rows=[{start},{end}) of {n_total} (n={len(df_slice)})")

    if args.modality == "image":
        if args.benchmark_only:
            extract_image_embeddings(cfg, df_slice, benchmark_only=True, benchmark_n=args.benchmark_n)
            return
        error_log_path = artifacts_dir / "image_extract_errors.log"
        out, n_bad, _ = extract_image_embeddings(cfg, df_slice, error_log_path=error_log_path)
    else:
        if args.benchmark_only:
            extract_text_embeddings(cfg, df_slice, benchmark_only=True, benchmark_n=args.benchmark_n)
            return
        out, n_bad, _ = extract_text_embeddings(cfg, df_slice)

    save_path = _final_path(artifacts_dir, args.modality) if is_full_range else _chunk_path(
        artifacts_dir, args.modality, start, end
    )
    np.save(save_path, out)
    print(f"[extract_embeddings] saved array shape={out.shape} dtype={out.dtype} -> {save_path}")


if __name__ == "__main__":
    main()
