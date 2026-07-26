from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "BooksDataset.csv"
ARTIFACTS_ROOT = APP_DIR / "artifacts"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 150
OVERLAP = 25
EMBED_BATCH_SIZE = 64

PLACEHOLDER_TOKENS = {"nan", "", "n/a", "na", "none", "unknown", "by"}
REQUIRED_COLUMNS = {"Title", "Authors", "Description", "Category"}


def clean_text_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.mask(cleaned.str.lower().isin({"nan", "none", ""}), np.nan)


def clean_authors(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^By\s+", "", regex=True).str.strip()
    return cleaned.mask(cleaned.str.lower().isin(PLACEHOLDER_TOKENS), np.nan)


def clean_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\s{2,}", " > ", regex=True)
    return cleaned.mask(cleaned.str.lower().isin({"nan", "none", ""}), np.nan)


def prepare_books(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    df = raw_df.copy()
    df["Title"] = clean_text_column(df["Title"])
    df["Authors"] = clean_authors(df["Authors"])
    df["Description"] = clean_text_column(df["Description"])
    df["Category"] = clean_category(df["Category"])

    if "Publisher" not in df.columns:
        df["Publisher"] = "Unknown"
    else:
        df["Publisher"] = clean_text_column(df["Publisher"]).fillna("Unknown")

    if "Price" not in df.columns:
        df["Price"] = "Not listed"
    else:
        df["Price"] = df["Price"].fillna("Not listed").astype(str)

    if "Publish Date" not in df.columns:
        df["Publish Date"] = pd.NaT
    else:
        df["Publish Date"] = pd.to_datetime(df["Publish Date"], errors="coerce")

    df = df.drop_duplicates(subset=["Title", "Authors", "Description"])
    df = df.dropna(subset=["Title", "Authors", "Description", "Category"])
    df = df.reset_index(drop=True)
    df["book_id"] = np.arange(len(df), dtype=np.int64)

    df["retrieval_text"] = (
        "Title: " + df["Title"].astype(str)
        + " | Authors: " + df["Authors"].astype(str)
        + " | Category: " + df["Category"].astype(str)
        + " | Publisher: " + df["Publisher"].astype(str)
        + " | Description: " + df["Description"].astype(str)
    )
    return df


def chunk_books(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for position, row in enumerate(df.itertuples(index=False), start=1):
        header = f"Title: {row.Title} | Authors: {row.Authors} | Category: {row.Category}"
        words = str(row.Description).split()

        if len(words) <= CHUNK_SIZE:
            rows.append(
                {
                    "book_id": int(row.book_id),
                    "chunk_index": 0,
                    "chunk_text": str(row.Description),
                    "search_text": f"{header} | {row.Description}",
                }
            )
        else:
            start = 0
            chunk_index = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE, len(words))
                piece = " ".join(words[start:end])
                rows.append(
                    {
                        "book_id": int(row.book_id),
                        "chunk_index": chunk_index,
                        "chunk_text": piece,
                        "search_text": f"{header} | {piece}",
                    }
                )
                if end >= len(words):
                    break
                start += CHUNK_SIZE - OVERLAP
                chunk_index += 1

        if position % 10000 == 0:
            print(f"Chunking: {position:,}/{len(df):,} books")

    return pd.DataFrame(rows)


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(str(CHUNK_SIZE).encode("utf-8"))
    digest.update(str(OVERLAP).encode("utf-8"))

    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()[:20]


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"BooksDataset.csv was not found beside this script:\n{DATA_FILE}"
        )

    fingerprint = source_fingerprint(DATA_FILE)
    output_dir = ARTIFACTS_ROOT / fingerprint
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BrownLeaf RAG artifact builder")
    print(f"Dataset: {DATA_FILE}")
    print(f"Artifacts: {output_dir}")
    print("=" * 65)

    print("\n[1/5] Reading and cleaning the dataset...")
    raw_df = pd.read_csv(DATA_FILE, engine="python", on_bad_lines="skip")
    books = prepare_books(raw_df)
    del raw_df
    gc.collect()
    print(f"Clean books: {len(books):,}")

    print("\n[2/5] Creating chunks...")
    chunks = chunk_books(books)
    print(f"Chunks: {len(chunks):,}")

    print("\n[3/5] Building BM25...")
    corpus = chunks["search_text"].astype(str).tolist()
    tokenized_corpus = []
    for start in range(0, len(corpus), 10000):
        stop = min(start + 10000, len(corpus))
        tokenized_corpus.extend(tokenize_bm25(text) for text in corpus[start:stop])
        print(f"BM25 tokenization: {stop:,}/{len(corpus):,}")

    bm25 = BM25Okapi(tokenized_corpus)
    joblib.dump(bm25, output_dir / "bm25.joblib", compress=3)
    del tokenized_corpus, bm25
    gc.collect()
    print("BM25 saved.")

    print("\n[4/5] Building FAISS incrementally...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    index = None

    for start in range(0, len(corpus), EMBED_BATCH_SIZE):
        stop = min(start + EMBED_BATCH_SIZE, len(corpus))
        batch_embeddings = embedder.encode(
            corpus[start:stop],
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        batch_embeddings = np.asarray(batch_embeddings, dtype="float32")

        if index is None:
            index = faiss.IndexFlatIP(batch_embeddings.shape[1])

        index.add(batch_embeddings)
        del batch_embeddings

        if stop % 5000 < EMBED_BATCH_SIZE or stop == len(corpus):
            print(f"Embeddings: {stop:,}/{len(corpus):,}")

    if index is None:
        raise RuntimeError("No chunks were available to build the FAISS index.")

    faiss.write_index(index, str(output_dir / "faiss.index"))
    print(f"FAISS saved with {index.ntotal:,} vectors.")

    print("\n[5/5] Saving data tables and metadata...")
    books.to_pickle(output_dir / "books.pkl")
    chunks.to_pickle(output_dir / "chunks.pkl")

    metadata = {
        "fingerprint": fingerprint,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "books": int(len(books)),
        "chunks": int(len(chunks)),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n✅ Finished successfully.")
    print(f"Open this folder:\n{output_dir}")
    print("It should contain:")
    print("  books.pkl")
    print("  chunks.pkl")
    print("  bm25.joblib")
    print("  faiss.index")
    print("  metadata.json")


if __name__ == "__main__":
    main()
