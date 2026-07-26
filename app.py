from __future__ import annotations
import joblib
import base64
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BrownLeaf Library",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / 'assets'
HERO_IMAGE_PATH = ASSETS_DIR / 'library_hero.png'
DEFAULT_DATA_FILES = (
    APP_DIR / "BooksDataset_cleaned_v2_filled_safe.csv",
    APP_DIR / "BooksDataset.csv",
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 500
OVERLAP = 50
PLACEHOLDER_TOKENS = {"nan", "", "n/a", "na", "none", "unknown", "by"}
REQUIRED_COLUMNS = {"Title", "Authors", "Description", "Category"}

REQUEST_HEADERS = {
    "User-Agent": "BrownLeafLibrary/1.0 (Streamlit book recommendation app)"
}




def image_file_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


HERO_IMAGE_URI = image_file_to_data_uri(HERO_IMAGE_PATH)

# -----------------------------------------------------------------------------
# Styling — premium futuristic library
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --midnight: #07111f;
        --navy: #10253b;
        --slate: #1d3557;
        --teal: #24b4c7;
        --aqua: #7cd8ff;
        --gold: #f0bb62;
        --cream: #f8fbff;
        --mist: #d8e7f5;
        --glass: rgba(11, 23, 40, 0.58);
        --panel: rgba(255,255,255,0.08);
        --text: #eaf4ff;
        --text-dark: #12304d;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 10% 10%, rgba(36,180,199,.15), transparent 26%),
            radial-gradient(circle at 90% 18%, rgba(240,187,98,.14), transparent 24%),
            linear-gradient(135deg, #06101d 0%, #0b1e31 42%, #163352 100%);
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: rgba(6,16,29,0.25);
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(5,10,20,.98) 0%, rgba(12,28,48,.97) 52%, rgba(21,51,82,.95) 100%);
        border-right: 1px solid rgba(124,216,255,.12);
    }}

    [data-testid="stSidebar"] * {{
        color: #eef7ff;
    }}

    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stTextInput input {{
        background: rgba(255,255,255,.07);
        border-color: rgba(124,216,255,.18);
        color: #f5fbff;
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        min-height: 370px;
        padding: 2.2rem 2.4rem;
        border-radius: 30px;
        background-image:
            linear-gradient(100deg, rgba(4, 10, 20, 0.88) 0%, rgba(8, 17, 31, 0.78) 40%, rgba(8, 17, 31, 0.52) 68%, rgba(8, 17, 31, 0.36) 100%),
            url('{HERO_IMAGE_URI}');
        background-size: cover;
        background-position: center;
        box-shadow: 0 22px 60px rgba(0,0,0,.28);
        border: 1px solid rgba(124,216,255,.12);
        color: white;
        margin-bottom: 1.25rem;
    }}

    .hero::before {{
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 75% 18%, rgba(36,180,199,.28), transparent 16%),
                    radial-gradient(circle at 83% 14%, rgba(240,187,98,.18), transparent 12%);
        pointer-events: none;
    }}

    .hero-content {{
        position: relative;
        z-index: 2;
        max-width: 760px;
    }}

    .hero-kicker {{
        display: inline-flex;
        gap: .45rem;
        align-items: center;
        padding: .4rem .85rem;
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.16);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.03);
        font-size: .8rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: #e7f4ff;
        backdrop-filter: blur(6px);
    }}

    .hero h1 {{
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.5rem, 5vw, 4.8rem);
        line-height: .95;
        margin: .9rem 0 .7rem;
        color: #ffffff;
        text-shadow: 0 8px 28px rgba(0,0,0,.24);
    }}

    .hero p {{
        max-width: 700px;
        color: #d5e9ff;
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }}

    .hero-accent {{
        display: flex;
        gap: 1rem;
        margin-top: 1.25rem;
        flex-wrap: wrap;
    }}

    .accent-chip {{
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14);
        padding: .65rem .9rem;
        border-radius: 18px;
        min-width: 150px;
        backdrop-filter: blur(8px);
    }}

    .accent-label {{
        font-size: .78rem;
        color: #a7d7f5;
        margin-bottom: .15rem;
    }}

    .accent-value {{
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
    }}

    .section-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #ffffff;
        margin: 1.25rem 0 .1rem;
    }}

    .section-subtitle {{
        color: #bfd3e7;
        margin-bottom: 1rem;
    }}

    .book-card {{
        height: 100%;
        min-height: 620px;
        border-radius: 24px;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(11,23,40,.95), rgba(16,37,59,.96));
        border: 1px solid rgba(124,216,255,.12);
        box-shadow: 0 16px 36px rgba(0,0,0,.18);
        transition: transform .2s ease, box-shadow .2s ease;
        margin-bottom: 1rem;
    }}

    .book-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 22px 46px rgba(0,0,0,.28);
    }}

    .cover-wrap {{
        position: relative;
        height: 310px;
        padding: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
            radial-gradient(circle at 20% 20%, rgba(36,180,199,.22), transparent 24%),
            radial-gradient(circle at 80% 12%, rgba(240,187,98,.18), transparent 18%),
            linear-gradient(145deg, #06101d, #0b1e31 58%, #173a5d);
    }}

    .cover-wrap img {{
        max-width: 190px;
        width: auto;
        height: 255px;
        object-fit: cover;
        border-radius: 8px 14px 14px 8px;
        box-shadow: -10px 14px 26px rgba(0,0,0,.38);
        background: #d9e7f5;
    }}

    .rank-badge {{
        position: absolute;
        top: 14px;
        left: 14px;
        min-width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #f0bb62, #ffd595);
        color: #0b1e31;
        border: 1px solid rgba(255,255,255,.5);
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(0,0,0,.22);
    }}

    .book-body {{
        padding: 1.2rem 1.2rem 1.35rem;
    }}

    .category-pill {{
        display: inline-block;
        max-width: 100%;
        padding: .32rem .7rem;
        border-radius: 999px;
        background: rgba(36,180,199,.12);
        color: #8be3ff;
        font-size: .73rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        border: 1px solid rgba(124,216,255,.15);
    }}

    .book-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.32rem;
        line-height: 1.18;
        color: #ffffff;
        margin: .8rem 0 .35rem;
        min-height: 3rem;
    }}

    .book-author {{
        color: #b9cee0;
        font-size: .88rem;
        margin-bottom: .85rem;
        min-height: 2.4rem;
    }}

    .book-description {{
        color: #d4e3ef;
        font-size: .89rem;
        line-height: 1.55;
        min-height: 7rem;
    }}

    .book-summary-label {{
        display: flex;
        align-items: center;
        gap: .4rem;
        margin: .9rem 0 .35rem;
        color: #8be3ff;
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .06em;
        text-transform: uppercase;
    }}

    .meta-row {{
        display: flex;
        justify-content: space-between;
        gap: .7rem;
        padding-top: .9rem;
        margin-top: .9rem;
        border-top: 1px solid rgba(124,216,255,.12);
        color: #b9cee0;
        font-size: .78rem;
    }}

    .score {{
        color: #ffd595;
        font-weight: 800;
    }}

    .empty-state {{
        text-align: center;
        padding: 3rem 1.25rem;
        border: 1px dashed rgba(124,216,255,.3);
        background: rgba(255,255,255,.05);
        border-radius: 24px;
        color: #d4e3ef;
        backdrop-filter: blur(10px);
    }}

    div[data-testid="stTextInput"] input {{
        border-radius: 16px;
        border: 1px solid rgba(124,216,255,.35);
        background: #f7fbff !important;
        min-height: 3.25rem;
        color: #10253b !important;
        -webkit-text-fill-color: #10253b !important;
        caret-color: #10253b !important;
        font-weight: 600;
    }}

    div[data-testid="stTextInput"] input::placeholder {{
        color: #71869a !important;
        -webkit-text-fill-color: #71869a !important;
        opacity: 1;
    }}

    div[data-testid="stTextInput"] input:focus {{
        border-color: #24b4c7 !important;
        box-shadow: 0 0 0 2px rgba(36,180,199,.18) !important;
    }}

    div[data-testid="stButton"] button {{
        min-height: 3.15rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(135deg, #18a6c1, #2b6fbe);
        color: white;
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(17, 66, 104, .25);
    }}

    div[data-testid="stButton"] button:hover {{
        background: linear-gradient(135deg, #27b9cf, #4d8adb);
        color: white;
        border: none;
    }}

    [data-testid="stMetric"] {{
        background: rgba(255,255,255,.06);
        padding: .9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(124,216,255,.12);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
    }}

    [data-testid="stMetric"] * {{
        color: #f2f9ff !important;
    }}

    .glass-panel {{
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(124,216,255,.12);
        border-radius: 24px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,.12);
        backdrop-filter: blur(8px);
    }}

    .footer-note {{
        color: #a8bfd4;
    }}

    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def clean_text_column(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.replace({"nan": np.nan, "": np.nan})


def clean_authors(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"^By\s+", "", regex=True).str.strip()
    return cleaned.mask(cleaned.str.lower().isin(PLACEHOLDER_TOKENS), np.nan)


def clean_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"\s{2,}", " > ", regex=True)
    return cleaned.replace({"nan": np.nan, "": np.nan})


def prepare_books(raw_df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

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

    for row in df.itertuples(index=False):
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
            continue

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

    return pd.DataFrame(rows)


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def read_source(source_bytes: Optional[bytes], source_path: str) -> pd.DataFrame:
    if source_bytes is not None:
        return pd.read_csv(io.BytesIO(source_bytes), engine="python", on_bad_lines="skip")
    return pd.read_csv(source_path, engine="python", on_bad_lines="skip")


@dataclass
class RetrievalEngine:
    books: pd.DataFrame
    chunks: pd.DataFrame
    bm25: BM25Okapi
    embedder: SentenceTransformer
    faiss_index: faiss.Index
    chunk_book_ids: np.ndarray

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        output = np.zeros_like(values, dtype=np.float32)
        if not finite.any():
            return output
        minimum = values[finite].min()
        maximum = values[finite].max()
        output[finite] = (values[finite] - minimum) / (maximum - minimum + 1e-9)
        return output

    def search(
        self,
        query: str,
        k: int = 12,
        alpha: float = 0.70,
        categories: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        query = query.strip()
        if not query:
            return self.books.head(k).assign(score=0.0, semantic_score=0.0, bm25_score=0.0)

        q_tokens = tokenize_bm25(query)
        bm25_chunk_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)

        q_vector = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")

                search_k = min(300, self.faiss_index.ntotal)

        semantic_sorted, semantic_indices = self.faiss_index.search(
            q_vector,
            search_k
        )

        semantic_chunk_scores = np.full(
            self.faiss_index.ntotal,
            -np.inf,
            dtype=np.float32
        )

        valid_indices = semantic_indices[0] >= 0

        semantic_chunk_scores[
            semantic_indices[0][valid_indices]
        ] = semantic_sorted[0][valid_indices]

        n_books = len(self.books)
        bm25_book = np.full(n_books, -np.inf, dtype=np.float32)
        semantic_book = np.full(n_books, -np.inf, dtype=np.float32)
        np.maximum.at(bm25_book, self.chunk_book_ids, bm25_chunk_scores)
        np.maximum.at(semantic_book, self.chunk_book_ids, semantic_chunk_scores)

        bm25_norm = self._minmax(bm25_book)
        semantic_norm = self._minmax(semantic_book)
        hybrid = alpha * semantic_norm + (1.0 - alpha) * bm25_norm

        if categories:
            category_mask = self.books["Category"].isin(categories).to_numpy()
            hybrid = np.where(category_mask, hybrid, -np.inf)

        valid_count = int(np.isfinite(hybrid).sum())
        if valid_count == 0:
            return pd.DataFrame(columns=list(self.books.columns) + ["score"])

        take = min(k, valid_count)
        if take == valid_count:
            top_ids = np.argsort(-hybrid)[:take]
        else:
            candidate_ids = np.argpartition(-hybrid, take - 1)[:take]
            top_ids = candidate_ids[np.argsort(-hybrid[candidate_ids])]

        results = self.books.iloc[top_ids].copy()
        results["score"] = hybrid[top_ids]
        results["semantic_score"] = semantic_norm[top_ids]
        results["bm25_score"] = bm25_norm[top_ids]
        return results.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_engine() -> RetrievalEngine:

   artifacts_dir = (
    APP_DIR
    / "artifacts"
    / "3fd4372126ba31dd23ba"
)

    books = pd.read_pickle(
        artifacts_dir / "books.pkl"
    )

    chunks = pd.read_pickle(
        artifacts_dir / "chunks.pkl"
    )

    bm25 = joblib.load(
        artifacts_dir / "bm25.joblib"
    )

    faiss_index = faiss.read_index(
        str(artifacts_dir / "faiss.index")
    )

    # إنشاء IDs من ملف chunks بدل الحاجة إلى ملف منفصل
    chunk_book_ids = (
        chunks["book_id"]
        .astype(np.int64)
        .to_numpy()
    )

    embedder = SentenceTransformer(
        EMBEDDING_MODEL
    )

    return RetrievalEngine(
        books=books,
        chunks=chunks,
        bm25=bm25,
        embedder=embedder,
        faiss_index=faiss_index,
        chunk_book_ids=chunk_book_ids,
    )

@st.cache_resource(show_spinner=False)
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank_results(query: str, results: pd.DataFrame, k: int) -> pd.DataFrame:
    if results.empty:
        return results

    reranker = load_reranker()
    pairs = [
        (
            query,
            f"Title: {row.Title} | Authors: {row.Authors} | "
            f"Category: {row.Category} | Description: {row.Description}",
        )
        for row in results.itertuples(index=False)
    ]
    scores = np.asarray(reranker.predict(pairs), dtype=np.float32)
    order = np.argsort(-scores)[:k]
    reranked = results.iloc[order].copy().reset_index(drop=True)
    reranked["reranker_score"] = scores[order]
    return reranked


# -----------------------------------------------------------------------------
# Book-cover retrieval
# -----------------------------------------------------------------------------
def _google_books_cover(title: str, author: str) -> tuple[Optional[str], Optional[str]]:
    query = f'intitle:"{title}" inauthor:"{author}"'
    response = requests.get(
        "https://www.googleapis.com/books/v1/volumes",
        params={"q": query, "maxResults": 1, "printType": "books"},
        headers=REQUEST_HEADERS,
        timeout=7,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    if not items:
        return None, None

    volume_info = items[0].get("volumeInfo", {})
    image_links = volume_info.get("imageLinks", {})
    cover = image_links.get("thumbnail") or image_links.get("smallThumbnail")
    info_link = volume_info.get("infoLink")
    if cover:
        cover = cover.replace("http://", "https://")
        cover = cover.replace("zoom=1", "zoom=2")
    return cover, info_link


def _open_library_cover(title: str, author: str) -> tuple[Optional[str], Optional[str]]:
    response = requests.get(
        "https://openlibrary.org/search.json",
        params={
            "title": title,
            "author": author,
            "limit": 1,
            "fields": "key,cover_i",
        },
        headers=REQUEST_HEADERS,
        timeout=7,
    )
    response.raise_for_status()
    docs = response.json().get("docs", [])
    if not docs:
        return None, None

    cover_id = docs[0].get("cover_i")
    work_key = docs[0].get("key")
    cover = (
        f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg?default=false"
        if cover_id
        else None
    )
    book_link = f"https://openlibrary.org{work_key}" if work_key else None
    return cover, book_link


def placeholder_cover(title: str) -> str:
    safe_title = html.escape(title[:42])
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='320' height='480'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='#3b2117'/>
          <stop offset='55%' stop-color='#78513a'/>
          <stop offset='100%' stop-color='#b8875f'/>
        </linearGradient>
      </defs>
      <rect width='320' height='480' rx='18' fill='url(#g)'/>
      <rect x='24' y='24' width='272' height='432' rx='12' fill='none' stroke='#efd9c2' stroke-width='2' opacity='.55'/>
      <text x='160' y='155' text-anchor='middle' font-size='62'>📖</text>
      <foreignObject x='42' y='215' width='236' height='165'>
        <div xmlns='http://www.w3.org/1999/xhtml' style='color:#fff8ef;font-family:Georgia,serif;font-size:25px;line-height:1.2;text-align:center;font-weight:bold;'>
          {safe_title}
        </div>
      </foreignObject>
      <text x='160' y='425' text-anchor='middle' fill='#ead7c4' font-family='Arial' font-size='15'>BrownLeaf Library</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_data(ttl=60 * 60 * 24 * 14, show_spinner=False, max_entries=3000)
def get_book_cover(title: str, author: str) -> tuple[str, Optional[str]]:
    for provider in (_google_books_cover, _open_library_cover):
        try:
            cover, book_link = provider(title, author)
            if cover:
                return cover, book_link
        except (requests.RequestException, ValueError, KeyError):
            continue
    return placeholder_cover(title), None


# -----------------------------------------------------------------------------
# Presentation helpers
# -----------------------------------------------------------------------------
def truncate(text: str, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def summarize_description(text: str, max_sentences: int = 3, max_chars: int = 420) -> str:
    """Create a concise extractive summary from the book description."""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned or cleaned.lower() in {"nan", "none"}:
        return "No summary is available for this book in the catalog."

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return truncate(cleaned, max_chars)

    selected: list[str] = []
    total_chars = 0
    for sentence in sentences:
        projected = total_chars + len(sentence)
        if selected and (len(selected) >= max_sentences or projected > max_chars):
            break
        selected.append(sentence)
        total_chars = projected

    return truncate(" ".join(selected), max_chars)


def display_price(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"nan", "none", "not listed", ""}:
        return "Not listed"
    return text


def render_book_card(row: pd.Series, rank: int) -> None:
    cover_url, book_link = get_book_cover(str(row["Title"]), str(row["Authors"]))
    title = html.escape(str(row["Title"]))
    author = html.escape(str(row["Authors"]))
    category = html.escape(str(row["Category"]))
    publisher = html.escape(str(row.get("Publisher", "Unknown")))
    summary = html.escape(summarize_description(str(row["Description"])))
    price = html.escape(display_price(row.get("Price", "Not listed")))
    score = float(row.get("score", 0.0))

    title_html = (
        f'<a href="{html.escape(book_link)}" target="_blank" '
        f'style="text-decoration:none;color:inherit">{title}</a>'
        if book_link
        else title
    )

    card = f"""
    <div class="book-card">
        <div class="cover-wrap">
            <div class="rank-badge">#{rank}</div>
            <img src="{html.escape(cover_url)}" alt="Cover of {title}" loading="lazy"
                 onerror="this.src='{placeholder_cover(str(row['Title']))}'" />
        </div>
        <div class="book-body">
            <span class="category-pill">{category}</span>
            <div class="book-title">{title_html}</div>
            <div class="book-author">by {author}</div>
            <div class="book-summary-label">✦ Book Summary</div>
            <div class="book-description">{summary}</div>
            <div class="meta-row">
                <span>🏛️ {publisher}</span>
                <span>💰 {price}</span>
            </div>
            <div class="meta-row">
                <span>AI relevance</span>
                <span class="score">{score * 100:.1f}%</span>
            </div>
        </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


def find_default_dataset() -> Optional[Path]:
    return next((path for path in DEFAULT_DATA_FILES if path.exists()), None)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
uploaded_file = None
with st.sidebar:
    st.markdown("## ✨ BrownLeaf Library")
    st.caption("Smart AI Book Discovery")
    st.markdown("<div class='glass-panel'><strong>Premium library experience</strong><br/><span style='color:#bfd3e7'>Elegant search, beautiful results, and smart retrieval powered by your book catalog.</span></div>", unsafe_allow_html=True)

    st.markdown("### Search settings")
    result_count = st.slider("Number of recommendations", 3, 18, 9, 3)
    semantic_weight = st.slider(
        "Semantic search weight",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Higher values focus more on meaning; lower values focus more on exact words.",
    )
    use_reranker = st.toggle(
        "Advanced Cross-Encoder reranking",
        value=False,
        help="Improves final ordering, but the first search can be slower while the model loads.",
    )

    st.markdown("---")
    st.caption(
        "Covers are fetched from Google Books, with Open Library as a fallback. "
        "Results come only from your local catalog."
    )


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <section class="hero">
        <div class="hero-content">
            <div class="hero-kicker">✦ Intelligent Library Discovery</div>
            <h1>Discover books in a<br/>smarter, richer way.</h1>
            <p>
                Explore your collection with a polished AI-powered interface inspired by a futuristic digital library.
                Search by topic, mood, author, genre, or a full natural-language request and get elegant results instantly.
            </p>
            <div class="hero-accent">
                <div class="accent-chip">
                    <div class="accent-label">Discovery Engine</div>
                    <div class="accent-value">Hybrid AI Search</div>
                </div>
                <div class="accent-chip">
                    <div class="accent-label">Experience</div>
                    <div class="accent-value">Modern & Visual</div>
                </div>
                <div class="accent-chip">
                    <div class="accent-label">Catalog</div>
                    <div class="accent-value">Your Books, Beautifully</div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

source_bytes: Optional[bytes] = uploaded_file.getvalue() if uploaded_file else None
default_dataset = find_default_dataset()
source_path = str(default_dataset) if default_dataset else ""

if source_bytes is None and not source_path:
    st.markdown(
        """
        <div class="empty-state">
            <h3>📂 Add your book catalog</h3>
            <p>Place <strong>BooksDataset.csv</strong> beside <code>app.py</code> to load your catalog automatically.</p>
            <p>Required columns: Title, Authors, Description, Category.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    with st.spinner("Preparing the library and building the AI search index…"):
        engine = build_engine()
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

with st.sidebar:
    categories = sorted(engine.books["Category"].dropna().astype(str).unique().tolist())
    selected_categories = st.multiselect(
        "Filter by category",
        options=categories,
        placeholder="All categories",
    )

metric_cols = st.columns(3)
metric_cols[0].metric("Books", f"{len(engine.books):,}")
metric_cols[1].metric("Categories", f"{engine.books['Category'].nunique():,}")
metric_cols[2].metric("Authors", f"{engine.books['Authors'].nunique():,}")

st.markdown('<div class="section-title">Explore the collection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Describe what you want to read — even in a complete sentence.</div>',
    unsafe_allow_html=True,
)

search_col, button_col = st.columns([6, 1])
with search_col:
    query = st.text_input(
        "Search",
        label_visibility="collapsed",
        placeholder="Example: a mysterious fantasy novel with magic, friendship, and adventure…",
    )
with button_col:
    search_clicked = st.button("Search books", use_container_width=True, type="primary")

quick_prompts = [
    "A mystery novel with a detective",
    "A romantic and emotional story",
    "A practical book about leadership",
    "A science book about space",
]
quick_cols = st.columns(4)
for index, prompt in enumerate(quick_prompts):
    if quick_cols[index].button(prompt, key=f"prompt_{index}", use_container_width=True):
        st.session_state["active_query"] = prompt

if search_clicked and query.strip():
    st.session_state["active_query"] = query.strip()

active_query = st.session_state.get("active_query", "")

if not active_query:
    featured = engine.books.sample(
        n=min(result_count, len(engine.books)),
        random_state=42,
    ).copy()
    featured["score"] = 0.0
    st.markdown('<div class="section-title">Featured books</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A warm selection from your catalog. Search above for AI-personalized matches.</div>',
        unsafe_allow_html=True,
    )
    results = featured
else:
    candidate_count = min(max(result_count * 5, 30), len(engine.books))
    with st.spinner("Searching the shelves…"):
        results = engine.search(
            active_query,
            k=candidate_count if use_reranker else result_count,
            alpha=semantic_weight,
            categories=selected_categories or None,
        )
        if use_reranker and not results.empty:
            results = rerank_results(active_query, results, result_count)
        else:
            results = results.head(result_count)

    st.markdown(
        f'<div class="section-title">Best matches for “{html.escape(active_query)}”</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="section-subtitle">Found {len(results)} recommendations from your own library catalog.</div>',
        unsafe_allow_html=True,
    )

if results.empty:
    st.markdown(
        """
        <div class="empty-state">
            <h3>No matching books found</h3>
            <p>Try broader wording or remove the category filter.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for row_start in range(0, len(results), 3):
        columns = st.columns(3, gap="large")
        row_slice = results.iloc[row_start : row_start + 3]
        for local_index, (_, result_row) in enumerate(row_slice.iterrows()):
            with columns[local_index]:
                render_book_card(result_row, row_start + local_index + 1)

st.markdown("---")
st.caption("BrownLeaf Library · Elegant AI search with BM25, Sentence Transformers, FAISS, and optional Cross-Encoder reranking")
