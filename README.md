# BrownLeaf Library — Streamlit Book Recommendation App

## Files
- `app.py` — the Streamlit interface and retrieval pipeline.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — brown theme and upload settings.

## Dataset
Place one of these beside `app.py`:
- `BooksDataset_cleaned_v2_filled_safe.csv`
- `BooksDataset.csv`

You can also upload the CSV from the app sidebar.

Required columns:
- `Title`
- `Authors`
- `Description`
- `Category`

Optional columns:
- `Publisher`
- `Price`
- `Publish Date`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the embedding model and builds the search index. The first search with advanced reranking enabled also downloads the Cross-Encoder model.

## Book covers
The app searches Google Books first and uses Open Library as a fallback. If no cover is found, it creates a brown custom placeholder cover.
