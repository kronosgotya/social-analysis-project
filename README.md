# SOCIAL ANALYTICS PROJECT · TELEGRAM + X · SENTIMENT · EMOTIONS · NETWORK

Multichannel social analysis using **Python + Hugging Face** for *sentiment analysis*, *emotion classification*, 
and *network analysis* (X/Twitter). Exports **Tableau‑ready** datasets and a **.gexf** graph for **Gephi**.  
Compatible with macOS (Apple Silicon).  
Tested with Python 3.11 and ARM-friendly requirements.

---

## 1) Objective and Scope
**Analytical Objective:** Detect and monitor relevant actors, influential channels, and emotional discourse peaks around keywords or topics.

**Technical Scope:**
- Ingests **Telegram** (CSV) and **X/Twitter** (CSV) data exported externally.
- Unified preprocessing in `src/preprocessing.py`.
- **Sentiment** model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
- **Emotions** model: zero‑shot XNLI (`joeddav/xlm-roberta-large-xnli`).
- **X network** (mentions and replies; exports `.gexf` graph and CSV metrics).
- Final export to `data/processed/` ready for **Tableau**.

> **Note:** The **network** is built **only from X data**; Telegram is used for content and emotion.

---

## 2) Repository Structure
```text
social-analytics-project/
├── data/
│   ├── raw/                        # Original CSVs (telegram.csv, x.csv)
│   └── processed/                  # Clean, normalized (Tableau-ready)
│       ├─ all_platforms.csv
│       ├─ facts_posts.csv
│       ├─ emotions_long.csv
│       ├─ telegram_sentiment.csv
│       ├─ x_sentiment.csv
│       └─ x_edges.csv              # Edges for Tableau/Gephi (X)
├── results/
│   ├── graphs/
│   │   ├─ x_nodes_metrics.csv      # Influence metrics (degree/betweenness/pagerank…)
│   │   └─ x_interactions.gexf      # Graph for Gephi
│   └── charts/                     # (optional) exported images
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── emotions.py
│   ├── network.py
├── scripts/
│   └── process_all.py
├── requirements.txt
└── README.txt  (this document)
```

---

## 3) Requirements and Installation
- **Python** 3.10–3.11 (recommended)
- `pip`/`venv` or `conda`
- *(Optional)* PyTorch with CUDA for GPU acceleration

```bash
# create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## 4) Input Data
### 4.1 Telegram (`data/raw/telegram.csv`)
**Required fields:** `timestamp, channel, messageId, uid, kind, summary, link, lang`  
The pipeline renames `summary`→`text` and `channel`→`author`.

### 4.2 X/Twitter (`data/raw/x.csv`)
**Required fields:**  
`Tweet ID, URL, Content, Language, Likes, Retweets, Replies, Quotes, Views, Date, Is Reply, Conversation ID, Reply to userid, Author ID, Author Name, Author Profile URL, Hashtags, Hashtags counts, MentionedUsers, Quoted X ID, Quoted Tweet URL, Retweeted X ID, Retweeted X URL`

> Handles slight variations/typos (e.g., `Reply told` → `Reply to userid`).

---

## 5) Pipeline Execution
```bash
python scripts/process_all.py --telegram data/raw/telegram.csv --x data/raw/x.csv --device -1   # CPU (-1) or GPU id (0)
```
**Parameters:**
- `--telegram`: path to Telegram CSV (optional)
- `--x`: path to X CSV (optional)
- `--max_rows`: limits rows for testing (`0` = all)
- `--device`: `-1` for CPU, `0` for first GPU, etc.
- `--emotion_model`: name of zero‑shot model for emotions

**Expected console output:**
```
✔ TG processed → data/processed/telegram_sentiment.csv
✔ X processed → data/processed/x_sentiment.csv
✔ X network → results/graphs/x_interactions.gexf, results/graphs/x_nodes_metrics.csv, data/processed/x_edges.csv
✔ facts_posts.csv, emotions_long.csv, all_platforms.csv generated in data/processed/
```

---

## 6) Modules (Technical Overview)
### `src/preprocessing.py`
- `load_telegram(path)` → normalize and return unified fields.
- `load_x(path)` → normalize X export.
- `unify_frames(df_tg, df_x)` → merge both sources for Tableau.
- Helpers: `normalize_whitespace`, `extract_emojis`, list parsing.

### `src/sentiment.py`
- Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual).
- `add_sentiment` adds: `sentiment_label`, `sentiment_score`, `sentiment_dist` (dict).
- Configurable batch size (default `64`).

### `src/emotions.py`
- Zero-shot: `joeddav/xlm-roberta-large-xnli` (or lightweight `MoritzLaurer/mMiniLMv2-L6-mnli-xnli`).
- `add_emotions` adds: `emotion_label`, `emotion_scores` (dict of 10 emotions).
- Handles empty texts gracefully.

### `src/network.py`
- **Network built only from X data**: mentions and replies, adds `weight` per directed pair.
- `build_x_graph(df_x)` → `nx.DiGraph` with nodes and edges.
- `graph_metrics(G)` → compute `degree, in_degree, out_degree, betweenness, eigenvector, pagerank, label`.
- `export_gexf(G, path)` → save for Gephi.
- `edges_from_x(df_x)` → `src_user, dst_user, edge_kind, weight, first_ts, last_ts`.

### `src/utils.py`
- `normalize_source(df)` → ensure `X`/`Telegram` consistency.
- `add_engagement(df)` → sum of `likes+retweets+replies+quotes` (0 for TG).
- `add_dominant_emotion(df)` → infer `emotion_label` from `emotion_scores`.
- `emotions_to_long(df)` → expand `emotion_scores` to **long format**.
- `export_tableau_csv(df, path)` → robust export (UTF‑8‑BOM; `sep=';'`).

### `scripts/process_all.py`
- Orchestrates **load → sentiment → emotions → X network → unified → export**.
- Flags: `--device`, `--max_rows`, `--emotion_model`.

---

## 7) Outputs (Data Dictionary)
- **`data/processed/facts_posts.csv`** — main Tableau dataset (*Overview/Influencers*).  
  Fields: `source, timestamp, author, item_id, lang, sentiment_label, sentiment_score, emotion_label, emoji_count, text_clean, link, likes, retweets, replies, quotes, engagement`.

- **`data/processed/emotions_long.csv`** — for heatmaps and radial plots (*Emotions & Narratives*).  
  Schema: `item_id, source, emotion, prob, sentiment_label, lang, author, timestamp`.

- **`data/processed/x_edges.csv`** — for Sankey diagrams or direct import to Gephi.  
  Schema: `src_user, dst_user, edge_kind, weight, first_ts, last_ts`.

- **`results/graphs/x_nodes_metrics.csv`** — influence metrics joined with sentiment/emotion.  
  Fields: `node, degree, in_degree, out_degree, betweenness, eigenvector, pagerank, label`.

- **`results/graphs/x_interactions.gexf`** — open in **Gephi** for layout and communities.

---

## 8) Gephi (Minimum Steps)
1. Open `results/graphs/x_interactions.gexf` → *Directed* ✓  
2. **Layout:** *ForceAtlas2* (Scaling 10–50; Gravity 1–5; Prevent overlap ✓)  
3. **Appearance:** node size = `Degree`/`Pagerank`; color = **Modularity** (computed in Gephi)  
4. **Labels:** enable + *Label Adjust*  
5. **Export:** PNG/SVG or save `.gexf` with positions

> **Empty Graph Tip:** If few edges or most weights=1, check `MentionedUsers` and `Reply to userid` in X CSV. Enable extraction of `@handles` from text/URLs if missing.

---

## 9) Tableau (Minimum Viable Setup)
- Connect to `data/processed/` + `results/graphs/x_nodes_metrics.csv`.  
- Create a **Data Extract**.  
- Relate by `author` ↔ `label` (or `@handle` if unified).

**Quick Views:**
- *Timeline:* `COUNTD(item_id)` by day, colored by `sentiment_label`.
- *Top Authors:* bar chart by `author` with `COUNTD(item_id)` and `AVG(sentiment_score)`.
- *Emotions Heatmap:* using `emotions_long.csv` (`emotion`×`author`, color=`AVG(prob)`).

---

## 10) Suggested Improvements (Denser X Network)
1. Flexible column normalization (aliases/typos).  
2. Extract `@handles` from **text** and **URLs** (retweets/quotes).  
3. Label nodes with `@handle` when available (more readable than ID).  
4. Include `retweet` and `quote` edges if available.  
5. Weighted metrics so `degree/pagerank` reflect interaction strength.

> These enhancements will be available in an alternative version of `preprocessing.py` and `network.py` and have to be integrated.

---

## 11) Performance and Reproducibility
- **Batching:** `add_sentiment(batch_size=64)`, `add_emotions(batch_size=16)`.
- **CPU vs GPU:** `--device 0` for CUDA; `--device -1` for CPU.
- **Export:** `export_tableau_csv` uses `utf-8-sig` and `sep=';'`.

---

## 12) Troubleshooting
- **Sparse graph:** check `data/processed/x_edges.csv`; if poor, extract `@handles` from text/URLs.  
- **Missing columns (X):** aliases included; extend if your exporter uses different names.  
- **Memory (emotions):** lower `batch_size` or use the lightweight model.

---

## 13) Short Roadmap
- [ ] Thematic filter (*keyword sets*) for Tableau.  
- [ ] Export Gephi node positions (`node_positions.csv`) for reproducible layouts in Tableau.  
- [ ] More robust language detection (fallback).  
- [ ] CI to regenerate outputs from new CSVs.

---

## 14) Credits
Models: CardiffNLP and Hugging Face community  
Libraries: `transformers`, `pandas`, `networkx`, `tqdm`, `emoji`

## Author

**Rodrigo Medrano**  
_Year: 2025_
