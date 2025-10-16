# SOCIAL ANALYSIS PROJECT · TELEGRAM + X

Multichannel social analytics pipeline built with Python. The project ingests Telegram and X/Twitter exports, cleans and unifies both sources, scores multilingual sentiment and zero-shot emotions, extracts entity-conditioned context, fits BERTopic, and builds an interaction graph for X. Outputs are optimized for Tableau and Gephi.

## Overview
- Flexible CSV ingestion for Telegram and X with resilient column alias detection.
- Unified preprocessing pipeline in `src/preprocessing.py` that normalizes metrics and text fields.
- Hugging Face models: sentiment (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) and emotions (`joeddav/xlm-roberta-large-xnli` by default).
- Topic discovery via BERTopic with optional disk cache.
- Entity-aware sentiment/emotion/topic scoring (defaults: NATO/OTAN and Russia).
- Directed interaction network for X (mentions, replies, retweets, quotes) exported as `.gexf` plus metrics.

## Repository Layout
```text
social-analysis-project/
├── data/
│   ├── raw/                       # place telegram.csv, x.csv or other raw exports here
│   └── processed/                 # pipeline outputs (CSV, UTF-8 with BOM)
├── models/
│   └── bertopic/                  # optional BERTopic cache (created at runtime)
├── results/
│   ├── graphs/                    # network metrics and GEXF graphs
│   └── topics/                    # BERTopic summaries and metadata
├── scripts/
│   └── process_all.py             # main entry point
├── src/                           # reusable library modules
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── emotions.py
│   ├── entity_analysis.py
│   ├── entities_runtime.py
│   ├── topics_bertopic.py
│   ├── network.py
│   └── utils.py
├── requirements.txt
└── README.md
```
> Empty folders include `.gitkeep` files to preserve the tree.

## Requirements
- Python 3.10 or 3.11 (tested on macOS Apple Silicon).
- `pip`, `venv`, or `conda` for environment management.
- PyTorch (CPU by default; GPU optional via `--device`).
- First run requires internet access to download Hugging Face models.

## Quick Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Expected Inputs
### Telegram (`data/raw/telegram.csv`)
- Required columns detected automatically: `timestamp`, `channel`, `messageId`, `summary/text`.
- Metric aliases (`reactions`, `forwards`, `views`, etc.) are mapped to the unified analytics columns `likes`, `retweets`, `replies`, `quotes`, `views`.
- Generates cleaned fields such as `text_clean`, `emoji_count`, `uid`, `link`.

### X / Twitter (`data/raw/x.csv`)
- Detects aliases for `Tweet ID`, `URL`, `Content`, `Author`, `Language`, metrics, and dates.
- Missing `Tweet ID` values are inferred from URLs or replaced with a deterministic hash.
- Normalizes dates to `YYYY-MM-DD` and extracts emojis/mentions when available.

Both readers accept `, ; \t |` separators, UTF-8 with or without BOM, and skip malformed lines with `on_bad_lines="skip"`.

## Running the Pipeline
```bash
python scripts/process_all.py \
  --telegram data/raw/telegram.csv \
  --x data/raw/x.csv \
  --device -1 \
  --max_rows 0 \
  --emotion_model joeddav/xlm-roberta-large-xnli \
  --entities "OTAN,Rusia" \
  --entity_window 160
```

Key parameters:
- `--telegram`, `--x`: CSV file paths (optional per platform).
- `--max_rows`: limit rows for smoke tests (`0` keeps all data).
- `--device`: GPU index (0,1,...) or CPU (`-1`).
- `--emotion_model`: zero-shot model name; switch to `MoritzLaurer/mMiniLMv2-L6-mnli-xnli` for lighter workloads.
- `--entities`: comma-separated list of entity names (aliases inferred automatically).
- `--entities_file`: YAML/JSON/TXT file with custom entity definitions and aliases.
- `--entity_window`: number of characters captured around each mention for targeted scoring.

Directories are created on demand through `utils.ensure_dirs`.

## Core Outputs (`data/processed/`)
- `telegram_sentiment.csv`, `x_sentiment.csv`: platform-level enriched datasets.
- `all_platforms.csv`: combined dataset containing topic, sentiment, emotion, and entity mention payloads (per-row JSON).
- `facts_posts.csv`: Tableau-ready fact table with aggregated engagement.
- `emotions_long.csv`: tidy emotion probabilities (`item_id`, `emotion`, `prob`).
- `topics_assignments.csv`, `topics_summary_daily.csv`: BERTopic assignments and daily evolution.
- `entity_mentions.csv`, `entity_topic_summary.csv`: entity-conditioned sentiment, stance, and emotion aggregations.
- `x_edges.csv`: edge list for Tableau/Gephi workflows.
- `../results/graphs/x_nodes_metrics.csv`, `../results/graphs/x_interactions.gexf`: network metrics and GEXF graph.
- `../results/topics/topic_info.csv`: topic catalogue exported by BERTopic.

> All CSV files use `;` as separator and `utf-8-sig` encoding for compatibility with Excel/Tableau.

## Key Components
- `src/preprocessing.py`: resilient CSV loaders, date normalization, emoji extraction, and unified schema (`unify_frames`).
- `src/sentiment.py`: batch wrapper around the multilingual sentiment pipeline (XLM-R) returning full distributions.
- `src/emotions.py`: multilingual zero-shot classifier with configurable batches and model name.
- `src/network.py`: builds the directed X graph and exports metrics/edges.
- `src/topics_bertopic.py`: trains or reloads BERTopic, transforms documents, and summarizes topics over time.
- `src/entities_runtime.py`: loads entities from CLI arguments or auxiliary files.
- `src/entity_analysis.py`: detects mentions, scores targeted sentiment/emotions, and aggregates by entity/topic.
- `src/utils.py`: shared helpers (directory creation, source normalization, Tableau-friendly CSV export).

## Topics and Entity Analytics
- BERTopic is trained on the combined `text_clean` corpus; if `models/bertopic/global/` already exists the cached model is reused.
- Topic assignments populate `topic_id`, `topic_label`, and `topic_score` within the unified dataset.
- Entity analysis supports CLI lists or structured files, and persists `entity_mentions` JSON per post for dashboarding.
- `entity_topic_summary.csv` stores stance indices/labels and average emotion distributions per `(entity, topic_id)`.

## Best Practices & Troubleshooting
- **Large models:** if memory is constrained, switch to a lighter emotion model or reduce the batch size in `add_emotions`.
- **Sparse graph:** inspect `data/processed/x_edges.csv` to confirm mention/reply extraction in the X export.
- **Date issues:** loaders normalize to `YYYY-MM-DD`; verify `timestamp` if the source includes time zones.
- **Reproducibility:** BERTopic uses a fixed `seed` to stabilize topic assignments.

## Credits
- Models: CardiffNLP and the Hugging Face community.
- Core libraries: `transformers`, `pandas`, `networkx`, `tqdm`, `emoji`, `bertopic`, `sentence-transformers`.
- Author: Rodrigo Medrano (2025).
