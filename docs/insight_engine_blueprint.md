# Insight Engine Blueprint

This blueprint captures the canonical data schema, the end-to-end pipeline (scraper → analytics), and the quality KPIs that keep the NATO airspace incursion use case reliable. It is written to be domain-agnostic so the same structure can be reused for future verticals.

---

## Canonical Dataset Schema

| Field | Dataset(s) | Type | Description |
| --- | --- | --- | --- |
| `item_id` | all_platforms, facts_posts, entity_mentions | string | Stable unique identifier (`messageId` for Telegram, `tweet_id` for X, deterministic hash fallback). |
| `source` | all_platforms, facts_posts | enum | `Telegram` or `X`. |
| `timestamp` | all_platforms, facts_posts | date / ISO datetime | Normalized publication date (`YYYY-MM-DD` in the unified dataset; ISO 8601 in the scraper/API). |
| `message_timezone` | telegram scraper store | string | Time zone label inferred from metadata (UTC±HH:MM). |
| `author` | all_platforms, facts_posts | string | Channel name (Telegram) or display name/handle (X). |
| `lang` | all_platforms, facts_posts | ISO-639-1 | Language detected by `langdetect` (n-grams). |
| `text` / `text_clean` | all_platforms, telegram_preprocessed/x_preprocessed | string | Raw text and whitespace-normalized text used for NLP. |
| `text_topic` | all_platforms | string | Text cleaned for topic modeling (lemmatized, multilingual stopword-free). |
| `likes`, `retweets`, `replies`, `quotes`, `views` | all_platforms, facts_posts | int | Engagement metrics harmonised across platforms. |
| `engagement` | all_platforms, facts_posts | float | Sum of likes + retweets/forwards + replies + quotes (auto-filled when columns exist). |
| `topic_id`, `topic_label`, `topic_score` | all_platforms, topics_assignments, entity_mentions | int, string, float | BERTopic assignment, label, and per-document score (higher = better fit). |
| `topic_terms` | all_platforms, facts_posts, topics_assignments | JSON (list) | Ordered list of top terms per topic (for dashboards). |
| `entities_detected` | all_platforms, facts_posts | JSON (list) | Unique entity names detected in the post (JSON string in exports). |
| `n_entity_mentions` | all_platforms, facts_posts, entity_topic_summary | int | Count of entity mentions extracted from the post. |
| `stance` | all_platforms, facts_posts, entity_mentions | enum (`pos`, `neu`, `neg`) | Sign of the entity-conditioned sentiment. |
| `stance_value` | all_platforms, facts_posts, entity_mentions | float | Weighted stance index (−1 negative, 0 neutral, +1 positive). |
| `sentiment_label`, `sentiment_score`, `sentiment_dist` | all_platforms, facts_posts, entity_mentions | string, float, JSON | Aggregated targeted sentiment label, confidence, and probability distribution. |
| `emotion_label`, `emotion_scores` | all_platforms, facts_posts, entity_mentions | string, JSON | Dominant emotion and distribution inferred from the mention context. |
| `impact_score` | all_platforms, facts_posts, entity_mentions, entity_topic_summary | float | Entity-conditioned impact metric (stance × sentiment confidence × log engagement × emotion/topic weights). |
| `impact_score_mean` | all_platforms, facts_posts, entity_topic_summary | float | Mean impact for the post (or entity/topic cell). |
| `entity_mentions` | all_platforms, entity_mentions | JSON | Per-entity payload with stance, sentiment, emotions, snippet, impact_score, engagement, reach. |
| `geolocations` | telegram scraper store/API | list | List of coordinate hits or heuristic location hints (`source`, `lat`, `lon`, `country`, `confidence`). |
| `message_timezone`, `tz_message`, `tz_author` | scraper store, API response | string | Time zone hints used for geolocation heuristics. |
| `link` | all_platforms, facts_posts | string | Permalink when available. |

> **Impact score definition (per mention)**:  
> `impact = stance_value × sentiment_score × (0.6 + 0.4·topic_score∈[0,1]) × (0.7 + 0.3·max_emotion_prob) × log1p(engagement + 0.25·reach)` clipped to ±25. The post-level score is the sum of its mentions.

---

## Pipeline Overview

### Telegram Scraper (`telegram-scraper`)
1. **Discovery & Filtering**
   - Channel search honors `config/channel_filters.json` (`blocked_usernames`, regex patterns) plus request-level exclusions.
   - Broadcast channels can be toggled via `INCLUDE_BROADCAST`.
2. **Collection**
   - Messages pulled with Telethon (`iter_messages`) respecting absolute windows (`date_from/date_to`) or `since_days`.
   - Media bytes stored in GridFS when requested; metadata appended to `media_meta`.
3. **Normalization**
   - Text normalized (whitespace, emoji demojized), language detected (`langdetect`), metrics mapped to unified fields.
   - Message time zone labeled (`timezone_label`) and persisted as `message_timezone`.
4. **Geolocation**
   - Primary: metadata geo (venues/live) and coordinate patterns in text.
   - Fallback heuristics: timezone + language + username/title + keywords mapped through `LOCATION_CATALOG` to return country hints with confidence scores.
5. **Storage / API**
   - Upserted into Mongo (messages + channel metadata) and exposed via `/messages/recent` FastAPI endpoint. Response includes `geolocations`, `tz_message`, `lang`.

### Insight Engine (`social-analysis-project`)
1. **Preprocessing**
   - `load_telegram` / `load_x` harmonise raw CSVs; `add_engagement` computes aggregate interaction.
   - Multilingual stopword removal + lemmatization produce `text_topic`, optimised for BERTopic, and we retain `topic_terms` arrays generated from top n-grams.
   - Frames unified via `unify_frames` with consistent IDs and text fields.
2. **Topic Modelling**
   - BERTopic fits on `text_clean`, stores assignments and daily summaries, resets when cache absent.
3. **Entity-conditioned Scoring**
   - `extract_entity_mentions` finds aliases within the topic-aware text window (`entity_window`).
   - `score_entity_mentions` runs targeted sentiment/emotions per mention and derives `impact_score`.
   - `aggregate_mentions_per_item` enriches posts with stance, sentiment, emotion, impact, and entity coverage.
4. **Exports**
   - `facts_posts.csv` and `all_platforms.csv` carry the canonical schema for BI tools (Tableau, Gephi), including `topic_terms` and emotion probability columns (`emotion_prob_*`).
   - `entity_mentions.csv` (long format) and `entity_topic_summary.csv` summarise per-entity metrics, allowing comparisons across topics, days, or countries.
   - `emotions_long.csv`, `topics_assignments.csv`, `x_edges.csv` serve specialised dashboards.

5. **Downstream Readiness**
   - All CSV outputs are UTF-8 with BOM and semicolon-separated to stay Excel/Tableau-friendly.
   - JSON payload fields (`entity_mentions`, `entities_detected`, `sentiment_dist`, `emotion_scores`) are serialized to ensure consistent ingestion.

---

## Quality KPIs

| KPI | Target / Method | Notes |
| --- | --- | --- |
| **Topic Coherence** | Track BERTopic topic coherence (e.g., `c_v`) per run; flag drops > 10%. | Use BERTopic's internal metrics or re-score via `bertopic.get_topic_info()`. |
| **Entity Stance Precision** | Manual spot-check / active learning: ≥80% agreement on stance label vs analyst judgment. | Sample by entity-topic pairs with high impact to prioritise validation. |
| **Impact Score Stability** | Monitor mean and std per entity to catch spikes. Sudden ±3σ shifts trigger review. | Distinguishes organic vs manipulated narratives (e.g., coordinated bursts). |
| **Entity Coverage** | `% posts with n_entity_mentions > 0`. Target depends on entity list; track trending downwards. | Low coverage suggests alias gaps or filter issues. |
| **Geo Hint Hit Rate** | `(heuristic hints + coordinates with country match) / total posts`. Aim ≥60% for channels with consistent signals. | Investigate if timezone/alias dictionaries need updates. |
| **Pipeline Latency** | End-to-end time (scrape → processed CSV) per batch. Keep under 30 min for daily updates. | Broken down by scraping, topic modelling, entity scoring. |
| **LLM QA Accuracy (future)** | Spot-check LLM-generated “what-if” answers vs analyst baseline once integrated. | Placeholder until LLM layer is wired through n8n / orchestration. |

---

## Next Steps

- Integrate the LLM “what-if” layer (n8n + OpenAI/Gemini) using the canonical datasets as context, returning explanations grounded in `impact_score` trends.
- Expand `LOCATION_CATALOG` as new theatres emerge (e.g., Baltic Sea, Arctic) and feed analyst feedback back into alias lists.
- Automate KPI tracking (e.g., lightweight dashboard or notebook) so regressions surface immediately after each batch run.
- Consider adding post-level confidence fields (e.g., `impact_confidence`) if multiple mentions disagree sharply, useful for alert prioritisation.
