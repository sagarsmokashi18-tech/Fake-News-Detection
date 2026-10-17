# Project Audit: Fake News Detection & Verification System

## Executive summary

This repository contains a Streamlit app, two CSV datasets, a TF-IDF + Logistic Regression artifact, and a prototype training script. The application is now better guarded and reports classification separately from optional evidence review. The model's strong same-dataset metrics are not evidence of real-world fact-checking accuracy: labels are entangled with collection source, subjects are disjoint by class, and the inputs represent a narrow historical corpus.

## Scope and observed structure

- `app.py`: Streamlit interface, model inference, Wikipedia API client, optional embedding and NLI inference.
- `save_model.py`: dataset cleaning, stratified holdout, vectorizer/model training, metrics and artifact serialization.
- `text_processing.py`: shared text normalization for training and inference.
- `fake_news_model.joblib`: serialized classifier and vectorizer, roughly 0.91 MiB in the current working tree.
- `True.csv` and `Fake.csv`: combined 44,898 rows before cleaning; approximately 111 MiB combined.
- `fake_news.py`: older exploratory training script with preprocessing/splitting differences; not used by the app.

## Findings

### High: strong dataset-source and temporal confounding

Evidence:

- The REAL rows use two subject values (`politicsNews`, `worldnews`); FAKE rows use six different subject values. There is no subject-label overlap in the provided files.
- The source datasets use different field names and are not an interleaved sample from the same publishers: the REAL file has a `dates` column, the FAKE file has `date`.
- The data is historical and largely ends in 2017/early 2018.

Impact: article wording, source conventions, topic tags, and time may allow a model to recognize corpus origin instead of learning general veracity. The subject column is not passed to the classifier, but source/style signals can remain in article text. All internal holdouts share this source construction.

Recommendation: do not describe scores as real-world accuracy. For a stronger claim, collect independently sourced, publisher-disjoint, current, human-verified data and define a prospective evaluation before training.

### High: verification models cannot establish truth

Evidence: Wikipedia introductions may be incomplete or unrelated. Sentence similarity and NLI labels are model outputs, not independently verified evidence. The thresholds are heuristic. The NLI comparison now uses the evidence as premise and the user's text as the paired claim, but this remains a coarse classifier.

Impact: even high model scores can be mistaken or lack context.

Mitigation: the UI distinguishes classification and evidence, suppresses classifier probability/confidence, requires a relevance gate before NLI, handles unavailable models as unverified, and warns on conflicting signals. Human review and primary sources are still required.

### Medium: optional model latency and memory requirements

Evidence: `all-MiniLM-L6-v2` and `facebook/bart-large-mnli` download on first use. The app pins them to CPU and caches them in memory per process. BART-large-MNLI is substantially larger than the classifier artifact.

Impact: first use may be slow, cold starts repeat downloads, and small hosts may run out of memory.

Mitigation: load models lazily, cache successful loads, display unavailable states, and document an 8 GiB recommended host size for NLI use. Actual cold model-download and peak-memory measurements still depend on host/network and are not claimed here.

### Medium: weak original deployment reproducibility

Evidence: no dependency manifest, setup instructions, or project-level ignore file were present. A fresh artifact load emitted an `InconsistentVersionWarning` when the global Python 3.13 interpreter (scikit-learn 1.7.1) loaded an artifact serialized by scikit-learn 1.7.2.

Mitigation: the model was rebuilt under the project venv's pinned scikit-learn 1.7.2; `requirements.txt`, `.gitignore`, setup docs, and fresh-process checks are added. Do not load the artifact outside its pinned environment.

### Medium: duplicate and empty data records

Evidence: 44,898 raw rows; 631 texts normalize to empty; 6,280 normalized duplicate rows include the repeated empty texts. After removing empties and exact normalized-text duplicates, 38,617 rows remain (21,170 REAL and 17,447 FAKE). The only cross-label normalized duplicate group in the raw data was the empty-text group; no non-empty exact normalized text group crossed labels. There are 6,178 repeated title rows before the final unique-text set.

Mitigation: training removes normalized empty rows and exact normalized-text duplicates before splitting. A title-group holdout was also measured. A title is not a reliable publisher or full near-duplicate key, so more extensive publisher/provenance deduplication remains open.

### Medium: external API and UI safety

Evidence: the original app treated network and model failures inconsistently and placed external text into HTML.

Mitigation: Wikipedia calls now have explicit connect/read timeouts, bounded retries, response-shape checks, one-hour result caching, and explicit errors. Evidence card fields are HTML-escaped, links are generated to the Wikipedia article host, and missing/weak/contradictory evidence is not reported as definitive.

### Low: privacy and user-input handling

Evidence: claims are sent to Wikipedia when analysis runs. The app does not keep a project database, but Streamlit caches lookup results for up to one hour.

Mitigation: the UI and README disclose that public API use and advise against sending confidential or personal data. No application API keys are required.

## Evaluation

See [EVALUATION.md](EVALUATION.md) for exact stratified, title-group, and time-based holdout metrics and split limitations. The measured 98.76% accuracy is reproducible on this deduplicated dataset but is not a public-deployment quality guarantee.

## Remaining blockers and recommendations

1. Do not present the internal score as real-world accuracy or the product as a definitive fact-checker.
2. Validate with a publisher-disjoint, current, independently labeled dataset before making performance claims.
3. Test the deployment target's actual RAM, disk, network access, model download duration, concurrency, and cold-start behavior.
4. Confirm no staged large/old artifacts or local environments are included before committing or publishing; inspect `git status` and the index. No files were deleted as part of this audit.
