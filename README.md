# Fake News Detection & Verification System

A Streamlit research demo that combines a TF-IDF + Logistic Regression text classifier with optional Wikipedia evidence retrieval, sentence-embedding relevance ranking, and BART-MNLI evidence comparison.

**This is not a definitive fact-checking service.** Predictions and evidence-model outputs are fallible and should be checked against independent primary sources.

## Requirements

- Windows 10/11 or a supported Linux host
- Python 3.13 (tested locally with Python 3.13.7)
- 4 GB RAM minimum for classifier-only use; 8 GB RAM recommended when using the optional BART-MNLI model
- Internet access for first-time Hugging Face model downloads and Wikipedia searches

The saved classifier is compatible with scikit-learn 1.7.2. Use the pinned dependencies in `requirements.txt`.

## Install and run on Windows

From the project root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell prevents virtual-environment activation, run the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit. The app loads `fake_news_model.joblib` relative to `app.py`; launching from another working directory is supported.

## Train or evaluate the classifier

Training needs the supplied `True.csv` and `Fake.csv` files, which are not needed at inference time:

```powershell
.\.venv\Scripts\python.exe save_model.py
```

The script normalizes, removes empty records and exact normalized-text duplicates before a stratified holdout, trains TF-IDF + Logistic Regression, prints class metrics and confusion matrix, and saves model/evaluation metadata to `fake_news_model.joblib`.

## Tests and startup check

Run the focused unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Check the artifact and a prediction in a fresh process:

```powershell
.\.venv\Scripts\python.exe -c "import joblib; a=joblib.load('fake_news_model.joblib'); x=a['tfidf'].transform(['A detailed news report']); print(a['model'].predict(x))"
```

Start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Evidence and privacy

Evidence lookup uses Wikipedia's public MediaWiki API and caches successful search results in Streamlit for up to one hour. Optional embedding and NLI models download from Hugging Face the first time they are needed and are cached in memory for the application process. If retrieval or an optional model fails, the classifier remains separate and available, and the evidence status is presented as unavailable or unverified.

Claim text is not stored in a project database. Submitting a claim sends it to Wikipedia for search. Do not submit confidential or personal information.

## Limitations

- The model learns dataset-specific language and source patterns. The provided corpus is not representative of all publishers, claims, topics, or current events.
- Random, title-grouped, and chronological splits are still evaluations on the same source dataset; none establishes real-world reliability.
- Dataset subject labels are strongly associated with the REAL/FAKE label and likely reflect the source construction. They are not classifier inputs, but source/style cues can still be present in article text.
- Classifier scores are not shown as confidence because they have not been calibrated for real-world claims.
- Wikipedia summaries and NLI labels are not proof. Thresholds are heuristic and ambiguous or conflicting evidence must remain unresolved.
- The first NLI request can be slow and memory intensive. Use a host with sufficient memory or expect optional evidence comparison to fail gracefully.

See [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for audit findings, evaluation details, and hosting guidance.
