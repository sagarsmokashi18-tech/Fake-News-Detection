# Deployment Guide

## Deployment status

The application can run locally with the provided classifier artifact. Do not represent the classifier as production-validated: the dataset is source-biased and the reported holdouts are internal dataset evaluations only. Public deployment is suitable only as an educational demonstration with this limitation displayed prominently.

The classifier artifact is about 0.91 MiB. The training CSVs total about 116 MiB and are not required at runtime. Do not upload the CSVs to a hosting provider unless training is performed there.

## Clean local environment (Windows PowerShell)

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -c "import joblib; a=joblib.load('fake_news_model.joblib'); print(a['metadata']['scikit_learn_version']); print(a['model'].predict(a['tfidf'].transform(['A detailed news report'])))"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The artifact stores scikit-learn estimator objects. Keep `scikit-learn==1.7.2` pinned when loading it. Retrain with `save_model.py` after changing that version, text preprocessing, vectorizer settings, or training data.

## Streamlit Community Cloud or another Linux host

1. Create a deployment repository containing `app.py`, `text_processing.py`, `fake_news_model.joblib`, and `requirements.txt`.
2. Do not include `venv/`, `.venv/`, `__pycache__/`, `.env`, `.streamlit/secrets.toml`, or training CSVs unless needed for a separate training workflow.
3. Set the entry point to `app.py`.
4. Select Python 3.13 where the host permits; otherwise use a Python version supported by all pinned packages and test it before deployment.
5. Allocate at least 4 GiB memory for the classifier and runtime; 8 GiB is recommended if users will run BART-MNLI evidence comparison. Hardware limits and concurrent users may require more.
6. Configure no secrets for the current app. Never add API credentials to source control. If a future provider requires a key, use the host's secret manager and `st.secrets`.
7. Expect the first request that uses Sentence-BERT or BART-MNLI to download model weights from Hugging Face. Ensure outbound access and enough disk, RAM, and startup time. Subsequent calls in the same live process reuse cached models; a cold restart downloads/reloads them again.
8. Verify the deployed URL manually: check classifier-only path, successful/no-result Wikipedia retrieval, API errors, and optional model availability. Do not announce deployment as successful until these live checks pass.

## Runtime and resource behavior

- The saved TF-IDF classifier is loaded once per Streamlit process.
- Wikipedia results are cached for one hour. Network requests have connect/read timeouts and a bounded retry policy for transient errors and rate limiting.
- Sentence-BERT and BART-MNLI are loaded lazily and cached once per process.
- Sentence-BERT and BART-MNLI use CPU inference. BART-large-MNLI is memory intensive and may fail to load on constrained hosts; such a failure must remain an explicit unavailable/unverified state, never an implied verdict.
- Multiple app replicas each maintain their own in-memory model and search caches.

## Troubleshooting

### Model artifact or scikit-learn compatibility error

Confirm the package version and artifact metadata:

```powershell
.\.venv\Scripts\python.exe -c "import sklearn, joblib; a=joblib.load('fake_news_model.joblib'); print('sklearn', sklearn.__version__); print('trained with', a.get('metadata', {}).get('scikit_learn_version'))"
```

Install the pinned requirements or retrain the model with the target environment. Do not ignore an `InconsistentVersionWarning`.

### Optional models fail to download or load

The classifier remains usable, but semantic ranking or NLI comparison will be marked unavailable. Confirm outbound HTTPS access to Hugging Face and available disk/RAM. Restart after increasing host memory if the model load caused an out-of-memory failure.

### Wikipedia lookup unavailable

The claim classifier remains available. Check outbound HTTPS access to `en.wikipedia.org`, wait if the API reports rate limiting, and retry later.

### Port already in use

Choose a different port:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## No public deployment performed

This guide and local startup checks do not constitute a public deployment. No hosting provider has been configured or validated as part of this change.
