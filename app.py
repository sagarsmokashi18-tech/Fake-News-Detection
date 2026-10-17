import base64
import html
import logging
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import quote

import joblib
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from text_processing import normalize_text

st.set_page_config(page_title="Fake News Detection & Verification System", page_icon="🧠", layout="wide")

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fake_news_model.joblib"
BACKGROUND_IMAGE_PATH = PROJECT_DIR / "tech-background.jpg"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
SIMILARITY_THRESHOLD = 0.65
NLI_SCORE_THRESHOLD = 0.60
EXAMPLE_CLAIMS = (
    "The 2024 election was canceled in the United States.",
    "A major vaccine trial showed a 100% success rate in all patients.",
    "The moon landing was faked by Hollywood filmmakers.",
)


class EvidencePage(TypedDict, total=False):
    title: str
    extract: str
    pageid: int
    similarity: float
    verdict: str
    score: float


class WikipediaServiceError(RuntimeError):
    pass


class OptionalModelUnavailable(RuntimeError):
    pass


def validate_claim_input(claim: str) -> Optional[str]:
    if not claim or not claim.strip():
        return "Please enter a claim or article before analysis."
    if len(claim.strip()) < 8:
        return "The input is too short to classify reliably. Please add a more complete claim or article."
    if len(claim.strip()) > 10000:
        return "The input is too long for this demo. Please limit it to 10,000 characters."
    if len(normalize_text(claim)) < 8:
        return "The text does not contain enough meaningful content after cleaning."
    return None


@st.cache_resource
def load_model_artifact() -> dict[str, Any]:
    try:
        artifact = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(f"The trained model file '{MODEL_PATH.name}' was not found.") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to read the trained model artifact: {exc}") from exc

    if not isinstance(artifact, dict) or "model" not in artifact or "tfidf" not in artifact:
        raise RuntimeError("The model artifact is incomplete or invalid.")
    return artifact


@st.cache_resource
def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as exc:
        LOGGER.exception("Sentence-BERT could not be loaded.")
        raise OptionalModelUnavailable("Semantic relevance ranking is unavailable.") from exc


@st.cache_resource
def load_nli_model():
    try:
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=-1,
            top_k=None,
        )
    except Exception as exc:
        LOGGER.exception("The NLI model could not be loaded.")
        raise OptionalModelUnavailable("Natural-language evidence comparison is unavailable.") from exc


@st.cache_data(ttl=3600)
def fetch_wikipedia_articles(query: str, limit: int = 3) -> list[EvidencePage]:
    if len(normalize_text(query)) < 4:
        return []

    retry_policy = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=False,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "FakeNewsDetection/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))

    try:
        search_response = session.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": limit,
                "format": "json",
                "utf8": 1,
                "formatversion": 2,
            },
            timeout=(5, 12),
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        if not isinstance(search_data, dict) or not isinstance(search_data.get("query"), dict):
            raise ValueError("Wikipedia search response did not include query results.")
        search_results = search_data["query"].get("search")
        if not isinstance(search_results, list):
            raise ValueError("Wikipedia search response did not include a result list.")
        titles = [item["title"] for item in search_results if item.get("title")]
        if not titles:
            return []

        extract_response = session.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exchars": 700,
                "titles": "|".join(titles),
                "format": "json",
                "formatversion": 2,
                "redirects": 1,
            },
            timeout=(5, 12),
        )
        extract_response.raise_for_status()
        extract_data = extract_response.json()
        if not isinstance(extract_data, dict) or not isinstance(extract_data.get("query"), dict):
            raise ValueError("Wikipedia extract response did not include query results.")
        pages = extract_data["query"].get("pages")
        if not isinstance(pages, list):
            raise ValueError("Wikipedia extract response did not include a page list.")
        return [
            EvidencePage(
                title=page.get("title", "Wikipedia article"),
                extract=page["extract"].strip(),
                pageid=page.get("pageid"),
            )
            for page in pages
            if isinstance(page.get("extract"), str) and page["extract"].strip()
        ][:limit]
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 429:
            message = "Wikipedia is rate limiting requests. Please try again later."
        else:
            message = "Wikipedia returned an HTTP error while searching for evidence."
        LOGGER.warning("Wikipedia HTTP request failed with status %s.", status_code)
        raise WikipediaServiceError(message) from exc
    except requests.exceptions.RetryError as exc:
        LOGGER.warning("Wikipedia request retries were exhausted: %s", exc)
        raise WikipediaServiceError("Wikipedia is temporarily unavailable after repeated retries.") from exc
    except requests.Timeout as exc:
        LOGGER.warning("Wikipedia request timed out: %s", exc)
        raise WikipediaServiceError("The Wikipedia search timed out. The classifier result is still available.") from exc
    except requests.RequestException as exc:
        LOGGER.warning("Wikipedia request failed: %s", exc)
        raise WikipediaServiceError("Wikipedia could not be reached. The classifier result is still available.") from exc
    except (ValueError, AttributeError, TypeError, KeyError) as exc:
        LOGGER.exception("Wikipedia returned an invalid response.")
        raise WikipediaServiceError("Wikipedia returned an invalid response. The classifier result is still available.") from exc


def rank_wikipedia_evidence(claim: str, pages: list[EvidencePage]) -> list[EvidencePage]:
    if not pages:
        return []

    try:
        sentence_model = load_sentence_model()
        from sentence_transformers import util

        texts = [normalize_text(claim)] + [normalize_text(page["extract"]) for page in pages]
        embeddings = sentence_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        similarities = util.cos_sim(embeddings[0], embeddings[1:]).flatten().tolist()
    except OptionalModelUnavailable:
        raise
    except Exception as exc:
        LOGGER.exception("Semantic similarity calculation failed.")
        raise OptionalModelUnavailable("Semantic relevance ranking failed.") from exc

    ranked = [
        EvidencePage(**page, similarity=float(score))
        for page, score in zip(pages, similarities)
    ]
    return sorted(ranked, key=lambda page: page["similarity"], reverse=True)


def _interpret_nli_result(result: list) -> tuple[str, float]:
    """Map a single NLI result (a list of label/score dicts) to a verdict."""
    if not isinstance(result, list) or not result:
        raise OptionalModelUnavailable("The NLI model returned an invalid result.")

    top_result = max(result, key=lambda item: float(item.get("score", 0.0)))
    candidate = str(top_result.get("label", "")).lower()
    score = float(top_result.get("score", 0.0))
    if score < NLI_SCORE_THRESHOLD:
        return "Insufficient information", score
    if "entailment" in candidate:
        return "Supports", score
    if "contradiction" in candidate:
        return "Refutes", score
    return "Insufficient information", score


def verify_claim_against_evidence(claim: str, evidence_text: str) -> tuple[str, float]:
    try:
        nli_pipeline = load_nli_model()
        result = nli_pipeline(
            {
                "text": evidence_text[:1600],
                "text_pair": claim[:1000],
            },
            truncation=True,
            max_length=1024,
        )
    except OptionalModelUnavailable:
        raise
    except Exception as exc:
        LOGGER.exception("NLI comparison failed.")
        raise OptionalModelUnavailable("Natural-language evidence comparison failed.") from exc
    if result and isinstance(result[0], list):
        result = result[0]
    return _interpret_nli_result(result)


@st.cache_data(ttl=3600, show_spinner=False)
def verify_claim_against_evidence_batch(claim: str, evidence_texts: tuple[str, ...]) -> list[tuple[str, float]]:
    """Run the NLI model once for all evidence texts and cache the verdicts.

    Batching keeps a single model forward pass per request, and the result is
    cached by ``(claim, evidence)`` so repeat or re-run queries are instant.
    """
    if not evidence_texts:
        return []
    try:
        nli_pipeline = load_nli_model()
        payload = [{"text": text[:1600], "text_pair": claim[:1000]} for text in evidence_texts]
        raw = nli_pipeline(payload, truncation=True, max_length=1024)
    except OptionalModelUnavailable:
        raise
    except Exception as exc:
        LOGGER.exception("NLI comparison failed.")
        raise OptionalModelUnavailable("Natural-language evidence comparison failed.") from exc

    # A list input yields one result per pair; a scalar-shaped result is wrapped.
    items = raw if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else [raw]
    return [_interpret_nli_result(item) for item in items]


def predict_claim(claim: str) -> str:
    artifact = load_model_artifact()
    clean_claim = normalize_text(claim)
    features = artifact["tfidf"].transform([clean_claim])
    return str(artifact["model"].predict(features)[0])


def predict_claim_with_confidence(claim: str) -> tuple[str, Optional[float]]:
    """Return the predicted label and the classifier's own probability.

    The model (Logistic Regression) exposes ``predict_proba``; the returned
    value is a genuine model output rather than a hardcoded number. It is an
    uncalibrated probability and is presented as such in the UI.
    """
    artifact = load_model_artifact()
    clean_claim = normalize_text(claim)
    features = artifact["tfidf"].transform([clean_claim])
    model = artifact["model"]
    label = str(model.predict(features)[0])
    confidence: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Classifier confidence could not be computed.")
            confidence = None
    return label, confidence


def wikipedia_url(title: str) -> str:
    return f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'), safe='()_')}"


def build_card(page: EvidencePage, tone: str) -> str:
    title = html.escape(page.get("title", "Wikipedia article"))
    snippet = html.escape(page.get("extract", "").replace("\n", " "))
    url = html.escape(wikipedia_url(page.get("title", "")), quote=True)
    label = html.escape(tone.title())
    similarity = page.get("similarity")
    relevance = f"<span class='relevance'>Relevance score: {similarity:.2f}</span>" if similarity is not None else ""
    return f"""
    <article class="evidence-card">
      <div class="card-header"><span class="pill {tone}">{label}</span><strong>{title}</strong></div>
      <p>{snippet}</p>
      {relevance}
      <a href="{url}" target="_blank" rel="noopener noreferrer">Open Wikipedia article</a>
    </article>
    """


@st.cache_data(show_spinner=False)
def get_background_css_value() -> str:
    """Base64-encode the tech background once per session for efficient reuse."""
    try:
        if not BACKGROUND_IMAGE_PATH.exists():
            LOGGER.warning("Background image '%s' was not found.", BACKGROUND_IMAGE_PATH.name)
            return ""
        encoded = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode()
        return f"url('data:image/jpeg;base64,{encoded}') center / cover no-repeat fixed"
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("The background image could not be loaded.")
        return ""


def render_css() -> None:
    overlay = "linear-gradient(160deg, rgba(7, 12, 26, 0.88), rgba(4, 8, 18, 0.94))"
    background = get_background_css_value()
    app_background = f"{overlay}, {background}" if background else overlay

    css = """
    <style>
    :root {
        --navy-900: #04081a;
        --navy-800: #070c1a;
        --panel: rgba(15, 23, 42, 0.68);
        --panel-strong: rgba(11, 18, 38, 0.82);
        --panel-soft: rgba(30, 41, 59, 0.55);
        --border: rgba(148, 197, 255, 0.16);
        --border-strong: rgba(56, 189, 248, 0.32);
        --text: #eaf1fb;
        --text-muted: #9fb2cd;
        --cyan: #22d3ee;
        --blue: #3b82f6;
        --success: #34d399;
        --danger: #fb7185;
        --warning: #fbbf24;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: __APP_BACKGROUND__;
        background-color: var(--navy-900);
        color: var(--text);
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    /* Blend the default Streamlit chrome into the themed background. */
    [data-testid="stHeader"] {
        background: transparent;
        backdrop-filter: none;
    }
    [data-testid="stToolbar"] { right: 1rem; }
    #MainMenu, footer { visibility: hidden; }

    .block-container {
        box-sizing: border-box;
        max-width: 1080px;
        padding: 1.5rem clamp(1rem, 4vw, 2.5rem) 3.5rem;
    }

    h1, h2, h3, h4, h5, p, span, li, label { color: var(--text); }
    a { color: var(--cyan); }

    /* --- Hero header --- */
    .hero {
        box-sizing: border-box;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: clamp(1.4rem, 4vw, 2.4rem);
        margin-bottom: 1.6rem;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
    }
    .eyebrow {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--cyan);
        background: rgba(34, 211, 238, 0.10);
        border: 1px solid var(--border-strong);
        border-radius: 999px;
        padding: 0.32rem 0.8rem;
    }
    .hero h1 {
        margin: 0.85rem 0 0.4rem;
        font-size: clamp(1.7rem, 4.5vw, 2.7rem);
        line-height: 1.15;
        font-weight: 800;
        background: linear-gradient(92deg, #ffffff, #a8d7f5 55%, var(--cyan));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero .subtitle {
        margin: 0;
        color: var(--text-muted);
        font-size: clamp(0.95rem, 1.6vw, 1.08rem);
        max-width: 720px;
    }

    /* --- Panels --- */
    .panel {
        box-sizing: border-box;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.25rem clamp(1rem, 2.5vw, 1.5rem);
        margin-bottom: 1.15rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin: 0 0 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-hint { color: var(--text-muted); font-size: 0.86rem; margin: 0 0 0.4rem; }

    /* --- Result cards --- */
    .result-card {
        box-sizing: border-box;
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-left: 6px solid var(--accent, var(--cyan));
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
    }
    .result-card.fake { --accent: var(--danger); }
    .result-card.real { --accent: var(--success); }
    .result-card .tag {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        font-weight: 700;
    }
    .result-card .verdict {
        font-size: clamp(1.6rem, 5vw, 2.2rem);
        font-weight: 800;
        line-height: 1.1;
        margin: 0.2rem 0 0.5rem;
        color: var(--accent, var(--cyan));
    }
    .result-card .meta { color: var(--text-muted); font-size: 0.92rem; margin: 0; }
    .confidence {
        margin-top: 0.7rem;
    }
    .confidence .bar {
        height: 8px;
        border-radius: 999px;
        background: rgba(148, 197, 255, 0.16);
        overflow: hidden;
    }
    .confidence .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--blue), var(--cyan));
    }
    .confidence .label {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }

    /* --- Evidence cards --- */
    .evidence-card {
        box-sizing: border-box;
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    .evidence-card:hover {
        border-color: var(--border-strong);
        transform: translateY(-1px);
    }
    .card-header {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 0.45rem;
    }
    .card-header strong { font-size: 1.02rem; }
    .evidence-card p { color: var(--text); margin: 0 0 0.5rem; line-height: 1.5; }
    .evidence-card .relevance { display: block; color: var(--text-muted); font-size: 0.82rem; margin-bottom: 0.35rem; }
    .evidence-card a { font-weight: 600; text-decoration: none; }
    .evidence-card a:hover { text-decoration: underline; }

    .pill {
        display: inline-flex;
        padding: 0.24rem 0.62rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .pill.supports { background: rgba(52, 211, 153, 0.16); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.35); }
    .pill.refutes { background: rgba(251, 113, 133, 0.16); color: var(--danger); border: 1px solid rgba(251, 113, 133, 0.35); }
    .pill.insufficient { background: rgba(251, 191, 36, 0.14); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.32); }

    /* --- Streamlit widgets --- */
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stSelectbox"] label,
    .stRadio label, .stCheckbox label { color: var(--text) !important; }

    .stTextArea textarea {
        background: rgba(4, 10, 24, 0.78) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        line-height: 1.55 !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
    }
    .stTextArea textarea::placeholder { color: rgba(159, 178, 205, 0.75); }

    div[data-testid="stSelectbox"] > div, div[data-testid="stTextInput"] > div {
        background: rgba(4, 10, 24, 0.6);
        border-radius: 10px;
    }

    div.stButton > button {
        min-height: 46px;
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.01em;
        border: 1px solid var(--border);
        background: var(--panel-soft);
        color: var(--text);
        transition: all 0.15s ease;
    }
    div.stButton > button:hover {
        border-color: var(--cyan);
        color: #ffffff;
        transform: translateY(-1px);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(94deg, var(--blue), var(--cyan));
        border: none;
        color: #04121f;
        font-weight: 800;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 10px 26px rgba(34, 211, 238, 0.28);
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: var(--panel-strong);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text); }
    section[data-testid="stSidebar"] .stMarkdown code { color: var(--cyan); }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        color: #ffffff;
    }
    .brand-sub { color: var(--text-muted); font-size: 0.85rem; }
    .brand-list { margin: 0.4rem 0 0; padding-left: 1.1rem; color: var(--text-muted); font-size: 0.88rem; line-height: 1.7; }

    /* --- Alerts --- */
    .stAlert, [data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--border); background: var(--panel); }
    [data-testid="stAlertContainer"] { border-radius: 12px; }

    .footnote {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.85rem;
        text-align: center;
    }

    /* --- Responsive (mobile) --- */
    @media (max-width: 760px) {
        .block-container { padding: 1rem 0.85rem 2.5rem; }
        .stApp { background-attachment: scroll; }
        [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
        [data-testid="stColumn"], div[data-testid="column"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        div.stButton > button { width: 100%; white-space: normal; }
        .hero h1 { font-size: 1.7rem; }
        .result-card { padding: 1.1rem; }
        .evidence-card { padding: 0.9rem; }
    }
    </style>
    """
    css = css.replace("__APP_BACKGROUND__", app_background)
    st.markdown(css, unsafe_allow_html=True)


def render_app_header() -> None:
    st.markdown(
        """
        <header class="hero">
            <span class="eyebrow">AI &middot; Cybersecurity &middot; Fact Verification</span>
            <h1>Fake News Detection &amp; Verification System</h1>
            <p class="subtitle">Analyze a news claim, review the machine-learning classification, and inspect the supporting evidence retrieved from public sources.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.markdown('<div class="brand-title">🧠 Verify, don\'t assume</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="brand-sub">Fake News Detection &amp; Verification System</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <ul class="brand-list">
            <li>TF-IDF + Logistic Regression classification</li>
            <li>Wikipedia evidence retrieval</li>
            <li>Optional semantic ranking &amp; NLI comparison</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Developed by Sagar & Team. Educational research tool — no model output is proof of truth or falsity. "
        "Always check primary sources."
    )


render_css()
render_app_header()
render_sidebar()

with st.container():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">News claim analysis</p>'
        '<p class="section-hint">Paste a headline, rumor, or article snippet to classify and verify.</p>',
        unsafe_allow_html=True,
    )

    example = st.selectbox(
        "Load an example claim",
        ["Choose an example…", *EXAMPLE_CLAIMS],
        label_visibility="collapsed",
    )
    example_col, _ = st.columns([1, 3])
    with example_col:
        if st.button("Load example", use_container_width=True) and example != "Choose an example…":
            st.session_state["claim_input"] = example

    st.text_area(
        "Enter a news claim or article",
        key="claim_input",
        max_chars=10000,
        height=220,
        placeholder="For example: “A major vaccine trial showed a 100% success rate in all patients, according to an unnamed report.”",
    )

    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        analyze_clicked = st.button("Analyze News", type="primary", use_container_width=True)
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state["claim_input"] = ""
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if analyze_clicked:
    user_input = st.session_state.get("claim_input", "")
    validation_error = validate_claim_input(user_input)
    if validation_error:
        st.warning(validation_error)
    else:
        try:
            with st.spinner("Classifying the text…"):
                prediction, confidence = predict_claim_with_confidence(user_input)
        except RuntimeError as exc:
            st.error(f"Classification could not be completed: {exc}")
            prediction, confidence = None, None

        if prediction is not None:
            # --- Classification result (model prediction) ---
            st.markdown('<p class="section-title">Model classification</p>', unsafe_allow_html=True)
            st.caption("This is the machine-learning classifier's prediction — not an independent fact-check.")
            tone = "fake" if prediction == "FAKE" else "real"
            verdict_label = "Likely FAKE" if prediction == "FAKE" else "Likely REAL"
            confidence_block = ""
            if confidence is not None:
                pct = confidence * 100
                confidence_block = f"""
                <div class="confidence">
                    <div class="label"><span>Classifier probability (uncalibrated)</span><span>{pct:.1f}%</span></div>
                    <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
                </div>
                """
            st.markdown(
                f"""
                <div class="result-card {tone}">
                    <div class="tag">Classification</div>
                    <div class="verdict">{html.escape(verdict_label)}</div>
                    <p class="meta">Predicted class: <strong>{html.escape(prediction)}</strong>. Scores have not been calibrated for real-world claims.</p>
                    {confidence_block}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Evidence verification (independent from the classifier) ---
            st.markdown(
                '<p class="section-title" style="margin-top:1.4rem;">Evidence verification</p>'
                '<p class="section-hint">Independent of the classifier — retrieved from public Wikipedia references.</p>',
                unsafe_allow_html=True,
            )

            supporting: list[EvidencePage] = []
            contradicting: list[EvidencePage] = []
            unclear: list[EvidencePage] = []
            articles: list[EvidencePage] = []
            ranked_articles: list[EvidencePage] = []
            evidence_error = None
            semantic_error = None
            nli_error = None
            not_relevant = False

            def _bucket(article: EvidencePage, verdict: str, score: float) -> None:
                article["verdict"] = verdict
                article["score"] = score
                if verdict == "Supports":
                    supporting.append(article)
                elif verdict == "Refutes":
                    contradicting.append(article)
                else:
                    unclear.append(article)

            with st.status("Verifying against public evidence…", expanded=True) as status:
                st.write("Searching Wikipedia for references…")
                try:
                    articles = fetch_wikipedia_articles(user_input)
                except WikipediaServiceError as exc:
                    articles = []
                    evidence_error = str(exc)

                if evidence_error:
                    status.update(label="Evidence retrieval failed", state="error")
                elif not articles:
                    status.update(label="No Wikipedia references found", state="complete")
                else:
                    st.write("Ranking sources by relevance…")
                    try:
                        ranked_articles = rank_wikipedia_evidence(user_input, articles)
                    except OptionalModelUnavailable as exc:
                        ranked_articles = articles
                        semantic_error = str(exc)

                    if semantic_error:
                        status.update(label="Sources retrieved (relevance ranking unavailable)", state="complete")
                    else:
                        relevant_articles = [
                            article
                            for article in ranked_articles[:3]
                            if article.get("similarity", 0.0) >= SIMILARITY_THRESHOLD
                        ]
                        if not relevant_articles:
                            not_relevant = True
                            status.update(label="No sufficiently relevant sources to compare", state="complete")
                        else:
                            st.write("Comparing the claim with the most relevant source…")
                            try:
                                top = relevant_articles[0]
                                verdict, score = verify_claim_against_evidence_batch(
                                    user_input, (top["extract"],)
                                )[0]
                                _bucket(top, verdict, score)

                                rest = relevant_articles[1:]
                                if verdict == "Insufficient information" and rest:
                                    st.write("Checking additional sources…")
                                    batch_results = verify_claim_against_evidence_batch(
                                        user_input, tuple(article["extract"] for article in rest)
                                    )
                                    for article, (extra_verdict, extra_score) in zip(rest, batch_results):
                                        _bucket(article, extra_verdict, extra_score)

                                status.update(label="Evidence comparison complete", state="complete", expanded=False)
                            except OptionalModelUnavailable as exc:
                                nli_error = str(exc)
                                status.update(label="Automated comparison unavailable", state="error")

            if evidence_error:
                st.warning(evidence_error)
            if semantic_error:
                st.warning(f"{semantic_error} Sources are shown without a relevance or verdict score.")
            if nli_error:
                st.warning(f"{nli_error} Retrieved sources are available below without an NLI verdict.")
            if not articles and not evidence_error:
                st.info("Wikipedia returned no article results. No external evidence was available for this claim.")
            if not_relevant:
                st.info(
                    "The retrieved references were not relevant enough for automated comparison. "
                    "The claim remains unverified."
                )

            if supporting and contradicting:
                st.warning(
                    "The evidence signals conflict. Do not treat the automated comparison as a definitive verdict."
                )
            elif supporting:
                st.info("The NLI model found possible supporting evidence; this is not proof of truth.")
            elif contradicting:
                st.info("The NLI model found possible contradictory evidence; this is not proof of falsity.")
            elif (
                ranked_articles
                and semantic_error is None
                and not nli_error
                and not not_relevant
                and not supporting
                and not contradicting
            ):
                st.info("No conclusive support or contradiction was identified. The claim remains unverified.")

            if supporting:
                st.markdown("**Possible supporting evidence**")
                for article in supporting:
                    st.markdown(build_card(article, "supports"), unsafe_allow_html=True)
            if contradicting:
                st.markdown("**Possible contradictory evidence**")
                for article in contradicting:
                    st.markdown(build_card(article, "refutes"), unsafe_allow_html=True)
            if unclear:
                st.markdown("**Insufficient or unclear evidence**")
                for article in unclear:
                    st.markdown(build_card(article, "insufficient"), unsafe_allow_html=True)
            if (semantic_error or nli_error) and ranked_articles:
                for article in ranked_articles[:3]:
                    st.markdown(build_card(article, "insufficient"), unsafe_allow_html=True)

            with st.expander("Methodology, limitations & privacy"):
                st.markdown(
                    """
                    The classifier uses TF-IDF features and Logistic Regression trained on the supplied
                    `Fake.csv` and `True.csv` datasets; its output is not an independent fact-check. Wikipedia
                    search results are ranked with sentence embeddings, then relevant introductions are compared
                    by a zero-shot natural-language inference model. Relevance and NLI thresholds are routing
                    heuristics, not calibrated confidence measures.

                    Dataset writing style, time period, duplicate reporting, and topic patterns can make a random
                    holdout score overstate performance on new publishers or current events.

                    **Privacy:** the app does not persist claim text to a project database. When analysis is
                    requested, the claim is sent to Wikipedia's public search API. Successful searches are cached
                    by Streamlit for up to one hour. Do not submit confidential or personal information.
                    """
                )

st.markdown(
    "<div class='footnote'>Fake News Detection &amp; Verification System &middot; Developed by Sagar & Team &middot; "
    "Educational demonstration only. Verify important claims with independent primary sources.</div>",
    unsafe_allow_html=True,
)
