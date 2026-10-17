import unittest
from unittest.mock import MagicMock, patch

import requests

import app
from text_processing import normalize_text


class WikipediaRetrievalTests(unittest.TestCase):
    def make_response(self, payload, status_code=200):
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = payload
        response.raise_for_status.side_effect = (
            requests.HTTPError(response=response) if status_code >= 400 else None
        )
        return response

    def fetch_without_cache(self, query):
        return app.fetch_wikipedia_articles.__wrapped__(query)

    @patch("app.requests.Session")
    def test_fetches_articles_and_uses_timeouts(self, session_factory):
        session = session_factory.return_value
        session.get.side_effect = [
            self.make_response({"query": {"search": [{"title": "Example article"}]}}),
            self.make_response(
                {"query": {"pages": [{"title": "Example article", "extract": "A source extract.", "pageid": 5}]}}
            ),
        ]

        pages = self.fetch_without_cache("A complete example claim")

        self.assertEqual(pages[0]["title"], "Example article")
        self.assertEqual(pages[0]["extract"], "A source extract.")
        self.assertEqual(pages[0]["pageid"], 5)
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(session.get.call_args_list[0].kwargs["timeout"], (5, 12))
        self.assertEqual(session.get.call_args_list[1].kwargs["timeout"], (5, 12))

    @patch("app.requests.Session")
    def test_empty_search_is_a_normal_no_result(self, session_factory):
        session_factory.return_value.get.return_value = self.make_response(
            {"query": {"search": []}}
        )
        self.assertEqual(self.fetch_without_cache("A complete example claim"), [])
        self.assertEqual(session_factory.return_value.get.call_count, 1)

    @patch("app.requests.Session")
    def test_timeout_is_reported(self, session_factory):
        session_factory.return_value.get.side_effect = requests.Timeout("timed out")
        with self.assertRaisesRegex(app.WikipediaServiceError, "timed out"):
            self.fetch_without_cache("A complete example claim")

    @patch("app.requests.Session")
    def test_http_error_is_reported(self, session_factory):
        session_factory.return_value.get.return_value = self.make_response(
            {"error": "server"}, status_code=503
        )
        with self.assertRaisesRegex(app.WikipediaServiceError, "HTTP error"):
            self.fetch_without_cache("A complete example claim")

    @patch("app.requests.Session")
    def test_rate_limit_is_reported(self, session_factory):
        session_factory.return_value.get.return_value = self.make_response(
            {"error": "rate limited"}, status_code=429
        )
        with self.assertRaisesRegex(app.WikipediaServiceError, "rate limiting"):
            self.fetch_without_cache("A complete example claim")

    @patch("app.requests.Session")
    def test_malformed_response_is_reported(self, session_factory):
        session_factory.return_value.get.return_value = self.make_response({"unexpected": []})
        with self.assertRaisesRegex(app.WikipediaServiceError, "invalid response"):
            self.fetch_without_cache("A complete example claim")


class ClassificationAndEvidenceTests(unittest.TestCase):
    def test_training_and_inference_normalization(self):
        self.assertEqual(normalize_text("  Health: NEWS!  "), "health news")
        self.assertEqual(normalize_text(None), "")

    def test_input_validation_rejects_empty_short_and_contentless_text(self):
        self.assertIsNotNone(app.validate_claim_input(""))
        self.assertIsNotNone(app.validate_claim_input("tiny"))
        self.assertIsNotNone(app.validate_claim_input("!!!!!!!!!"))
        self.assertIsNone(app.validate_claim_input("A sufficiently detailed claim."))

    @patch("app.load_nli_model")
    def test_nli_uses_actual_claim_and_reports_support(self, load_pipeline):
        pipeline = MagicMock(return_value=[
            {"label": "ENTAILMENT", "score": 0.91},
            {"label": "CONTRADICTION", "score": 0.04},
            {"label": "NEUTRAL", "score": 0.05},
        ])
        load_pipeline.return_value = pipeline

        verdict, score = app.verify_claim_against_evidence(
            "The treaty was signed in 1998.",
            "The treaty was signed in 1998 according to the official record.",
        )

        self.assertEqual(verdict, "Supports")
        self.assertEqual(score, 0.91)
        pair = pipeline.call_args.args[0]
        self.assertEqual(pair["text_pair"], "The treaty was signed in 1998.")

    @patch("app.load_nli_model")
    def test_nli_reports_contradictory_evidence(self, load_pipeline):
        load_pipeline.return_value = MagicMock(return_value=[
            {"label": "CONTRADICTION", "score": 0.87},
            {"label": "ENTAILMENT", "score": 0.06},
            {"label": "NEUTRAL", "score": 0.07},
        ])
        self.assertEqual(
            app.verify_claim_against_evidence("Claim text", "Evidence text"),
            ("Refutes", 0.87),
        )

    @patch("app.load_nli_model")
    def test_low_nli_score_remains_unverified(self, load_pipeline):
        load_pipeline.return_value = MagicMock(return_value=[
            {"label": "ENTAILMENT", "score": 0.42},
            {"label": "CONTRADICTION", "score": 0.36},
            {"label": "NEUTRAL", "score": 0.22},
        ])
        self.assertEqual(
            app.verify_claim_against_evidence("Claim text", "Weak evidence"),
            ("Insufficient information", 0.42),
        )

    def test_evidence_card_escapes_untrusted_html_and_uses_wikipedia_url(self):
        page = {
            "title": "<script>alert(1)</script>",
            "extract": "<img src=x onerror=alert(1)>",
            "similarity": 0.8,
        }
        card = app.build_card(page, "insufficient")
        self.assertNotIn("<script>", card)
        self.assertNotIn("<img", card)
        self.assertIn("https://en.wikipedia.org/wiki/", card)
        self.assertIn("&lt;script&gt;", card)


if __name__ == "__main__":
    unittest.main()
