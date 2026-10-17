from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from text_processing import normalize_text


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    true_news = pd.read_csv(PROJECT_DIR / "True.csv")
    fake_news = pd.read_csv(PROJECT_DIR / "Fake.csv")

    true_news["label"] = "REAL"
    fake_news["label"] = "FAKE"

    df = pd.concat([true_news, fake_news], ignore_index=True)
    df["text"] = df["text"].map(normalize_text)
    empty_count = int(df["text"].eq("").sum())
    exact_duplicate_count = int(df.duplicated(subset=["text"]).sum())
    df = df[df["text"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text"]).copy()

    class_counts = df["label"].value_counts()
    print("Class counts:")
    print(class_counts)
    print(f"Empty rows removed: {empty_count}")
    print(f"Exact normalized duplicates removed: {exact_duplicate_count}")

    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        ngram_range=(1, 2),
        max_features=20000,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro precision: {precision_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print(f"Macro F1-score: {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}")
    print("Confusion matrix:")
    print(conf_matrix)
    print("\nClassification report:")
    report = classification_report(y_test, predictions, digits=4, output_dict=True)
    print(classification_report(y_test, predictions, digits=4))

    artifact = {
        "model": model,
        "tfidf": tfidf,
        "metadata": {
            "training_version": "1.0",
            "text_preprocessing": "lowercase + remove punctuation + whitespace normalization",
            "class_counts": class_counts.to_dict(),
            "rows_after_cleaning": int(len(df)),
            "test_size": 0.2,
            "random_state": 42,
            "scikit_learn_version": sklearn.__version__,
            "evaluation": {
                "accuracy": float(accuracy),
                "classification_report": report,
                "confusion_matrix_labels": ["FAKE", "REAL"],
                "confusion_matrix": conf_matrix.tolist(),
            },
        },
    }
    joblib.dump(artifact, PROJECT_DIR / "fake_news_model.joblib")
    print("Saved fake_news_model.joblib")


if __name__ == "__main__":
    main()
