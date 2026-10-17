import pandas as pd

# Load datasets
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

# Add a label
true_news['label'] = 'REAL'
fake_news['label'] = 'FAKE'

# Combine datasets
df = pd.concat([true_news, fake_news], ignore_index=True)

# See the first 5 rows
print(df.head())
# Convert text to lowercase
df['text'] = df['text'].str.lower()

# Optional: remove punctuation
df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)

print(df['text'].head())
from sklearn.model_selection import train_test_split

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
news = ["Aliens landed in Mumbai today!"]
news_tfidf = tfidf.transform(news)
prediction = model.predict(news_tfidf)
print(prediction)  # Output: FAKE or REAL