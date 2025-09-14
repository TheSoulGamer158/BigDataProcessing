import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import time

# 1. Daten laden (duplizierte Version)
df = pd.read_csv("spam_bigger.csv")  # Spalten: label, text

# 2. Features & Labels
X = df["text"]
y = df["label"]

# 3. Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature-Vektorisierung (TF-IDF)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),       # auch 2-Gramme, um Feature-Space größer zu machen
    max_features=2500      # Dimension beschränken (anpassen!)
)

# 5. Training & Timing
t0 = time.time()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
train_time = time.time() - t0

# 6. Modelltraining
t1 = time.time()
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
train_nb_time = time.time() - t1

# 7. Evaluation
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"Feature-Extraktion: {train_time:.2f} Sekunden")
print(f"Naive Bayes Training: {train_nb_time:.2f} Sekunden")
