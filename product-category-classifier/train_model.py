
import pandas as pd
import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

print("Učitavanje podataka...")

df = pd.read_csv("products.csv")
df.columns = df.columns.str.strip()

df = df.dropna(
    subset=["Product Title", "Category Label"]
).copy()

df["Product Title"] = df["Product Title"].astype(str).str.strip()
df["Category Label"] = df["Category Label"].astype(str).str.strip()

df["title_length"] = df["Product Title"].str.len()
df["word_count"] = df["Product Title"].str.split().str.len()
df["contains_number"] = df["Product Title"].str.contains(
    r"\d", regex=True
)

X = df["Product Title"]
y = df["Category Label"]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_tfidf = vectorizer.fit_transform(X)

numeric_features = df[[
    "title_length",
    "word_count",
    "contains_number"
]].copy()

numeric_features["contains_number"] = (
    numeric_features["contains_number"].astype(int)
)

scaler = MaxAbsScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

X_combined = hstack([
    X_tfidf,
    numeric_scaled
])

model = LinearSVC(random_state=42)
model.fit(X_combined, y)

model_data = {
    "vectorizer": vectorizer,
    "scaler": scaler,
    "model": model
}

with open("product_category_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("==========================================")
print("Model je uspešno istreniran!")
print("Sačuvan je kao product_category_model.pkl")
print("Broj proizvoda:", len(df))
print("Broj kategorija:", y.nunique())
print("==========================================")
