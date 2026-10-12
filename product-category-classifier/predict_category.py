
import pickle
import numpy as np
from scipy.sparse import hstack

with open("product_category_model.pkl", "rb") as file:
    model_data = pickle.load(file)

vectorizer = model_data["vectorizer"]
scaler = model_data["scaler"]
model = model_data["model"]

def extract_features(title):
    title_length = len(title)
    word_count = len(title.split())
    contains_number = int(any(char.isdigit() for char in title))
    return np.array([[title_length, word_count, contains_number]])

print("==========================================")
print(" Predikcija kategorije proizvoda")
print("==========================================")
print("Unesite 'q' za izlaz.\n")

while True:
    title = input("Unesite naslov proizvoda: ").strip()

    if title.lower() == "q":
        print("Program je završen.")
        break

    if not title:
        print("Naslov ne može biti prazan.\n")
        continue

    title_tfidf = vectorizer.transform([title])
    numeric_features = extract_features(title)
    numeric_scaled = scaler.transform(numeric_features)

    combined_features = hstack([
        title_tfidf,
        numeric_scaled
    ])

    prediction = model.predict(combined_features)[0]

    print(f"Predviđena kategorija: {prediction}\n")
