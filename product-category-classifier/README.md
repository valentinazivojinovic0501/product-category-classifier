# Predikcija kategorije proizvoda na osnovu naslova

## Opis projekta

Cilj projekta je automatska predikcija kategorije proizvoda na osnovu njegovog naslova.

Korišćen je skup podataka products.csv sa više od 30.000 proizvoda.

## Korišćeni modeli

Testirana su tri algoritma:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC

Najbolji rezultat ostvario je Linear SVC.

## Feature engineering

Korišćene su sledeće karakteristike:

- TF-IDF reprezentacija teksta
- dužina naslova proizvoda
- broj reči u naslovu
- informacija da li naslov sadrži broj

## Rezultati

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Linear SVC | 0.958405 | 0.742408 | 0.954989 |
| Logistic Regression | 0.952422 | 0.737586 | 0.949056 |
| Multinomial Naive Bayes | 0.940883 | 0.726884 | 0.937337 |

### Linear SVC sa feature engineering-om

- Accuracy: 0.959972
- Macro F1: 0.743580
- Weighted F1: 0.956609

## Pokretanje

Za treniranje modela:

python3 train_model.py

Za predikciju kategorije:

python3 predict_category.py

## Struktura projekta

- products.csv
- product_category_model.ipynb
- product_category_model.pkl
- train_model.py
- predict_category.py
- README.md
