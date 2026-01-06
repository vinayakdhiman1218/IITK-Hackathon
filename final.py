import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("train.csv")

def build_text(df):
    return df["book_name"].fillna("") + " " + df["char"].fillna("") + " " + df["content"].fillna("")

X = build_text(train_df)
y = train_df["label"]

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

vec = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english")
Xtrv = vec.fit_transform(Xtr)
Xvav = vec.transform(Xva)

clf = LogisticRegression(max_iter=2000)
clf.fit(Xtrv, ytr)

print("Acc:", accuracy_score(yva, clf.predict(Xvav)))

try:
    test_df = pd.read_csv("test.csv")
    if not test_df.empty:
        Xt = build_text(test_df)
        Xt = vec.transform(Xt)
        pd.DataFrame({"id": test_df["id"], "prediction": clf.predict(Xt)}).to_csv("results.csv", index=False)
        print("results.csv ready")
    else:
        print("test.csv empty")
except:
    print("test.csv empty")
