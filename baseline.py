# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ===============================
# STEP 2: LOAD TRAIN DATA
# ===============================
train_df = pd.read_csv("train.csv")

# Input text (backstory / content)
X = train_df["content"].fillna("")

# Output label (0 or 1)
y = train_df["label"]


# ===============================
# STEP 3: SPLIT TRAIN / VALIDATION
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# STEP 4: TEXT → NUMBERS (TF-IDF)
# ===============================
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)


# ===============================
# STEP 5: TRAIN MODEL
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


# ===============================
# STEP 6: VALIDATION CHECK
# ===============================
val_preds = model.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_preds)

print("Validation Accuracy:", val_acc)


# ===============================
# STEP 7: LOAD TEST DATA
# ===============================
test_df = pd.read_csv("test.csv")

X_test = test_df["content"].fillna("")
X_test_vec = vectorizer.transform(X_test)


# ===============================
# STEP 8: PREDICT ON TEST DATA
# ===============================
test_preds = model.predict(X_test_vec)


# ===============================
# STEP 9: SAVE SUBMISSION FILE
# ===============================
submission = pd.DataFrame({
    "id": test_df["id"],
    "prediction": test_preds
})

submission.to_csv("results.csv", index=False)

print("results.csv generated successfully")
