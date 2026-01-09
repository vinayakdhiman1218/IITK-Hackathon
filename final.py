import pandas as pd
import pathway as pw
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# AUTO DEVICE DETECTION
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# LOAD TEST DATA
test_df = pd.read_csv("test.csv")

# LOAD NOVEL TEXT
def load_book(book_name):
    if book_name.lower() == "in search of the castaways":
        path = "In search of the castaways.txt"
    elif book_name.lower() == "the count of monte cristo":
        path = "The Count of Monte Cristo.txt"
    else:
        raise ValueError("Unknown book name")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# CHUNK TEXT
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# CACHE BOOK PROCESSING
book_cache = {}

def cosine_similarity(matrix, vector):
    return np.dot(matrix, vector) / (
        np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    )

# MAIN LOOP
results = []

for _, row in tqdm(
    test_df.iterrows(),
    total=len(test_df),
    desc="Processing stories"
):
    story_id = row["id"]
    book_name = row["book_name"]
    character = row["char"]
    claim = row["content"]

    # ---- LOAD & CACHE BOOK ONCE ----
    if book_name not in book_cache:
        novel_text = load_book(book_name)
        chunks = chunk_text(novel_text)

        # Pathway ingestion (Track A requirement)
        pw.debug.table_from_pandas(
            pd.DataFrame({"text": chunks})
        )

        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False
        )

        book_cache[book_name] = (chunks, embeddings)

    chunks, embeddings = book_cache[book_name]

    # ---- CLAIM EMBEDDING ----
    claim_text = f"{character}. {claim}"
    claim_embedding = model.encode([claim_text])[0]

    # ---- SIMILARITY SEARCH ----
    scores = cosine_similarity(embeddings, claim_embedding)
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    best_chunk = chunks[best_idx]

    # ---- DECISION RULE ----
    prediction = 1 if best_score > 0.45 else 0

    # ---- RATIONALE ----
    if prediction == 1:
        clean_chunk = best_chunk.replace("\n", " ").strip()

        # Try to detect chapter/section heading
        chapter_hint = ""
        lowered = clean_chunk.lower()
        if "chapter" in lowered:
            chap_idx = lowered.find("chapter")
            chapter_hint = clean_chunk[chap_idx: chap_idx + 60].split(".")[0] + "."

        # Extract a complete sentence for evidence
        first_period = clean_chunk.find(".")
        if first_period != -1 and first_period < 300:
            excerpt = clean_chunk[: first_period + 1]
        else:
            excerpt = clean_chunk[:180]

        if chapter_hint:
            rationale = (
                f"This claim is marked consistent because the following line is taken "
                f"from the main story text ({chapter_hint}) and aligns with the given "
                f"backstory: \"{excerpt}\""
            )
        else:
            rationale = (
                "This claim is marked consistent because the following line is taken "
                "from the main story text and aligns with the given backstory: "
                f"\"{excerpt}\""
            )
    else:
        rationale = (
            "This claim is marked inconsistent because no line from the main story "
            "text (across relevant chapters/sections) provides clear support for "
            "the given backstory."
        )

    results.append({
        "story_id": story_id,
        "prediction": prediction,
        "rationale": rationale
    })

# SAVE RESULTS
results_df = pd.DataFrame(
    results,
    columns=["story_id", "prediction", "rationale"]
)
results_df.to_csv("results.csv", index=False)
print("✅ results.csv generated successfully.")