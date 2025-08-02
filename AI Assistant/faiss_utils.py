import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, small, good for semantic search

# Singleton for model loading
def get_model():
    if not hasattr(get_model, "_model"):
        get_model._model = SentenceTransformer(MODEL_NAME)
    return get_model._model

def record_to_text(row):
    # Compose a descriptive string for semantic embedding
    return f"{row.get('BEDROOMS', '')} bedroom {row.get('PROPERTY TYPE', '')} in {row.get('subdistrict_code', '')}, {row.get('SIZE', '')} sq ft, {row.get('BATHROOMS', '')} bathrooms, £{row.get('rent', '')}/month"

def build_faiss_index(csv_path, index_path, id_col="faiss_id"):
    df = pd.read_csv(csv_path)
    model = get_model()
    texts = df.apply(record_to_text, axis=1).astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, index_path)
    df[id_col] = range(len(df))
    df.to_csv(csv_path, index=False)  # Ensure id column is present
    return index, df

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def semantic_search(query_text, index, df, top_k=5):
    model = get_model()
    query_vec = model.encode([query_text])
    D, I = index.search(np.array(query_vec).astype('float32'), top_k)
    return df.iloc[I[0]]
