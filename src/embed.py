import numpy as np
import sqlite3
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

DB_PATH = Path("db/cricket_drona.db")
EMBED_DIR = Path("db/embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'

def generate_embeddings():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Embed Player Names
    print("Embedding player names...")
    # Get unique players from all relevant tables
    players = set()
    for table in ["matchups_1v1", "bowler_phase_stats", "batter_phase_stats", "recent_form", "pressure_performance"]:
        # Guessing column name
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        name_col = next((c for c in cols if c in ["batter", "bowler", "player"]), None)
        if name_col:
            res = conn.execute(f"SELECT DISTINCT {name_col} FROM {table}").fetchall()
            players.update([r[0] for r in res if r[0]])
    
    player_list = sorted(list(players))
    player_embeddings = model.encode(player_list, show_progress_bar=True)
    
    np.save(EMBED_DIR / "player_names.npy", np.array(player_list))
    np.save(EMBED_DIR / "player_embeddings.npy", player_embeddings)
    
    # 2. Embed Lessons (if any)
    print("Embedding lessons...")
    MEMORY_PATH = Path("db/learned_lessons.json")
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH) as f:
            lessons = json.load(f)
        lesson_texts = [l["lesson"] for l in lessons]
        if lesson_texts:
            lesson_embeddings = model.encode(lesson_texts, show_progress_bar=True)
            np.save(EMBED_DIR / "lesson_embeddings.npy", lesson_embeddings)
            with open(EMBED_DIR / "lessons_raw.json", "w") as f:
                json.dump(lessons, f)
    
    conn.close()
    print("Embeddings saved to db/embeddings/")

def semantic_lookup(query, type="player", top_k=1):
    model = SentenceTransformer(MODEL_NAME)
    if type == "player":
        names = np.load(EMBED_DIR / "player_names.npy")
        embeddings = np.load(EMBED_DIR / "player_embeddings.npy")
        query_emb = model.encode([query])
        similarities = np.dot(embeddings, query_emb.T).flatten()
        best_idx = np.argsort(similarities)[-top_k:][::-1]
        return [(names[i], float(similarities[i])) for i in best_idx]
    return []

if __name__ == "__main__":
    generate_embeddings()
