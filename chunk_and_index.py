import json
import uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "youtube_playlist_hybrid"
EMBED_MODEL_NAME = "BAAI/bge-m3"

chunks = json.loads(Path("chunks.json").read_text(encoding="utf-8"))
texts = [f"Video: {c['title']}\n{c['text']}" for c in chunks]