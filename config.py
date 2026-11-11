import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
BASE = Path(__file__).parent
CORPUS_DIR = BASE / "corpus"
INDEX_DIR = BASE / "index"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
