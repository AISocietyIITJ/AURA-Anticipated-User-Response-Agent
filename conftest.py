import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# app.core.config instantiates Settings() at import time and requires these
# env vars (they have no defaults). Set them before any app module is imported.
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("STT_API_KEY", "test-key")
os.environ.setdefault("PLANNER_CONFIDENCE_THRESHOLD", "0.60")
