from __future__ import annotations

import os


os.environ.setdefault("USE_HASH_EMBEDDINGS", "1")
os.environ.setdefault("CHROMA_PERSIST_DIR", ".test_chroma")
os.environ.setdefault("EMAIL_MODE", "mock")
os.environ["LLM_PROVIDER"] = ""
os.environ["GROQ_API_KEY"] = ""
