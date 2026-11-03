"""Chunking + vector retrieval on ChromaDB.

Embeddings are corpus-fitted TF-IDF vectors so the pipeline runs offline and reproducibly in CI.
Swap `embed()` for a real embedding model in production; the eval suite will tell you if it helps.
"""
import re
import uuid
from dataclasses import dataclass
import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from . import config

def normalize(text: str) -> str:
    """Glue numbers to units ('5 GHz' == '5GHz') and unify spellings ('Wi-Fi' == 'wifi')."""
    t = text.lower().replace("wi-fi", "wifi")
    return re.sub(r"(\d)\s+(ghz|mhz|hz|kw|w|v|mm|gb|mb|rupees)\b", r"\1\2", t)


EXTRA_STOP = {"does", "did", "doing", "long", "take", "many", "much", "tell", "know", "want", "like"}


def stem(t: str) -> str:
    for suf in ("ing", "ed", "s"):
        if t.endswith(suf) and not t.endswith("ss") and len(t) > len(suf) + 3:
            return t[: -len(suf)]
    return t


def analyzer(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", normalize(text))
            if t not in ENGLISH_STOP_WORDS and t not in EXTRA_STOP]
    toks = [stem(t) for t in toks]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


_vec = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)


def embed(texts: list[str]) -> list[list[float]]:
    return _vec.transform(texts).toarray().astype(np.float32).tolist()


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    score: float = 0.0


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Sentence-aware chunking with character overlap."""
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x) + 1 for x in cur) + len(s) > size:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if overlap and len(cur[-1]) <= overlap * 3 else []  # carry last sentence
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class Retriever:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        # unique name per instance: several pipelines (tests, eval) can coexist in one process
        self.col = self.client.create_collection(f"support-kb-{uuid.uuid4().hex[:12]}",
                                                 metadata={"hnsw:space": "cosine"})

    def ingest_dir(self, docs_dir=config.DOCS_DIR) -> int:
        ids, texts, metas = [], [], []
        for path in sorted(docs_dir.glob("*.md")):
            for i, c in enumerate(chunk_text(path.read_text())):
                ids.append(f"{path.stem}-{i}"); texts.append(c); metas.append({"source": path.stem})
        _vec.fit(texts)
        self.col.add(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metas)
        return len(ids)

    def search(self, query: str, k: int = config.TOP_K) -> list[Chunk]:
        res = self.col.query(query_embeddings=embed([query]), n_results=k)
        out = []
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(Chunk(id=i, source=m["source"], text=d, score=round(1 - dist, 4)))
        return out
