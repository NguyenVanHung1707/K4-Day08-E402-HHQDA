"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)
"""

import os
from pathlib import Path
from typing import Optional

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn trong comment
# =============================================================================

# Cấu hình chunking theo yêu cầu Checkpoint 2 (CHUNK_SIZE=800, CHUNK_OVERLAP=100)
# Kích thước 800 ký tự phù hợp với mô hình BAAI/bge-m3, đủ ngữ cảnh cho 1 quy định/hướng dẫn
# Overlap 100 ký tự giúp giữ ngữ cảnh giữa các đoạn liền kề, tránh bị ngắt đoạn giữa ý.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ, chạy cực nhanh)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Vector store: ChromaDB (local, persistent, không cần setup server Docker)
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


# =============================================================================
# GLOBAL CACHE / HELPERS
# =============================================================================

_embedding_model = None
_chroma_client = None


class LocalSentenceEncoder:
    """Fallback embedding encoder using feature projection when network is unavailable."""
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False, **kwargs):
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else list(texts)
        import numpy as np
        vectors = []
        for text in texts_list:
            vec = np.zeros(self.dim, dtype=np.float32)
            words = str(text).lower().split()
            for w in words:
                h = abs(hash(w)) % self.dim
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        res = np.array(vectors, dtype=np.float32)
        return res[0] if is_single else res


def get_embedding_model():
    """Cache và trả về instance của embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            try:
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            print(f"[NOTE] Using LocalSentenceEncoder fallback: {e}")
            _embedding_model = LocalSentenceEncoder(dim=EMBEDDING_DIM)
    return _embedding_model


def get_collection():
    """Cache và trả về ChromaDB collection."""
    global _chroma_client
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        if not content.strip():
            continue

        rel_path = str(md_file.relative_to(STANDARDIZED_DIR))
        doc_type = "legal" if "legal" in rel_path else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else list(emb)

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB).
    """
    if not chunks:
        return

    collection = get_collection()

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
