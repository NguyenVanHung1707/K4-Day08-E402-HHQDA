"""Task 5 — Semantic Search trên ChromaDB, hỗ trợ HyDE tùy chọn."""

import os
from functools import lru_cache
from pathlib import Path

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "ecommerce_support_docs"
EMBEDDING_MODEL = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Dùng đúng embedding model đã cấu hình ở Task 4."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _get_collection():
    """Mở collection do Task 4 tạo trong persistent ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME)


def generate_hypothetical_document(query: str) -> str:
    """Sinh hypothetical document; thiếu API key thì dùng query gốc an toàn."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return query

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            messages=[{
                "role": "user",
                "content": (
                    "Viết một đoạn tài liệu ngắn có khả năng trả lời câu hỏi sau. "
                    "Chỉ viết nội dung giả định, không thêm trích dẫn: " + query
                ),
            }],
            temperature=0.0,
        )
        passage = response.choices[0].message.content
        return passage.strip() if passage else query
    except Exception:
        return query


def semantic_search(
    query: str,
    top_k: int = 10,
    use_hyde: bool = False,
) -> list[dict]:
    """Tìm kiếm dense retrieval và trả kết quả theo cosine similarity giảm dần."""
    if top_k <= 0 or not query.strip():
        return []

    # Vector store thuộc Task 4 có thể chưa được Role 2 tạo xong.
    if not CHROMA_DIR.exists():
        return []

    try:
        collection = _get_collection()
    except (ImportError, ValueError):
        return []

    search_text = generate_hypothetical_document(query) if use_hyde else query
    model = _get_embedding_model()
    query_vector = model.encode(search_text).tolist()

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    output = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append({
            "content": document,
            "score": round(similarity, 4),
            "metadata": metadata or {},
        })

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")