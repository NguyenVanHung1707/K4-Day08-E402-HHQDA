"""
Task 6 — Lexical Search Module (BM25 + TF-IDF).

Mặc định sử dụng BM25 cho `lexical_search()` (interface chính, được test dùng).
Bổ sung thêm `tfidf_search()` (scikit-learn) để so sánh 2 kỹ thuật sparse retrieval
với nhau — dùng cho phần "giải thích cơ chế" +5 bonus.

Cài đặt:
    pip install rank-bm25 scikit-learn

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

TF-IDF (bổ trợ) khác BM25 ở đâu:
    - TF-IDF dùng cosine similarity giữa vector query và vector document, TF tăng
      tuyến tính theo số lần xuất hiện (không có saturation như BM25's k1).
    - Không có length normalization tường minh bằng tham số b — dựa vào L2-normalize
      của vector.
    - Nhìn chung BM25 thường cho kết quả tốt hơn TF-IDF trên văn bản dài vì
      có term-frequency saturation + length normalization, đó là lý do BM25 là
      lựa chọn mặc định cho `lexical_search()`.
"""

import re
from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Corpus được build 1 lần (lazy) từ data/standardized/ và cache lại.
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def _tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản: lowercase + tách theo whitespace, bỏ dấu câu.

    Lưu ý: đây là tokenizer baseline (không xử lý từ ghép tiếng Việt như
    underthesea/pyvi). Đủ dùng cho demo BM25/TF-IDF, nhưng recall cho tiếng
    Việt sẽ tốt hơn nếu dùng word-segmentation thật sự.
    """
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text)
    return text.split()


def _load_corpus() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ và chia nhỏ thành đoạn
    (chunk theo blank-line) để đơn vị tìm kiếm không phải nguyên 1 file dài.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'chunk_index': int}}
    """
    corpus: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return corpus

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"

        # Chunk theo đoạn (paragraph), gộp lại nếu đoạn quá ngắn (<200 ký tự)
        # để tránh chunk rác kiểu tiêu đề 1 dòng.
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buffer = ""
        for para in raw_paragraphs:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para
            if len(buffer) >= 200:
                chunks.append(buffer)
                buffer = ""
        if buffer:
            chunks.append(buffer)
        if not chunks:
            chunks = [text]

        for i, chunk_text in enumerate(chunks):
            corpus.append({
                "content": chunk_text,
                "metadata": {"source": md_file.name, "type": doc_type, "chunk_index": i},
            })

    return corpus


@lru_cache(maxsize=1)
def _get_corpus_and_bm25():
    """Build (lazy, 1 lần) corpus + BM25 index từ data/standardized/."""
    from rank_bm25 import BM25Okapi

    corpus = _load_corpus()
    global CORPUS
    CORPUS = corpus

    if not corpus:
        return corpus, None

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return corpus, bm25


@lru_cache(maxsize=1)
def _get_corpus_and_tfidf():
    """Build (lazy, 1 lần) corpus + TF-IDF vectorizer/matrix từ data/standardized/."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = _load_corpus()
    if not corpus:
        return corpus, None, None

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
    matrix = vectorizer.fit_transform([doc["content"] for doc in corpus])
    return corpus, vectorizer, matrix


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ 1 corpus tuỳ ý (không nhất thiết là corpus mặc định
    load từ data/standardized/). Hữu ích khi muốn build index cho 1 tập chunk
    khác (vd: dùng chung chunks với Task 4).

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance đã fit trên corpus truyền vào.
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25 (sparse retrieval mặc định).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending. Result với score = 0 bị loại bỏ.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus, bm25 = _get_corpus_and_bm25()
    if bm25 is None:
        return []

    import numpy as np

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng TF-IDF + cosine similarity (bổ trợ, so sánh với BM25).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted desc.
    """
    if top_k <= 0 or not query.strip():
        return []

    corpus, vectorizer, matrix = _get_corpus_and_tfidf()
    if vectorizer is None:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]

    import numpy as np

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    query = "phương thức thanh toán shopee"

    print("=" * 60)
    print(f"Query: {query!r}")
    print("=" * 60)

    print("\n--- BM25 (lexical_search) ---")
    for r in lexical_search(query, top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")

    print("\n--- TF-IDF (tfidf_search) ---")
    for r in tfidf_search(query, top_k=5):
        print(f"[{r['score']:.3f}] ({r['metadata']['source']}) {r['content'][:100]}...")
