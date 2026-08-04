"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


REGISTRY_PATH = Path(__file__).parent.parent / '.pageindex_documents.json'


def _client():
    if not PAGEINDEX_API_KEY:
        raise RuntimeError('PAGEINDEX_API_KEY is not configured in .env')
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _read_registry() -> list[dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _markdown_pdf(markdown_path: Path, output_dir: str) -> Path:
    from fpdf import FPDF
    font = Path('C:/Windows/Fonts/arial.ttf')
    if not font.exists():
        font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if not font.exists():
        raise RuntimeError('A Unicode TrueType font is required')
    target = Path(output_dir) / (markdown_path.stem + '.pdf')
    pdf = FPDF()
    pdf.add_font('Unicode', fname=str(font))
    pdf.set_font('Unicode', size=10)
    pdf.add_page()
    pdf.multi_cell(0, 5, markdown_path.read_text(encoding='utf-8'))
    pdf.output(str(target))
    return target


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _client()
    registry = _read_registry()
    known = {item.get('path') for item in registry}
    files = sorted(STANDARDIZED_DIR.rglob('*.md'))
    if not files:
        raise FileNotFoundError('No standardized Markdown files found for PageIndex upload')
    temp_dir = tempfile.TemporaryDirectory(prefix='.pageindex-', dir=Path(__file__).parent.parent)
    for pdf_file in files:
        relative = pdf_file.relative_to(STANDARDIZED_DIR).as_posix()
        if relative in known:
            continue
        response = client.submit_document(str(_markdown_pdf(pdf_file, temp_dir.name)))
        doc_id = response.get('doc_id') or response.get('id')
        if not doc_id:
            raise RuntimeError(f'PageIndex did not return a doc_id for {relative}')
        registry.append({'path': relative, 'name': pdf_file.name, 'doc_id': str(doc_id)})
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
    return registry

    # TODO: Implement upload
    #
    # Tham khảo: https://github.com/VectifyAI/PageIndex
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    #
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     # Lưu ý: PageIndex nhận PDF, không nhận .md trực tiếp — có thể cần
    #     # convert markdown sang PDF đơn giản bằng fpdf2 trước khi upload.
    #     resp = client.submit_document(str(pdf_path))
    #     doc_id = resp.get("doc_id") or resp.get("id")
    #     print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")
    raise NotImplementedError("Implement upload_documents")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    client = _client()
    documents = _read_registry()
    if not documents:
        listing = client.list_documents(limit=100)
        docs = listing.get('documents', []) if isinstance(listing, dict) else []
        documents = [{'doc_id': str(d.get('doc_id') or d.get('id')), 'name': str(d.get('name') or '')} for d in docs if d.get('doc_id') or d.get('id')]
    found = []
    for document in documents:
        submitted = client.submit_query(doc_id=document['doc_id'], query=query)
        retrieval_id = submitted.get('retrieval_id') or submitted.get('id')
        if not retrieval_id:
            raise RuntimeError('PageIndex did not return a retrieval_id')
        deadline = time.monotonic() + 60
        while True:
            response = client.get_retrieval(str(retrieval_id))
            status = str(response.get('status', '')).lower()
            if 'retrieved_nodes' in response or status in {'completed', 'complete', 'success'}:
                break
            if status in {'failed', 'error', 'cancelled'}:
                raise RuntimeError(f'PageIndex retrieval failed: {response}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'PageIndex retrieval {retrieval_id} timed out')
            time.sleep(1)
        for node in response.get('retrieved_nodes', []):
            for group in node.get('relevant_contents', []):
                for item in group:
                    content = item.get('relevant_content', '')
                    if content:
                        found.append({'content': content, 'metadata': {'section': item.get('section_title', ''), 'document': document.get('name', ''), 'doc_id': document['doc_id']}, 'source': 'pageindex'})
    results = found[:top_k]
    for rank, result in enumerate(results, 1):
        result['score'] = 1.0 / rank
    return results

    # TODO: Implement PageIndex query
    #
    # from pageindex.client import PageIndexClient
    #
    # client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    # resp = client.submit_query(doc_id=doc_id, query=query)
    # retrieval_id = resp.get("retrieval_id") or resp.get("id")
    #
    # # Poll cho đến khi status == "completed"
    # retrieval = client.get_retrieval(retrieval_id)
    #
    # # Parse retrieval["retrieved_nodes"] — mỗi node có "relevant_contents"
    # results = []
    # for node in retrieval.get("retrieved_nodes", [])[:2]:
    #     for group in node.get("relevant_contents", []):
    #         for item in group:
    #             results.append({
    #                 "content": item.get("relevant_content", ""),
    #                 "score": ...,  # PageIndex không trả score trực tiếp — tự gán theo rank
    #                 "metadata": {"section": item.get("section_title")},
    #                 "source": "pageindex",
    #             })
    # return results[:top_k]
    raise NotImplementedError("Implement pageindex_search")


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
