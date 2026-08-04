"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

CONFIGS = {
    "hybrid": {"mode": "hybrid"},          # Config A: BM25 + Semantic → RRF merge → rerank (Task 9 đầy đủ)
    "dense_only": {"mode": "dense_only"},  # Config B: chỉ Semantic Search (Task 5), không BM25, không RRF
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_with_config(query: str, top_k: int = 5, mode: str = "hybrid") -> dict:
    """
    Chạy retrieval (Task 5/6/9) + generation (Task 10) theo 1 trong 2 mode,
    dùng để dựng Config A (Hybrid) vs Config B (Dense-Only) trong compare_configs().

    Args:
        mode: "hybrid" — BM25 (Task 6) + Semantic (Task 5) → RRF merge → rerank (Task 9 đầy đủ)
              "dense_only" — CHỈ semantic_search() (Task 5), bỏ qua BM25 và RRF hoàn toàn

    Logic phần sau (reorder → format_context → gọi LLM) sao chép lại từ
    generate_with_citation() (Task 10) để giữ nguyên prompt/response format giữa 2 config,
    chỉ khác nhau ở nguồn retrieval.
    """
    from src.task10_generation import (
        SYSTEM_PROMPT, TEMPERATURE, TOP_P, LLM_MODEL,
        reorder_for_llm, format_context,
    )

    if mode == "dense_only":
        from src.task5_semantic_search import semantic_search
        chunks = semantic_search(query, top_k=top_k)
        for c in chunks:
            c["source"] = "dense_only"
    else:
        from src.task9_retrieval_pipeline import retrieve
        chunks = retrieve(query, top_k=top_k, use_reranking=True)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openrouter_key and not openai_key:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY / OPENAI_API_KEY trong .env — không gọi được LLM.")

    from openai import OpenAI

    if openrouter_key:
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        model = os.getenv("LLM_MODEL", LLM_MODEL)
    else:
        client = OpenAI(api_key=openai_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return {"answer": answer.strip(), "sources": chunks}


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(generate_fn, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    pip install ragas datasets langchain-openai

    Args:
        generate_fn: callable(question: str) -> {'answer': str, 'sources': list[dict]}
        golden_dataset: list of {'question', 'expected_answer', 'expected_context'}

    Returns:
        pandas.DataFrame — 1 dòng / câu hỏi, cột = 4 metrics + question/answer/contexts.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = generate_fn(item["question"])
        contexts = [c.get("content", "") for c in result.get("sources", [])]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        # RAGAS yêu cầu contexts non-empty — nếu retrieval rỗng (vd out-of-domain
        # đúng nghĩa), dùng placeholder để không crash, điểm context_* sẽ tự thấp.
        eval_data["contexts"].append(contexts or ["(không có context — retrieval rỗng)"])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 configs (xem CONFIGS ở đầu file):
    - Config A "hybrid": BM25 (Task 6) + Semantic (Task 5) → RRF (Task 7) → rerank
    - Config B "dense_only": chỉ Semantic Search (Task 5), không BM25, không RRF

    Returns:
        {config_name: pandas.DataFrame} — mỗi DataFrame là kết quả evaluate_with_ragas
        cho config đó (1 dòng / câu hỏi).
    """
    results = {}
    for config_name, params in CONFIGS.items():
        def generate_fn(question, _params=params):
            return _generate_with_config(question, mode=_params["mode"])

        results[config_name] = evaluate_with_ragas(generate_fn, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def export_results(comparison: dict, config_labels: dict | None = None):
    """
    Export evaluation results ra results.md.

    Args:
        comparison: {config_name: pandas.DataFrame} — output của compare_configs()
        config_labels: {config_name: mô tả hiển thị}, mặc định lấy từ CONFIGS
    """
    config_labels = config_labels or {
        "hybrid": "Config A (Hybrid Search)",
        "dense_only": "Config B (Dense-Only)",
    }

    names = list(comparison.keys())
    averages = {name: {m: float(comparison[name][m].mean()) for m in METRICS} for name in names}

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework sử dụng\n\n> RAGAS (`ragas.evaluate`) — 4 metrics: faithfulness, answer_relevancy, context_recall, context_precision.\n\n---\n\n"

    content += "## Overall Scores\n\n"
    header = "| Metric | " + " | ".join(config_labels.get(n, n) for n in names) + " | Δ |\n"
    sep = "|--------|" + "|".join(["---"] * len(names)) + "|---|\n"
    content += header + sep
    for m in METRICS:
        row_vals = [averages[n][m] for n in names]
        delta = row_vals[0] - row_vals[1] if len(row_vals) >= 2 else 0.0
        content += f"| {m} | " + " | ".join(f"{v:.3f}" for v in row_vals) + f" | {delta:+.3f} |\n"
    avg_row = [sum(averages[n][m] for m in METRICS) / len(METRICS) for n in names]
    avg_delta = avg_row[0] - avg_row[1] if len(avg_row) >= 2 else 0.0
    content += "| **Average** | " + " | ".join(f"**{v:.3f}**" for v in avg_row) + f" | {avg_delta:+.3f} |\n"

    content += "\n---\n\n## A/B Comparison Analysis\n\n"
    for name in names:
        content += f"**{config_labels.get(name, name)}:**\n> {CONFIGS.get(name, {})}\n\n"
    if len(names) >= 2:
        better = names[0] if avg_row[0] >= avg_row[1] else names[1]
        content += f"**Kết luận:** `{config_labels.get(better, better)}` có average score cao hơn ({max(avg_row):.3f} vs {min(avg_row):.3f}). Chênh lệch lớn nhất ở metric: "
        biggest_gap_metric = max(METRICS, key=lambda m: abs(averages[names[0]][m] - averages[names[1]][m]))
        content += f"`{biggest_gap_metric}`.\n\n"

    content += "---\n\n## Worst Performers (Bottom 3)\n\n"
    primary_df = comparison[names[0]].copy()
    primary_df["avg_score"] = primary_df[METRICS].mean(axis=1)
    worst = primary_df.sort_values("avg_score").head(3)
    content += "| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause (điền tay sau khi đọc answer/context) |\n"
    content += "|---|----------|-------------|-----------|--------|-----------|----------------|\n"
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        q = str(row.get("question", ""))[:60].replace("|", "/")
        content += (
            f"| {i} | {q} | {row['faithfulness']:.3f} | {row['answer_relevancy']:.3f} | "
            f"{row['context_recall']:.3f} | {row['context_precision']:.3f} | |\n"
        )

    content += "\n---\n\n## Recommendations\n\n"
    content += "### Cải tiến 1\n**Action:** Xem 3 worst performers ở trên, đọc `answer` + `contexts` thật (không có trong bảng này) để xác định lỗi đến từ retrieval (context sai/thiếu) hay generation (LLM bịa dù có context đúng).\n**Expected impact:** Tăng faithfulness/context_recall cho nhóm câu yếu nhất.\n\n"
    content += "### Cải tiến 2\n**Action:** Nếu Config B (Dense-Only) không thua nhiều so với Config A (Hybrid), cân nhắc bỏ nhánh BM25 + RRF để giảm độ phức tạp/latency; nếu thua rõ rệt, đó là bằng chứng giữ hybrid là đúng.\n**Expected impact:** Quyết định kiến trúc retrieval cho bản production dựa trên số liệu, không phải cảm tính.\n\n"
    content += "### Cải tiến 3\n**Action:** Với câu ngoài domain, kiểm tra context_precision — nếu thấp bất thường nghĩa là hybrid search đang trả \"rác\" thay vì trả rỗng để trigger fallback đúng.\n**Expected impact:** Fallback PageIndex kích hoạt đúng lúc, giảm câu trả lời sai domain.\n\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    return content


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ Thiếu OPENROUTER_API_KEY / OPENAI_API_KEY trong .env.")
        print("  RAGAS cần gọi LLM thật (cho answer generation VÀ cho các metric LLM-based)")
        print("  nên KHÔNG thể chạy evaluation thật khi thiếu key. Dừng ở đây — không tạo")
        print("  results.md với số liệu giả.")
        raise SystemExit(1)

    try:
        comparison = compare_configs(golden_dataset)
    except Exception as exc:
        print(f"\n⚠ Evaluation thất bại: {exc}")
        print("  Kiểm tra: .env có OPENROUTER_API_KEY hợp lệ? chroma_db/ (Task 4) đã build chưa?")
        raise

    export_results(comparison)
    print(f"\n✓ Done! Kết quả đã ghi vào {RESULTS_PATH}")
