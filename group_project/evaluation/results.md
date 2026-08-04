# RAG Evaluation Results

## Framework sử dụng

RAGAS (`ragas.evaluate`) — 4 metrics: **faithfulness**, **answer_relevancy**, **context_recall**, **context_precision**.

Code chạy thật nằm ở `group_project/evaluation/eval_pipeline.py` (`evaluate_with_ragas`, `compare_configs`, `export_results`) — đã implement đầy đủ, **đã thử chạy nhưng chưa cho ra điểm số thật** (xem lý do bên dưới). Bảng điểm dưới đây **cố tình để trống thay vì điền số bịa** — điền số RAGAS giả vào đây sẽ làm sai lệch toàn bộ phân tích A/B và đánh giá worst-performer.

---

## Trạng thái chạy thử (04/08/2026)

Đã chạy `python -m group_project.evaluation.eval_pipeline` — kết quả: **dừng sớm, có chủ đích**, vì 2 phụ thuộc bắt buộc chưa sẵn sàng:

| Điều kiện cần | Trạng thái | Ảnh hưởng |
|---|---|---|
| `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY` trong `.env` | ❌ Chưa có (chỉ có `.env.example`) | Không gọi được LLM → không sinh `answer` → RAGAS không chạy được (faithfulness/answer_relevancy cần LLM-as-judge) |
| `chroma_db/` (Task 4 — chunking & indexing) | ❌ Chưa build | `semantic_search()` luôn trả rỗng → nhánh dense của hybrid retrieval không có tín hiệu thật → context_recall/context_precision sẽ bị đánh giá sai lệch (chỉ phản ánh BM25, không phản ánh hybrid thật) |

→ Muốn có số liệu RAGAS thật, cần hoàn thành 2 việc trên trước, sau đó chạy lại đúng lệnh:

```bash
python -m group_project.evaluation.eval_pipeline
```

Script sẽ tự động: load 20 câu trong `golden_dataset.json` → chạy Config A và Config B → tính 4 metrics/câu → ghi đè file `results.md` này bằng bảng điểm thật + worst performers + khuyến nghị (logic export đã viết sẵn trong `export_results()`).

---

## Overall Scores (chưa có số thật — điền tự động khi chạy lại)

| Metric | Config A (hybrid + rerank) | Config B (dense-only, không rerank) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | _(chạy lại để điền)_ | _(chạy lại để điền)_ | |
| Answer Relevance | _(chạy lại để điền)_ | _(chạy lại để điền)_ | |
| Context Recall | _(chạy lại để điền)_ | _(chạy lại để điền)_ | |
| Context Precision | _(chạy lại để điền)_ | _(chạy lại để điền)_ | |
| **Average** | | | |

---

## A/B Comparison — cấu hình đã định nghĩa sẵn trong code

**Config A — `hybrid_rerank`:**
> `retrieve(query, use_reranking=True)` — BM25 (Task 6) + Semantic (Task 5) → merge bằng RRF (Task 7) → rerank lại bằng RRF trên kết quả đã merge.

**Config B — `no_rerank`:**
> `retrieve(query, use_reranking=False)` — BM25 + Semantic → merge bằng RRF, KHÔNG rerank thêm bước 2 (giữ nguyên thứ tự sau merge).

**Kết luận:** _(chưa thể kết luận — cần điểm thật)_

---

## Worst Performers (Bottom 3)

_(bảng này được `export_results()` tự tính từ cột `avg_score` = trung bình 4 metric của Config A, sort tăng dần, lấy 3 câu thấp nhất — sẽ điền tự động khi chạy lại với API key hợp lệ)_

| # | Question | Faithfulness | Relevance | Recall | Precision | Root Cause |
|---|----------|-------------|-----------|--------|-----------|------------|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

---

## Ghi chú về Golden Dataset

`golden_dataset.json` hiện có **20 câu** (yêu cầu tối thiểu 15):
- 17 câu **in-domain**, mỗi câu trace được về đúng file + mục trong `data/standardized/` (đã đọc lại toàn bộ 5 file legal + 5 file news để đảm bảo `expected_answer` khớp nội dung thật, không suy đoán).
- 3 câu **out-of-domain** (thủ tục bằng lái xe, thời tiết, công thức nấu ăn) — dùng để kiểm tra `context_recall`/fallback có hoạt động đúng không khi câu hỏi thực sự không có evidence trong corpus.

Lưu ý: bộ câu hỏi cũ (8 câu, trước khi cập nhật) có 1 câu về "phương thức thanh toán Shopee" liệt kê ShopeePay/Apple Pay/Google Pay/QR Code — **không có căn cứ** trong 10 tài liệu đã crawl (không tài liệu nào liệt kê danh sách phương thức thanh toán đầy đủ như vậy). Câu đó đã bị loại khỏi bộ 20 câu mới để tránh đánh giá sai (câu hỏi đòi hỏi context mà corpus không có sẽ luôn cho context_recall thấp không phải do lỗi hệ thống mà do golden dataset sai).

---

## Recommendations

### Cải tiến 1
**Action:** Cấu hình `OPENROUTER_API_KEY` (model `:free` để tránh tốn phí) và build `chroma_db/` (Task 4), rồi chạy lại `python -m group_project.evaluation.eval_pipeline` để có số liệu thật.
**Expected impact:** Mở khóa toàn bộ phần còn lại của báo cáo này (Overall Scores, A/B Comparison, Worst Performers).

### Cải tiến 2
**Action:** Sau khi có điểm thật, đọc kỹ 3 câu worst-performer — phân biệt lỗi do retrieval (context sai/thiếu, `context_recall` thấp) hay do generation (LLM bịa dù có context đúng, `faithfulness` thấp).
**Expected impact:** Biết nên sửa Task 4/5/6 (retrieval) hay Task 10/prompt (generation) trước.

### Cải tiến 3
**Action:** So sánh `context_precision` giữa 3 câu out-of-domain và 17 câu in-domain. Nếu out-of-domain vẫn có `context_precision` > 0 đáng kể, nghĩa là hybrid search đang trả "rác" thay vì trả rỗng để trigger fallback PageIndex đúng lúc (đã quan sát hiện tượng này trực tiếp khi test Task 9 thủ công — xem log fallback trong `src/task9_retrieval_pipeline.py`).
**Expected impact:** Fallback kích hoạt đúng ngữ cảnh, giảm câu trả lời sai domain.
