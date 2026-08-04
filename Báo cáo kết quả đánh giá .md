# Báo Cáo Kết Quả Đánh Giá RAGAS — Hybrid Search vs Dense-Only

**Checkpoint:** CP6 — Báo cáo kết quả đánh giá RAGAS và phân tích hiệu quả Hybrid Search vs Dense-Only
**Ngày:** 04/08/2026
**Framework:** RAGAS (`ragas.evaluate`) — 4 metrics: faithfulness, answer_relevancy, context_recall, context_precision.

---

## 1. Tóm tắt — chưa có điểm số RAGAS thật

Báo cáo này **không chứa số liệu RAGAS bịa**. Pipeline đánh giá (`group_project/evaluation/eval_pipeline.py`) đã implement đầy đủ và đã được chạy thử thật, nhưng dừng có chủ đích vì 2 phụ thuộc bắt buộc chưa sẵn sàng trên máy chạy đánh giá:

| Điều kiện cần | Trạng thái | Vì sao chặn |
|---|---|---|
| `OPENROUTER_API_KEY` hoặc `OPENAI_API_KEY` trong `.env` | ❌ Chưa có | RAGAS cần LLM thật để (a) sinh `answer` qua Task 10 và (b) làm LLM-as-judge cho faithfulness/answer_relevancy |
| `chroma_db/` (Task 4 — chunking & indexing) | ❌ Chưa build | Semantic Search (Task 5) luôn trả rỗng → Config B (Dense-Only) sẽ luôn cho kết quả rỗng/toàn "không xác minh được" — không phản ánh đúng khả năng thật của dense retrieval |

**Kết luận về việc này:** không thể đưa ra "hiệu quả Hybrid vs Dense-Only" bằng số liệu vì Dense-Only hiện tại chưa có gì để so sánh (chưa có vector index). Phần 3 dưới đây trình bày quan sát định tính đã thu thập được trong quá trình test thủ công — có giá trị tham khảo nhưng **không phải là kết quả RAGAS**.

---

## 2. Thiết kế so sánh (đã code sẵn, sẵn sàng chạy khi đủ điều kiện)

| | Config A — Hybrid Search | Config B — Dense-Only |
|---|---|---|
| Nguồn retrieval | BM25 (Task 6) + Semantic (Task 5) | Chỉ Semantic (Task 5) |
| Gộp kết quả | RRF — Reciprocal Rank Fusion (Task 7), k=60 | Không cần (chỉ 1 nguồn) |
| Rerank | Có (RRF trên kết quả đã merge) | Không |
| Fallback PageIndex | Có, khi cosine gốc < threshold (Task 9) | Không dùng trong config này (so sánh thuần Hybrid vs Dense) |
| Hàm gọi | `retrieve(query, use_reranking=True)` | `semantic_search(query)` trực tiếp |

Golden dataset: `group_project/evaluation/golden_dataset.json` — **20 câu** (17 in-domain, trace được về đúng file/mục trong `data/standardized/`; 3 out-of-domain để kiểm tra recall/fallback).

**Lệnh chạy khi đủ điều kiện:**
```bash
python -m group_project.evaluation.eval_pipeline
```
Script sẽ tự ghi đè `group_project/evaluation/results.md` với bảng điểm thật + worst performers + khuyến nghị. Báo cáo này (`Báo cáo kết quả đánh giá .md`) cần được cập nhật thủ công dựa trên `results.md` sau khi chạy xong.

---

## 3. Quan sát định tính sơ bộ (không phải điểm RAGAS)

Trong lúc test thủ công Task 9 (`src/task9_retrieval_pipeline.py`) với 5 câu hỏi (trong domain + ngoài domain), đã quan sát được hành vi sau — **do chroma_db chưa tồn tại nên đây thực chất là quan sát "BM25-only vs không có gì", chưa phải "Hybrid thật vs Dense thật"**:

- Với **câu trong domain** ("quy định trả hàng hoàn tiền shopee"), BM25 trả về đúng đoạn văn bản liên quan (`returns_and_refund_policy.md`) — cho thấy nhánh lexical của hybrid hoạt động tốt khi query dùng đúng từ khóa xuất hiện trong tài liệu.
- Với **câu ngoài domain** ("công thức nấu phở bò", "xin visa du học Canada"), BM25 vẫn trả về 3 kết quả thay vì rỗng — do trùng ngẫu nhiên vài từ đơn lẻ (vd "giấy tờ" ↔ "giấy phép đăng ký"). Đây là **false positive kinh điển của lexical search thuần túy**: không hiểu ngữ nghĩa, chỉ khớp bề mặt.
- Giả thuyết cần verify bằng số liệu RAGAS thật: **Dense-Only (semantic, dùng `BAAI/bge-m3` multilingual) sẽ có context_precision cao hơn Hybrid trên nhóm câu ngoài domain**, vì embedding có thể nhận ra "phở bò"/"visa Canada" không gần về ngữ nghĩa với bất kỳ đoạn nào trong corpus chính sách Shopee — trong khi Hybrid (có nhánh BM25) sẽ kéo điểm precision xuống vì false positive nêu trên.
- Ngược lại, giả thuyết cho **context_recall trên câu trong domain**: Hybrid có khả năng cao hơn Dense-Only, vì với truy vấn có từ khóa chính xác (số điều, tên chính sách cụ thể), BM25 khớp trực tiếp trong khi embedding đôi khi "làm mờ" các con số/tên riêng cụ thể.

Hai giả thuyết trên **cần RAGAS thật để xác nhận hoặc bác bỏ** — hiện tại chỉ là suy luận từ quan sát BM25 một mình, chưa có dense score thật để đối chiếu.

---

## 4. Việc cần làm để hoàn thiện báo cáo này

1. Đăng ký `OPENROUTER_API_KEY` (khuyến nghị dùng model gắn hậu tố `:free` trên OpenRouter để tránh phát sinh phí, lưu ý giới hạn 50 request/ngày/tài khoản) và điền vào `.env` (copy từ `.env.example`).
2. Build `chroma_db/` — hoàn thành Task 4 (`src/task4_chunking_indexing.py`, hiện còn `raise NotImplementedError`) để Dense-Only có index thật để truy vấn.
3. Chạy `python -m group_project.evaluation.eval_pipeline`.
4. Copy bảng điểm + worst performers từ `group_project/evaluation/results.md` sang báo cáo này, thay thế Mục 1 và 3 bằng số liệu thật, xác nhận/bác bỏ 2 giả thuyết ở Mục 3.

---

## 5. Ghi chú

- Golden dataset đã được rà soát lại để loại 1 câu hỏi cũ về "phương thức thanh toán Shopee" (ShopeePay/Apple Pay/Google Pay...) vì **không có căn cứ** trong 10 tài liệu đã crawl — giữ câu đó sẽ làm context_recall bị đánh giá sai do lỗi ở golden dataset, không phải lỗi hệ thống.
- File `group_project/evaluation/results.md` là bản chi tiết kỹ thuật (bảng điểm/câu, worst performers). File này (`Báo cáo kết quả đánh giá .md`) là bản tóm tắt cấp báo cáo/thuyết trình, đặt ở thư mục gốc cùng cấp `LAB_GUIDE.md`.
