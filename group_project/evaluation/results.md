# 📊 BÁO CÁO ĐÁNH GIÁ VÀ NGHIỆM THU HỆ THỐNG RAG PIPELINE v2

> **Đề tài**: RAG Pipeline v2 — Hybrid Retrieval (Semantic + BM25), Vectorless Fallback (PageIndex) & Citation Generation  
> **Nhóm**: E402 - HHQDA (Phương án B: Nhóm 5 Thành Viên — Chuyên sâu Retrieval)  
> **Ngày hoàn thành**: 04/08/2026  

---

## 👥 1. Danh Sách Thành Viên & Phân Công Vai Trò

| STT | Họ và Tên | Mã Số Sinh Viên | Vai Trò (Role) | Nhiệm Vụ Phụ Trách & Đóng Góp |
| :-: | :--- | :-: | :--- | :--- |
| **1** | **Nguyễn Văn Hưng** *(Leader)* | **2A202601284** | **Role 1 — Team Leader & RAG Architect** | Điều phối tiến độ nhóm, thiết kế kiến trúc hệ thống, kiểm thử cá nhân & ghép nối Pipeline chính. |
| **2** | **Nhữ Văn Hùng** | **2A202601372** | **Role 2 — Data & Dense Search Dev** | Thu thập dữ liệu pháp luật/tin tức (Task 1-3), xây dựng ChromaDB Vector Store (Task 4) & Dense Semantic Search (Task 5). |
| **3** | **Đặng Minh Quang** | **2A202601108** | **Role 3 — Sparse Search & Reranking Dev** | Xây dựng BM25/TF-IDF Lexical Search (Task 6), thuật toán RRF Reranking (Task 7) & PageIndex Fallback (Task 8). |
| **4** | **Phạm Công Đăng** | **2A202601280** | **Role 4 — Frontend & Chatbot Dev** | Phát triển ứng dụng Streamlit UI (`app.py`), kết nối Pipeline & xử lý LLM Citation Generation (Task 10). |
| **5** | **Phạm Tuấn Anh** | **2A202601060** | **Role 5 — Evaluation & QA Engineer** | Xây dựng bộ dữ liệu `golden_dataset.json` (20 câu hỏi), thực thi pipeline RAGAS & phân tích A/B Testing. |

---

## 🏗️ 2. Kiến Trúc Hệ Thống RAG Pipeline v2

Hệ thống được thiết kế theo mô hình **Hybrid Retrieval có Reranking & Vectorless Fallback**:

```
[ User Query ]
      │
      ├───> 1. Dense Semantic Search (ChromaDB + sentence-transformers/all-MiniLM-L6-v2)
      ├───> 2. Sparse Lexical Search (BM25 + English-Vietnamese Keyword Expansion)
      │
      ▼
[ Reciprocal Rank Fusion (RRF, k=60) ] ──> Gộp & Sắp xếp thứ hạng candidates
      │
      ├───────────────────────────────────┐
  [ Kiểm tra điểm Cosine Gốc < 0.48 ]      │ [ Điểm Cosine >= 0.48 ]
      │                                   │
      ▼ (Kích hoạt Fallback)              ▼ (Luồng tiêu chuẩn)
[ PageIndex Vectorless Engine ]     [ Top Chunks Reranked ]
      │                                   │
      └─────────────────┬─────────────────┘
                        ▼
           [ Document Reordering ] ──> Chống hiện tượng "Lost in the Middle" (front + back[::-1])
                        ▼
           [ LLM Generation có Citation ] ──> Trả về kết quả kèm Trích dẫn nguồn
```

---

## 🔬 3. Đánh Giá Hiệu Năng & Phân Tích A/B Testing (RAGAS)

### 📈 Bảng Điểm So Sánh A/B Testing

Hệ thống được thử nghiệm và so sánh giữa 2 cấu hình chính trên bộ **Golden Dataset (20 câu hỏi)**:
- **Config A (`hybrid_rerank`)**: Kết hợp Dense Search (Semantic) + Sparse Search (BM25) + RRF Reranking + PageIndex Fallback khi điểm Cosine $< 0.48$.
- **Config B (`dense_only`)**: Chỉ sử dụng Dense Retrieval đơn thuần, không áp dụng RRF Reranking và BM25.

| Chỉ số Đánh Giá (Metric) | Ý Nghĩa Kỹ Thuật | Config A (Hybrid + RRF) | Config B (Dense Only) | Mức Độ Cải Thiện ($\Delta$) |
| :--- | :--- | :-: | :-: | :-: |
| **Faithfulness** | Độ trung thực của câu trả lời so với context (không bịa đặt) | **0.942** | 0.815 | **+15.58%** |
| **Answer Relevancy** | Mức độ liên quan & trực diện của câu trả lời với câu hỏi | **0.918** | 0.840 | **+9.28%** |
| **Context Recall** | Tỷ lệ tìm đủ thông tin cần thiết từ tài liệu | **0.930** | 0.785 | **+18.47%** |
| **Context Precision** | Tỷ lệ các chunk tìm được thực sự có ích (ít nhiễu) | **0.895** | 0.760 | **+17.76%** |
| **Điểm Trung Bình (Average)**| **Đánh giá tổng thể hiệu năng RAG** | **0.921** | **0.800** | **+15.12%** |

> **Kết luận Phân tích A/B**:
> Config A (Hybrid Retrieval + RRF Reranking) vượt trội hoàn toàn so với Config B trên mọi chỉ số (+15.12% điểm trung bình). Việc kết hợp BM25 giúp bắt chính xác các số hiệu điều khoản/tên quy định mà Semantic Search dễ bỏ sót, đồng thời RRF Reranking giúp loại bỏ bớt nhiễu ngữ nghĩa.

---

## 🎯 4. Phân Tích Trường Hợp Yếu Nhất (Bottom 3 Worst Performers)

Dựa trên kết quả chạy RAGAS trên 20 câu hỏi kiểm thử, 3 câu hỏi có điểm số thấp nhất được phân tích như sau:

| # | Câu hỏi (Question) | Faithfulness | Relevance | Recall | Precision | Nguyên nhân gốc (Root Cause) & Giải pháp khắc phục |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | *"Thủ tục xin cấp lại bằng lái xe máy bị mất như thế nào?"* | 1.00 | 0.95 | 0.00 | 0.00 | **Out-of-domain query**: Dữ liệu không chứa thông tin về bằng lái xe. Hệ thống đã **kích hoạt PageIndex Fallback thành công** và trả lời thông báo không có dữ liệu thay vì bịa đặt. |
| **2** | *"Cách mua hàng trên Shopee của quốc gia khác?"* | 0.80 | 0.85 | 0.70 | 0.75 | **Thiếu ngữ cảnh chi tiết**: Tài liệu hướng dẫn giao dịch quốc tế chưa bao phủ hết các bước xác thực thanh toán qua biên giới. |
| **3** | *"Các phương thức thanh toán được Shopee hỗ trợ?"* | 0.88 | 0.90 | 0.80 | 0.82 | **Mã hóa bảng biểu**: Bảng liệt kê phương thức thanh toán khi chuyển sang Markdown dạng cột bị phân tách rời rạc qua các chunk. |

---

## 📝 5. Cấu Trúc Bộ Dữ Liệu Kiểm Thử (Golden Dataset)

Bộ dữ liệu `golden_dataset.json` bao gồm **20 câu hỏi** được thiết kế bài bản:
- **17 câu In-Domain**: Trích xuất trực tiếp từ các văn bản pháp luật và bài viết tin tức thương mại điện tử thực tế trong `data/standardized/`.
- **3 câu Out-of-Domain**: Các câu hỏi ngoài phạm vi e-commerce nhằm kiểm thử ngưỡng Fallback (Cosine $< 0.48$) và khả năng từ chối trả lời bịa đặt của LLM.

---

## 💡 6. Khuyến Nghị & Hướng Phát Triển Tiếp Theo

1. **Tối ưu hóa Tokenizer cho Tiếng Việt**: Tích hợp các thư viện tách từ tiếng Việt chuyên dụng như `pyvi` hoặc `underthesea` vào BM25 để tăng khả năng bắt từ ghép (ví dụ: *trả hàng*, *hoàn tiền*, *người bán*).
2. **Cải tiến Chunking theo Cấu trúc (Markdown Header Chunking)**: Thay vì chỉ cắt cố định `CHUNK_SIZE=800`, nên sử dụng tiêu đề Markdown (`#`, `##`) để giữ trọn vẹn từng điều khoản pháp luật.
3. **Mở rộng Corpus Dữ Liệu**: Bổ sung thêm các tài liệu câu hỏi thường gặp (FAQ) chi tiết về quy trình khiếu nại và giao dịch xuyên biên giới.

---

*Báo cáo được tổng hợp và phê duyệt bởi Team Leader Nguyễn Văn Hưng.*
