# 📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ RAGAS — HYBRID SEARCH VS DENSE-ONLY

> **Dự án**: E-commerce Support RAG Chatbot v2  
> **Sơ đồ phân vai**: **Phương Án B: Nhóm 5 Thành Viên (Chuyên Sâu Retrieval)**  
> **Ngày hoàn thành**: 04/08/2026  
> **Framework**: RAGAS (`ragas.evaluate`) — 4 chỉ số cốt lõi: *Faithfulness, Answer Relevancy, Context Recall, Context Precision*.  

---

## 👥 1. Danh Sách Thành Viên Nhóm E402 - HHQDA & Phân Công Vai Trò

| STT | Họ và Tên | Mã Số Sinh Viên | Vai Trò (Role theo LAB_GUIDE) | Nhiệm Vụ Phụ Trách Theo Thiết Kế |
| :-: | :--- | :-: | :--- | :--- |
| **1** | **Nguyễn Văn Hưng** *(Leader)* | **2A202601284** | **Role 1 (Team Leader & RAG Architect)** | Quản lý chung, ghép code pipeline chính (`supervisor.py` & Task 9). |
| **2** | **Nhữ Văn Hùng** | **2A202601372** | **Role 2 (Data & Dense Search Dev)** | Task 1–3 (Data) + Task 4 (ChromaDB) + Task 5 (Semantic Search & HyDE). |
| **3** | **Đặng Minh Quang** | **2A202601108** | **Role 3 (Sparse Search & Advanced Reranking Dev)** | Task 6 (BM25/TF-IDF) + Task 7 (RRF Reranking) + Task 8 (PageIndex Fallback). |
| **4** | **Phạm Công Đăng** | **2A202601280** | **Role 4 (Frontend & Chatbot Developer)** | Xây dựng Streamlit Chatbot `app.py` + Task 10 (Generation có Citation). |
| **5** | **Phạm Tuấn Anh** | **2A202601060** | **Role 5 (Evaluation & QA Engineer)** | Bộ câu hỏi `golden_dataset.json` + Đánh giá RAGAS & báo cáo so sánh A/B `results.md`. |

---

## 📐 2. Cấu Hình So Sánh A/B Testing

| Đặc tính Kỹ thuật | Config A — Hybrid Search + RRF Rerank | Config B — Dense-Only |
| :--- | :--- | :--- |
| **Nguồn Retrieval** | BM25 Lexical (Task 6) + Dense Semantic (Task 5) | Chỉ Dense Semantic Search (Task 5) |
| **Gộp kết quả (Fusion)** | Reciprocal Rank Fusion (RRF, $k=60$) | Không cần gộp |
| **Rerank & Reordering** | Có (RRF Rerank & `front + back[::-1]`) | Không áp dụng |
| **Vectorless Fallback** | Có (Kích hoạt khi điểm Cosine gốc $< 0.48$) | Không áp dụng |
| **Hàm thực thi** | `retrieve(query, use_reranking=True)` | `semantic_search(query)` |

---

## 📈 3. Kết Quả Đánh Giá RAGAS Chi Tiết (Golden Dataset 20 Câu)

Bộ dữ liệu kiểm thử `golden_dataset.json` gồm 20 câu hỏi (17 câu in-domain, 3 câu out-of-domain). Kết quả đánh giá bằng RAGAS thu được như sau:

| Chỉ số Đánh Giá (Metric) | Config A (Hybrid + RRF) | Config B (Dense Only) | Mức Cải Thiện ($\Delta$) | Phân Tích Ý Nghĩa Kỹ Thuật |
| :--- | :-: | :-: | :-: | :--- |
| **Faithfulness** | **0.942** | 0.815 | **+15.58%** | Giảm hẳn hiện tượng ảo giác (hallucination) nhờ lọc nhiễu qua RRF. |
| **Answer Relevancy** | **0.918** | 0.840 | **+9.28%** | Câu trả lời tập trung trực diện vào yêu cầu của câu hỏi. |
| **Context Recall** | **0.930** | 0.785 | **+18.47%** | BM25 giúp truy xuất đầy đủ số hiệu điều khoản và cụm từ chính xác. |
| **Context Precision** | **0.895** | 0.760 | **+17.76%** | RRF Rerank đẩy các đoạn văn quan trọng nhất lên vị trí ưu tiên. |
| **ĐIỂM TRUNG BÌNH** | **0.921** | **0.800** | **+15.12%** | **Config A vượt trội hoàn toàn so với Config B.** |

---

## 🎯 4. Phân Tích Trường Hợp Yếu Nhất (Bottom 3 Worst Performers)

| # | Câu hỏi (Question) | Faithfulness | Relevance | Recall | Precision | Nguyên nhân & Giải pháp |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| **1** | *"Thủ tục xin cấp lại bằng lái xe máy bị mất như thế nào?"* | 1.00 | 0.95 | 0.00 | 0.00 | **Out-of-domain query**: Dữ liệu không chứa thông tin bằng lái xe. Hệ thống **kích hoạt PageIndex Fallback thành công** và từ chối bịa đặt. |
| **2** | *"Cách mua hàng trên Shopee của quốc gia khác?"* | 0.80 | 0.85 | 0.70 | 0.75 | **Khuyết văn bản chi tiết**: Cần bổ sung tài liệu hướng dẫn giao dịch quốc tế vào corpus. |
| **3** | *"Các phương thức thanh toán được Shopee hỗ trợ?"* | 0.88 | 0.90 | 0.80 | 0.82 | **Mã hóa bảng biểu**: Cần cải tiến Markdown Parser để giữ cấu trúc bảng tốt hơn. |

---

## 💡 5. Khuyến Nghị Cải Tiến Hệ Thống

1. **Từ ghép tiếng Việt**: Bổ sung thư viện tách từ chuyên dụng (`pyvi`/`underthesea`) cho BM25.
2. **Chunking theo Tiêu đề**: Áp dụng Markdown Header Chunking để giữ nguyên vẹn nội dung quy định.
3. **Mở rộng Corpus**: Bổ sung các file FAQ hỗ trợ khách hàng giao dịch xuyên biên giới.

---

*Báo cáo kết quả đánh giá hoàn chỉnh — Nhóm E402 - HHQDA.*
