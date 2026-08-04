# 🎤 KỊCH BẢN THUYẾT TRÌNH DEMO LIVE & Q&A CHEATSHEET (CHECKPOINT 6)

> **Đề tài**: RAG Pipeline v2 — Hybrid Search, RRF Reranking, Vectorless Fallback & Citation Generation  
> **Nhóm**: E402 - HHQDA | **Phương án B (Nhóm 5 Thành Viên)**  
> **Repository GitHub**: [https://github.com/NguyenVanHung1707/K4-Day08-E402-HHQDA](https://github.com/NguyenVanHung1707/K4-Day08-E402-HHQDA)  

---

## 👥 THÀNH PHẦN BÁO CÁO & PHÂN CÔNG VAI TRÒ (CP6)

1. 👑 **Role 1 (Team Leader & RAG Architect) — Nguyễn Văn Hưng (2A202601284)**: Thuyết trình mở đầu, tổng quan kiến trúc hệ thống RAG Pipeline (3-4 phút).
2. 🎨 **Role 4 (Frontend & Chatbot Dev) — Phạm Công Đăng (2A202601280)**: Thao tác Live Demo Chatbot UI trên ứng dụng Streamlit (2-3 phút).
3. 📊 **Role 5 (Evaluation & QA Engineer) — Phạm Tuấn Anh (2A202601060)**: Trình bày kết quả đánh giá RAGAS & Phân tích so sánh A/B (1-2 phút).
4. ⚙️ **Role 2 & 3 (Backend & Retrieval Specialist) — Nhữ Văn Hùng (2A202601372) & Đặng Minh Quang (2A202601108)**: Phụ trách phần Q&A trả lời câu hỏi kỹ thuật từ Giảng viên / Coach.

---

## 📜 PHẦN 1: KỊCH BẢN THUYẾT TRÌNH DEMO LIVE (PRESENTATION SCRIPT)

### 🎙️ 1. Mở Đầu & Kiến Trúc Hệ Thống (Presenter: Nguyễn Văn Hưng - Role 1)

> *"Kính chào Thầy/Cô và các bạn! Em là Nguyễn Văn Hưng, Trưởng nhóm E402 - HHQDA. Hôm nay nhóm em xin đại diện trình bày sản phẩm RAG Pipeline v2: Hệ thống Trợ lý Hỏi đáp Chính sách Thương mại Điện tử và Hỗ trợ Khách hàng.*
>
> *Trong bài toán hỏi đáp quy định e-commerce, các hệ thống RAG truyền thống dựa trên Dense Retrieval (Vector Store) thường gặp phải 2 vấn đề lớn:*
> 1. *Bỏ sót các từ khóa số hiệu điều khoản hoặc tên chính xác (do hiện tượng mờ ngữ nghĩa của vector).*
> 2. *Suy giảm độ chú ý ở giữa văn bản (Lost in the Middle) và bịa đặt thông tin khi câu hỏi nằm ngoài cơ sở dữ liệu.*
>
> *Để giải quyết triệt để vấn đề này, nhóm em thiết kế kiến trúc RAG Pipeline v2 gồm 4 thành phần cốt lõi:*
> - **Hybrid Retrieval**: Kết hợp Dense Search (Vector ChromaDB) và Sparse Search (BM25 Lexical Search).
> - **Reciprocal Rank Fusion (RRF, $k=60$)**: Gộp thứ hạng đa nguồn mà không bị phụ thuộc vào thang điểm tuyệt đối.
> - **Vectorless Fallback (PageIndex Engine)**: Kích hoạt tự động khi điểm Cosine Similarity gốc $< 0.48$.
> - **Document Reordering & Citation Generation**: Sắp xếp lại chunk theo dạng `front + back[::-1]` và yêu cầu LLM trích dẫn nguồn chuẩn xác.*
>
> *Sau đây em xin nhường phần thao tác Live Demo ứng dụng cho bạn Phạm Công Đăng."*

---

### 💻 2. Thao Tác Live Demo Trên Streamlit UI (Presenter: Phạm Công Đăng - Role 4)

> *(Thao tác trực tiếp trên màn hình chiếu `streamlit run app.py`)*
>
> *"Dạ chào Thầy Cô, đây là giao diện Chatbot Streamlit của nhóm em.*
>
> **Demo Scenario 1 — Câu hỏi In-Domain (Khớp ngữ nghĩa & Từ khóa)**:
> - *Em bấm vào câu hỏi gợi ý: **'Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?'***
> - *Hệ thống lập tức truy vấn Hybrid Search, gộp kết quả qua RRF Reranking và hiển thị câu trả lời có Trích dẫn nguồn ngay lập tức.*
> - *Mở vùng **'📚 Nguồn tham khảo'**: Hiển thị rõ danh sách các chunk tài liệu gốc kèm chỉ số `score` và `doc_type`.*
>
> **Demo Scenario 2 — Câu hỏi Out-of-Domain (Kiểm thử Fallback)**:
> - *Em nhập câu hỏi ngoài lĩnh vực: **'Thủ tục xin cấp lại bằng lái xe máy bị mất như thế nào?'***
> - *Hệ thống nhận diện điểm Cosine Similarity gốc $< 0.48$, tự động kích hoạt bẫy điều kiện Fallback sang PageIndex Engine và đưa ra câu trả lời từ chối bịa đặt một cách an toàn.*
>
> *Tiếp theo xin mời bạn Phạm Tuấn Anh trình bày báo cáo đánh giá hiệu năng RAGAS."*

---

### 📊 3. Báo Cáo Kết Quả Đánh Giá RAGAS & A/B Testing (Presenter: Phạm Tuấn Anh - Role 5)

> *"Em xin báo cáo kết quả đánh giá hệ thống trên bộ dữ liệu chuẩn **Golden Dataset gồm 20 câu hỏi** (17 câu in-domain, 3 câu out-of-domain).*
>
> *Nhóm đã thực hiện A/B Testing so sánh giữa **Config A (Hybrid + RRF)** và **Config B (Dense Only)**:*
> - **Faithfulness (Độ trung thực)**: Config A đạt **0.942** vs Config B đạt 0.815 (**tăng +15.58%**).
> - **Context Recall (Khả năng bao phủ)**: Config A đạt **0.930** vs Config B đạt 0.785 (**tăng +18.47%**).
> - **Điểm Trung Bình Tất Cả Metrics**: Config A đạt **0.921** so với Config B đạt **0.800** (hiệu năng tổng thể **tăng +15.12%**).
>
> *Kết quả khẳng định việc kết hợp BM25 và RRF Reranking giúp hệ thống tìm kiếm chính xác và giảm thiểu hiện tượng hallucination vượt trội.*
>
> *Nhóm em xin cảm ơn Thầy Cô và sẵn sàng nhận câu hỏi Q&A!"*

---

## ❓ PHẦN 2: BỘ CÂU HỎI Q&A KỸ THUẬT (Q&A CHEATSHEET CHO ROLE 2 & ROLE 3)

*(Dành cho Nhữ Văn Hùng & Đặng Minh Quang trả lời khi Giảng viên/Coach đặt câu hỏi)*

### ❓ Câu 1: "Tại sao nhóm lại dùng RRF (Reciprocal Rank Fusion) mà không dùng Cross-Encoder Reranker?"
> **Trả lời (Role 3 - Đặng Minh Quang)**:  
> *"Dạ thưa Thầy/Cô, Cross-Encoder Reranker đòi hỏi mô hình nặng và chi phí tính toán cao. Trong khi đó, RRF là phương pháp gộp thứ hạng phi tham số (non-parametric) áp dụng công thức $RRF(d) = \sum \frac{1}{60 + r(d)}$. RRF có 3 ưu điểm vượt trội: (1) Không phụ thuộc vào thang điểm khác biệt giữa Cosine Similarity và BM25, (2) Tốc độ tính toán miligiây cực nhanh, (3) Không tốn chi phí gọi API key ngoài."*

### ❓ Câu 2: "Tại sao lại cần bẫy điều kiện Fallback điểm Cosine $< 0.48$? Nếu dùng điểm RRF để so ngưỡng được không?"
> **Trả lời (Role 2 - Nhữ Văn Hùng)**:  
> *"Dạ thưa Thầy/Cô, điểm RRF sau khi fuse CHỈ phụ thuộc vào vị trí thứ hạng. Top 1 của RRF luôn xấp xỉ $\frac{1}{60+1} \approx 0.0164$ bất kể câu hỏi có liên quan hay là câu rác ngoài domain. Nếu dùng điểm RRF làm ngưỡng fallback thì mọi câu hỏi đều bị lọt qua bẫy. Vì vậy, nhóm em giữ lại điểm **Cosine Similarity GỐC** từ mô hình Dense Search (`dense_results[0]['score']`) làm căn cứ kích hoạt Fallback sang PageIndex Engine khi điểm Cosine $< 0.48$."*

### ❓ Câu 3: "Kỹ thuật Document Reordering chống 'Lost in the Middle' hoạt động như thế nào?"
> **Trả lời (Role 1 - Nguyễn Văn Hưng)**:  
> *"Dạ theo nghiên cứu của Liu et al., các LLM ghi nhớ thông tin ở ĐẦU và CUỐI prompt tốt hơn ở GIỮA. Nhóm em sắp xếp các chunks có score giảm dần $[1, 2, 3, 4, 5]$ thành $[1, 3, 5, 4, 2]$ bằng công thức Python `front + back[::-1]`. Việc này đẩy chunk quan trọng nhất (Top 1) lên ĐẦU và chunk quan trọng thứ nhì (Top 2) xuống CUỐI prompt, giúp LLM chú ý tối đa tới các thông tin quan trọng nhất."*

---

## 🎨 PHẦN 3: DÀN Ý SLIDE THUYẾT TRÌNH (SLIDE PRESENTATION OUTLINE)

### Slide 1: Trang Tiêu Đề
- **Tên dự án**: RAG PIPELINE v2 — E-COMMERCE SUPPORT CHATBOT
- **Nhóm**: E402 - HHQDA | Phương án B (5 Thành viên)
- **Thành viên**: Nguyễn Văn Hưng, Nhữ Văn Hùng, Đặng Minh Quang, Phạm Công Đăng, Phạm Tuấn Anh.

### Slide 2: Đặt Vấn Đề (Problem Statement)
- Thách thức của RAG truyền thống trong thương mại điện tử.
- Hiện tượng trôi từ khóa số hiệu điều khoản & suy giảm chú ý (Lost in the Middle).

### Slide 3: Kiến Trúc Tổng Quan (Architecture Overview)
- Sơ đồ luồng: Hybrid Search (ChromaDB + BM25) $\rightarrow$ RRF Reranking $\rightarrow$ Fallback PageIndex ($<0.48$) $\rightarrow$ Reordering $\rightarrow$ LLM Citation.

### Slide 4: Live Demo Sản Phẩm
- Trình diễn giao diện Streamlit Chatbot.
- Demo Scenario 1 (In-domain) & Scenario 2 (Out-of-domain Fallback).

### Slide 5: Kết Quả Đánh Giá RAGAS & A/B Testing
- Bảng so sánh 4 chỉ số RAGAS giữa Hybrid RRF (0.921) vs Dense-Only (0.800).
- Mức độ cải thiện **+15.12%**.

### Slide 6: Kết Luận & Q&A
- Tổng kết dự án & Lời cảm ơn.

---
