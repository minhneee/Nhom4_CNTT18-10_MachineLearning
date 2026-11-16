# Khai báo sử dụng AI (AI Usage Declaration)

> **Đề tài 5:** Nhận dạng chữ số viết tay – **Học phần:** Nhập môn Học máy (FIT4013)
> **Nhóm 4 – Lớp CNTT 18-10 – Khoa CNTT, Trường Đại học Đại Nam**

## 1. Công cụ AI đã sử dụng

| STT | Công cụ | Nhà phát triển | Mục đích sử dụng |
|:-:|---|---|---|
| 1 | Antigravity AI Assistant | Google DeepMind | Gợi ý cấu trúc code, notebook, GUI Tkinter, định dạng README/LaTeX |
| 2 | Claude Code | Anthropic | Rà soát toàn bộ dự án theo hướng dẫn bài tập lớn, sửa lỗi, cải thiện tiền xử lý ảnh của GUI, hỗ trợ soạn báo cáo LaTeX |

## 2. Chi tiết phạm vi sử dụng

| Phần | Mức độ | Chi tiết |
|---|---|---|
| Cấu trúc notebook, pipeline | Hỗ trợ | Gợi ý cách tổ chức EDA → tiền xử lý → đặc trưng → huấn luyện → đánh giá → phân tích lỗi |
| Rà soát và sửa lỗi | Hỗ trợ | Phát hiện 5 cell notebook lỗi cú pháp; tham số đang được chọn theo tập Test (đã chuyển sang 5-Fold CV trên Train); số liệu trong README/báo cáo không khớp kết quả chạy thật; thiếu thử nghiệm `criterion=gini`; cột thời gian huấn luyện đo sai |
| GUI demo | Hỗ trợ | Cấu trúc giao diện Tkinter; tiền xử lý ảnh vẽ/ảnh chụp giống cách tạo bộ optdigits; tách nhiều chữ số |
| Báo cáo LaTeX, README | Hỗ trợ | Định dạng, cấu trúc 6 phần theo hướng dẫn, bảng biểu tự sinh từ kết quả chạy |

## 3. Phần nhóm tự thực hiện và chịu trách nhiệm

- Hiểu yêu cầu đề tài, lựa chọn thuật toán (K-NN, Cây quyết định) và các phương án biểu diễn đặc trưng.
- Chạy lại toàn bộ pipeline, kiểm tra kết quả, phân tích và diễn giải số liệu, phân tích lỗi.
- Kiểm tra tính đúng đắn của mã nguồn và nội dung báo cáo.

## 4. Cam kết

- Mọi số liệu trong báo cáo được **sinh ra từ việc chạy mã nguồn** (`generate_full_project.py`, `evaluate_gui_preprocessing.py`), có thể tái tạo với `random_state=42`.
- Các thành viên đã đọc hiểu và **có khả năng giải thích** toàn bộ pipeline, mã nguồn và kết quả khi bảo vệ.
- AI chỉ là công cụ hỗ trợ; nhóm chịu trách nhiệm về tính đúng đắn của sản phẩm nộp.
