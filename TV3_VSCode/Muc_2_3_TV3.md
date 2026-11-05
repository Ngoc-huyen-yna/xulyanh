# 2.3. Trích xuất đặc trưng và thử nghiệm tổ hợp đặc trưng

Mục này trình bày quy trình biểu diễn ảnh vệ tinh bằng đặc trưng màu sắc, hình dạng và kết cấu, đồng thời khảo sát ảnh hưởng của từng nhóm đặc trưng và các tổ hợp đến bài toán phân loại độ cao công trình trong PHDataset. Đầu vào là ảnh RGB kích thước 128 × 128 pixel được chuẩn hóa theo quy trình tiền xử lý của nhóm. Nhãn phân loại được xác định từ chiều cao lớn nhất của công trình trong patch (`h_max`): thấp (<10 m), trung bình (10–<30 m), cao (30–<60 m) và rất cao (≥60 m).

**Hình 2.3. Quy trình trích xuất và đánh giá tổ hợp đặc trưng**

Ảnh vệ tinh RGB 128 × 128 → trích xuất Histogram màu, HOG và GLCM → ghép vector theo cấu hình thử nghiệm → chuẩn hóa đặc trưng bằng StandardScaler → huấn luyện và đánh giá bộ phân loại.

## 2.3.1. Các nhóm đặc trưng

### a) Histogram màu

Histogram màu mô tả phân bố cường độ trên ba kênh RGB, qua đó biểu diễn đặc điểm màu tổng quát của khu vực trong ảnh. Mỗi kênh được lượng tử hóa thành 16 khoảng giá trị trên miền [0, 256). Histogram của từng kênh được chuẩn hóa theo tổng số pixel của kênh đó rồi nối thành một vector 48 chiều. Cách chuẩn hóa này làm giảm ảnh hưởng của tổng số pixel và giữ lại tỷ lệ tương đối của các mức màu.

### b) Histogram of Oriented Gradients (HOG)

HOG biểu diễn phân bố hướng gradient, giúp mô tả đường biên và cấu trúc hình học xuất hiện trong ảnh. Ảnh RGB được chuyển sang ảnh xám và chuẩn hóa cường độ về [0, 1]. Các tham số trích xuất gồm 9 hướng gradient, kích thước ô (cell) 16 × 16 pixel, khối (block) 2 × 2 ô và chuẩn hóa block L2-Hys. Với ảnh 128 × 128 pixel, vector HOG có 1.764 phần tử. Đặc trưng này tập trung vào cấu trúc cạnh và hình dạng, không trực tiếp giữ thông tin màu của ảnh.

### c) Gray-Level Co-occurrence Matrix (GLCM)

GLCM mô tả quan hệ đồng xuất hiện giữa các cặp mức xám ở những khoảng cách và hướng xác định. Ảnh xám được lượng tử hóa thành 8 mức; ma trận được tính với khoảng cách 1 và 2 pixel theo bốn hướng 0°, 45°, 90° và 135°. Sáu thuộc tính được trích xuất gồm tương phản (contrast), độ bất đồng (dissimilarity), tính đồng nhất (homogeneity), năng lượng (energy), tương quan (correlation) và mômen góc bậc hai (angular second moment, ASM). Sáu thuộc tính trên bốn hướng và hai khoảng cách tạo thành vector GLCM 48 chiều. Nhóm đặc trưng này nhằm biểu diễn kết cấu bề mặt và mức độ biến thiên không gian của cường độ ảnh.

**Bảng 2.3. Đặc trưng trích xuất từ mỗi ảnh**

| Nhóm đặc trưng | Mô tả cấu hình | Số chiều |
|---|---|---:|
| Histogram màu | 16 khoảng cho mỗi kênh RGB; chuẩn hóa riêng từng kênh | 48 |
| HOG | 9 hướng; cell 16 × 16; block 2 × 2 cell; L2-Hys | 1.764 |
| GLCM | 8 mức xám; 2 khoảng cách; 4 hướng; 6 thuộc tính | 48 |
| Tổ hợp đầy đủ | Nối Histogram màu, HOG và GLCM | 1.860 |

## 2.3.2. Chuẩn hóa đặc trưng và kiểm soát rò rỉ dữ liệu

Các nhóm đặc trưng có miền giá trị và độ lớn khác nhau. Trước khi huấn luyện, StandardScaler được sử dụng để chuẩn hóa từng chiều đặc trưng theo trung bình và độ lệch chuẩn:

\[
z_j = \frac{x_j - \mu_j}{\sigma_j},
\]

trong đó \(x_j\) là giá trị của chiều đặc trưng \(j\), còn \(\mu_j\) và \(\sigma_j\) lần lượt là trung bình và độ lệch chuẩn được ước lượng trên dữ liệu huấn luyện.

Để tránh rò rỉ dữ liệu, các thống kê chuẩn hóa không được tính từ Validation hoặc Test. Trong bước đánh giá chéo 5-fold trên Train, scaler của mỗi fold chỉ được fit bằng phần Train của fold đó. Sau khi lựa chọn cấu hình, pipeline cuối cùng được fit lại trên toàn bộ Train; scaler đã fit này được giữ nguyên để transform Validation và Test. Vì vậy, dữ liệu Validation và Test không tham gia vào việc ước lượng tham số chuẩn hóa hoặc lựa chọn tham số trong quá trình fit mô hình.

## 2.3.3. Thiết kế thử nghiệm ablation

Thử nghiệm ablation so sánh ba nhóm đặc trưng đơn lẻ với hai tổ hợp từng cặp và tổ hợp đầy đủ. Vector của các nhóm trong một cấu hình được nối theo chiều đặc trưng trước khi đưa vào bộ phân loại. Trong phạm vi mục này, “Full Combined” được định nghĩa là Histogram màu + HOG + GLCM.

**Bảng 2.4. Các cấu hình đặc trưng trong thử nghiệm ablation**

| Mã cấu hình | Thành phần | Số chiều | Mục đích so sánh |
|---|---|---:|---|
| Histogram | Histogram màu | 48 | Đánh giá riêng thông tin màu |
| HOG | HOG | 1.764 | Đánh giá riêng thông tin gradient và hình dạng |
| GLCM | GLCM | 48 | Đánh giá riêng thông tin kết cấu |
| Histogram + HOG | Histogram màu và HOG | 1.812 | Đánh giá bổ sung giữa màu sắc và cấu trúc hình học |
| HOG + GLCM | HOG và GLCM | 1.812 | Đánh giá bổ sung giữa biên dạng và kết cấu |
| Full Combined | Histogram màu + HOG + GLCM | 1.860 | Đánh giá hiệu quả kết hợp cả ba nhóm đặc trưng |

Các cấu hình được đánh giá trên cùng một cách chia dữ liệu và cùng quy trình huấn luyện để bảo đảm khả năng so sánh. Dữ liệu Train được dùng cho StratifiedGroupKFold 5-fold có phân nhóm theo `city_id`; Macro-F1 trên các fold được dùng làm tiêu chí lựa chọn tham số. Sau bước lựa chọn, hiệu năng được báo cáo riêng trên Validation. Tập Test được giữ độc lập và chỉ sử dụng cho đánh giá cuối cùng sau khi cấu hình đã được chốt. Các bộ phân loại Logistic Regression và K-Nearest Neighbors (KNN) được khảo sát theo quy trình này.

**Bảng 2.5. Khung ghi nhận kết quả ablation trên Validation**

| Cấu hình đặc trưng | Macro-F1 (CV Train) | Macro-F1 (Validation) | Accuracy (Validation) | Precision macro (Validation) | Recall macro (Validation) |
|---|---:|---:|---:|---:|---:|
| Histogram | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |
| HOG | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |
| GLCM | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |
| Histogram + HOG | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |
| HOG + GLCM | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |
| Full Combined | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu | Chưa có số liệu |

*Ghi chú: Cần điền các ô kết quả từ lần chạy thực nghiệm tương ứng trước khi sử dụng bảng như bảng kết quả chính thức. Macro-F1 được ưu tiên khi so sánh do đây là trung bình F1 không trọng số giữa các lớp.*

## 2.3.4. Nguyên tắc phân tích

Hiệu quả của từng nhóm được xem xét qua kết quả trên Validation, trong đó Macro-F1 là chỉ số so sánh chính; Accuracy, Precision macro và Recall macro được dùng làm chỉ số bổ sung. So sánh Histogram, HOG và GLCM riêng lẻ cho biết mức thông tin mà mỗi nhóm đóng góp khi đứng độc lập. So sánh các tổ hợp theo cặp với các nhóm thành phần giúp đánh giá mức cải thiện khi bổ sung nguồn thông tin khác. Full Combined được đối chiếu với từng cấu hình còn lại để xác định liệu việc ghép cả ba nhóm có cải thiện khả năng phân loại hay không. Chênh lệch giữa Macro-F1 trung bình trong cross-validation và Macro-F1 trên Validation cũng cần được xem xét như một chỉ báo về khả năng tổng quát hóa; kết luận cuối cùng không được dựa trên một chỉ số đơn lẻ.
