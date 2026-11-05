# TV3: làm đặc trưng và mô hình cơ bản trong VS Code

**Mục tiêu:** TV3 trích xuất Histogram, HOG, GLCM và Edge Detection, sau đó so sánh Logistic Regression/KNN trên sáu biến thể ảnh. Nguồn TV2 dùng để đọc là `NGUYENTHIANHHONG_2045240092_TV2/outputs`; không dùng demo TV5.

## Cấu hình đã chốt theo `NGUYENTHIANHHONG_2045240092_TV2.py`

- **Nhãn:** `h_max`, tức chiều cao lớn nhất trong các pixel công trình của patch; đơn vị mét.
- **Bốn lớp:** lớp 0 `<10 m`; lớp 1 `10–<30 m`; lớp 2 `30–<60 m`; lớp 3 `≥60 m`. Biên 10, 30 và 60 thuộc lớp phía trên.
- **Ảnh đầu vào TV3:** TV2 thu nhỏ ảnh 256×256 xuống 128×128 bằng `INTER_AREA`, chuẩn hóa band theo phân vị 2–98% ước lượng từ 300 patch train, chọn band `RGB_IDX = [2,1,0]`, rồi xuất ảnh RGB `uint8` 3 kênh.
- **Kênh RGB:** dataset lưu bảy band với thứ tự đầu là Blue, Green, Red. Vì thế `[2,1,0]` chọn Red, Green, Blue cho ảnh hiển thị và feature.
- **Các biến thể:** `original`, `contrast`, `clahe`, `denoise`, `sharpen`, `combo`. Tham số TV2: contrast p2–p98; CLAHE clip 2, grid 8; median blur kernel 3; sharpen sigma 1, amount 1; combo là denoise → CLAHE → sharpen.
- **NoData và patch trống:** height NoData là `-1`, pixel nền hợp lệ là `0`; TV2 chỉ lấy pixel `>0` để tính `h_max`. Với patch không có pixel công trình (kể cả patch toàn NoData), TV2 gán `h_max=0`, nên code hiện xếp patch đó vào lớp Thấp. Giữ đúng hành vi TV2 để kết quả khớp; lưu ý lớp Thấp theo code hiện tại có thể gồm patch không có công trình.
- **Chia tập:** dùng split chính thức city-disjoint của PHDataset. GridSearchCV chọn tham số bằng 5 fold theo nhóm `city_id` trên train; validation báo cáo riêng; test để dành đánh giá cuối.

Nếu TV1/giảng viên đổi cách quy nhãn hoặc ngưỡng, trước tiên cập nhật TV2, chạy lại dữ liệu/đếm lớp, rồi mới sửa cấu hình TV3. Không đổi riêng TV3.

## 1. Mở đúng thư mục trong VS Code

1. Cài VS Code và extension **Python** của Microsoft nếu máy chưa có.
2. Trong VS Code chọn **File → Open Folder...**.
3. Mở `DOAN_XULYANH`, nơi nhìn thấy `NGUYENTHIANHHONG_2045240092_TV2` và `TV3_VSCode`.
4. Chọn **Terminal → New Terminal**. Nếu Terminal không ở thư mục gốc, dùng lệnh `cd "đường_dẫn_đến_thư_mục_đồ_án"`.

## 2. Tạo môi trường Python riêng

Chạy lần lượt trong Terminal của VS Code, tại thư mục gốc:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\TV3_VSCode\requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, chạy lệnh này một lần trong Terminal hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó nhấn `Ctrl+Shift+P` → chọn **Python: Select Interpreter** → chọn interpreter có `.venv` trong đường dẫn.

## 3. Chuẩn bị dữ liệu từ TV2

1. Dùng đầu ra có sẵn từ `NGUYENTHIANHHONG_2045240092_TV2/outputs`; TV3 không sửa file TV2.
2. TV3 đọc cache ảnh gốc và cấu hình chuẩn hóa/biến thể do Hồng lưu. Không cần các file ảnh trong `processed` vì TV3 tạo từng biến thể trực tiếp từ cache bằng đúng tham số cấu hình.
3. Xác nhận các file đầu vào sau tồn tại:

```text
NGUYENTHIANHHONG_2045240092_TV2/outputs/index.csv
NGUYENTHIANHHONG_2045240092_TV2/outputs/stats.csv
NGUYENTHIANHHONG_2045240092_TV2/outputs/cache/x_raw_128.npy
NGUYENTHIANHHONG_2045240092_TV2/outputs/configs/norm_stats.json
NGUYENTHIANHHONG_2045240092_TV2/outputs/configs/transforms.json
```

TV3 dựng các ảnh RGB 3 kênh từ cache 7 band bằng `norm_stats.json`, `transforms.json` và cùng thứ tự RGB/tham số trong code Hồng. Height map gốc là raster chiều cao liên tục, không phải nhãn 4 lớp có sẵn. Code TV3 ghép `sample_id` với `h_max` trong `stats.csv` rồi áp dụng đúng ngưỡng. PHDataset dùng split cố định theo thành phố; không chia lại ngẫu nhiên patch.

## 4. Trích xuất đặc trưng

1. Mở `TV3_VSCode/tv3_extract_features.py` và `tv3_features.py`. Không cần sửa code ở lần chạy đầu.
2. Trong Terminal ở thư mục gốc chạy:

```powershell
python .\TV3_VSCode\tv3_extract_features.py --variant original
```

3. Chương trình dựng biến thể được chọn từ cache TV2, lọc ảnh quang học hợp lệ, ghép ID theo split trong `stats.csv`, rồi lưu vector trong `TV3_VSCode/feature_cache/`.
4. Khi lệnh `original` chạy xong, chạy tiếp từng phiên bản:

```powershell
python .\TV3_VSCode\tv3_extract_features.py --variant contrast
python .\TV3_VSCode\tv3_extract_features.py --variant clahe
python .\TV3_VSCode\tv3_extract_features.py --variant denoise
python .\TV3_VSCode\tv3_extract_features.py --variant sharpen
python .\TV3_VSCode\tv3_extract_features.py --variant combo
```

Mỗi ảnh 128×128 RGB có bốn vector: Histogram (48 số), HOG, GLCM (48 số) và Edge (9 số). Tham số nằm trong `tv3_features.py`; ghi lại đúng các giá trị này trong báo cáo để thí nghiệm lặp lại được.

## 5. Huấn luyện Logistic Regression và KNN

Chạy trước với ảnh gốc:

```powershell
python .\TV3_VSCode\tv3_train_classic.py --variant original
```

Script so sánh từng nhóm riêng, ba tổ hợp `Hist+HOG`, `HOG+GLCM`, `Hist+GLCM+Edge`, và `all`. `StandardScaler` nằm trong `Pipeline`; GridSearchCV chạy trên train với 5 fold nhóm theo thành phố. CSV ghi trung bình và độ lệch chuẩn Macro-F1 giữa các fold (`cv_f1_macro_mean`, `cv_f1_macro_std`). Validation lưu thêm chỉ số tham khảo; test được giữ lại, không dùng để chọn mô hình.

Sau đó chạy lại với từng phiên bản tiền xử lý, ví dụ:

```powershell
python .\TV3_VSCode\tv3_train_classic.py --variant clahe
```

Kết quả được ghi vào `TV3_VSCode/results_tv3.csv`; các pipeline tốt nhất được lưu trong `TV3_VSCode/models/`. Bảng kết quả gồm Accuracy, Precision, Recall, Macro-F1, ROC-AUC đa lớp nếu tính được, confusion matrix và tham số tốt nhất.

## 6. Kiểm tra kết quả và bàn giao

1. Mở `results_tv3.csv` trong VS Code hoặc Excel.
2. So sánh các nhóm đặc trưng trong cùng một phiên bản ảnh; rồi so sánh cùng nhóm đặc trưng giữa các bản tiền xử lý.
3. Xem Macro-F1 và confusion matrix để biết lớp nào thường bị nhầm. Không kết luận chỉ từ Accuracy nếu số mẫu giữa các lớp lệch nhau.
4. Gửi TV4 bảng CSV và model đã chọn bằng validation. Chỉ TV4/nhóm chạy test sau khi đã chốt cấu hình.
5. Bàn giao TV5: model `.joblib`, quy tắc nhãn, thứ tự kênh RGB, thứ tự feature, scaler nằm trong pipeline và phiên bản thư viện.

## Lỗi thường gặp

- **Thiếu cache hoặc config:** kiểm tra đúng thư mục `NGUYENTHIANHHONG_2045240092_TV2/outputs` và đủ năm file đầu vào đã liệt kê ở mục 3. Không trỏ sang `XLAS_demo/XLAS_demo/outputs`.
- **Thiếu `stats.csv`:** chạy TV2 trước; không tự nhập nhãn bằng tay.
- **Lỗi không khớp sample ID/nhãn:** kiểm tra TV2 đã xuất `ids_{split}.npy` cùng lượt với ảnh chưa; không sắp xếp riêng một trong hai mảng.
- **Thiếu lớp hoặc GridSearchCV báo lỗi:** nhờ TV1/TV4 kiểm tra quy tắc nhãn và số mẫu mỗi lớp trong train; không tự đổi ngưỡng chỉ để code chạy.
- **Chạy quá lâu:** trước tiên chỉ trích xuất `original`; xác nhận dữ liệu và nhãn chạy đúng rồi mới làm các phiên bản còn lại.
