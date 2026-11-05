# %%
# TV2 – PHDataset: khảo sát ảnh & tiền xử lý (chạy trên Spyder)
# PHẦN 0: IMPORT THƯ VIỆN (toàn bộ nằm ở đây)
# Cài thư viện bằng cửa sổ lệnh (cmd / Anaconda Prompt), KHÔNG cài trong Spyder:
#   pip install numpy pandas matplotlib opencv-python tifffile imagecodecs huggingface_hub tqdm
# (rasterio không bắt buộc: thiếu thì code tự dùng tifffile)
# =====================================================================
import os                                    # đường dẫn, số CPU, biến môi trường
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # tắt cảnh báo symlink của Hugging Face trên Windows
import io                                    # đọc file từ bộ nhớ (bytes)
import json                                  # đọc/ghi file cấu hình .json
import shutil                                # kiểm tra dung lượng ổ đĩa
import random                                # seed ngẫu nhiên
import zipfile                               # đọc file .zip trực tiếp (không cần giải nén)
import importlib.util                         # nạp file tv3_features.py của TV3 (Phần 11)
from pathlib import Path                     # đường dẫn gọn, chạy được cả Windows lẫn Linux
from concurrent.futures import ThreadPoolExecutor   # đọc nhiều file zip song song

import numpy as np                           # xử lý mảng số (ảnh là mảng numpy)
import pandas as pd                          # bảng dữ liệu (DataFrame)
import matplotlib.pyplot as plt              # vẽ biểu đồ, hiển thị ảnh (hiện ở khung Plots của Spyder)
import cv2                                   # OpenCV: resize, CLAHE, lọc nhiễu, làm nét
from tqdm import tqdm                        # thanh tiến trình

try:                                         # đọc GeoTIFF nhiều band: có thư viện nào dùng thư viện đó
    import rasterio
except ImportError:
    rasterio = None
try:
    import tifffile
except ImportError:
    tifffile = None
if rasterio is None and tifffile is None:
    raise ImportError("Cần cài ít nhất một trong hai: tifffile (khuyên dùng) hoặc rasterio")

from huggingface_hub import snapshot_download   # tải dataset từ Hugging Face bằng code

try:                                         # chỉ cần cho Phần 11 (so sánh có/không augmentation); thiếu thì các phần khác vẫn chạy
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    RandomForestClassifier = None            # cài bằng: pip install scikit-learn scikit-image

# %%
# PHẦN 1: CẤU HÌNH – mọi thứ cần sửa nằm ở đây
# =====================================================================
REPO_ID = "Yanjiao-WHU/PHDataset"

# Thư mục gốc của đồ án = thư mục chứa file .py này (nếu chạy từng ô trong Spyder mà không xác định được thì dùng thư mục làm việc hiện tại)
PROJECT_ROOT = Path(r"E:\XLAS")
BASE = PROJECT_ROOT  
DATA = PROJECT_ROOT / "DATA"                 # dữ liệu tải về (~6 GB)
OUT = PROJECT_ROOT / "outputs" 
DOWNLOAD = True                  # True: tải dữ liệu nếu chưa có; False: bỏ qua bước tải
CITIES = None                    # None = tải cả 26 thành phố (~6 GB). Muốn thử nhẹ: ["Adelaide", "Perth", "Ankara"] (tên như trong thư mục data/)

SEED = 42                        # cố định seed -> tái lập kết quả
RGB_IDX = [2, 1, 0]              # 3 band hiển thị RGB. TẠM THỜI – xác nhận lại theo bài báo!
IMG_SIZE = 128                   # kích thước ảnh sau resize (chốt với TV3)
THRESHOLDS = [10, 30, 60]        # mốc chiều cao (m) chia 4 lớp; TV1 sẽ chốt
CLASS_NAMES = ["Thấp (<10m)", "TB (10-30m)", "Cao (30-60m)", "Rất cao (>=60m)"]
LABEL_MODE = "h_max"             # cách quy nhãn pixel -> 1 số cho cả patch: h_max | h_mean | h_p90
P_LOW, P_HIGH = 2, 98            # phân vị dùng để chuẩn hóa từng band

LIMIT = None                     # None = chạy hết 9475 mẫu; đặt 300 để chạy thử nhanh
MIN_VALID_FRAC = 0.0             # bỏ patch có tỷ lệ pixel nhãn hợp lệ (khác nodata) thấp hơn mức này; 0 = giữ hết. TV1 quyết định.
N_WORKERS = min(8, os.cpu_count() or 4)   # số luồng đọc song song (mỗi luồng xử lý 1 file zip)
FORCE_REBUILD = False            # True: bỏ cache cũ, đọc lại toàn bộ
EXPORT_VARIANTS = False          # True: xuất các phiên bản ảnh ra file .npy (cho TV3 dùng)

FIG, CACHE, PROC, CFG = OUT / "figures", OUT / "cache", OUT / "processed", OUT / "configs"
for d in (DATA, FIG, CACHE, PROC, CFG):
    d.mkdir(parents=True, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# %%
# PHẦN 2: TẢI DỮ LIỆU BẰNG CODE (chỉ lấy phần cần dùng; bỏ qua file thừa như tools/, splits/, README...)
# Cần: mạng ổn định + ~10 GB trống. Nếu bị ngắt giữa chừng: chạy lại ô này, nó tải tiếp phần còn thiếu.
# =====================================================================
def need_download():
    """Cần tải nếu thiếu dataset.json/metadata hoặc thiếu file zip của các thành phố yêu cầu."""
    if not (DATA / "dataset.json").exists() or not (DATA / "metadata").exists():
        return True
    if CITIES is None:
        return len(list((DATA / "data").glob("*.zip"))) < 26
    return any(not (DATA / "data" / f"{c}.zip").exists() for c in CITIES)


if DOWNLOAD and need_download():
    DATA.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(DATA).free / 1e9
    print(f"Dung lượng trống: {free_gb:.1f} GB")
    if CITIES is None and free_gb < 12:
        raise RuntimeError("Cần ít nhất ~12 GB trống để tải toàn bộ dữ liệu. Giải phóng ổ đĩa hoặc đặt CITIES = [...] để tải ít hơn.")
    patterns = ["dataset.json", "metadata/*", "normalization/*"]                  # file thông tin + danh sách mẫu
    patterns += ["data/*.zip"] if CITIES is None else [f"data/{c}.zip" for c in CITIES]   # ảnh theo thành phố
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=str(DATA), allow_patterns=patterns)
    print("Đã tải xong về:", DATA)
else:
    print("Bỏ qua bước tải (đã đủ dữ liệu hoặc DOWNLOAD=False). Thư mục:", DATA)
print("Số file zip hiện có:", len(list((DATA / "data").glob("*.zip"))) if (DATA / "data").exists() else 0)

# %%
# PHẦN 3: BẢNG CHỈ MỤC – đọc từ file csv của dataset (1 dòng = 1 mẫu)
# Mỗi dòng có: sample_id, city, scene_id, split, archive (file zip), optical_path, height_path (đường dẫn BÊN TRONG zip)
# =====================================================================
info = json.load(open(DATA / "dataset.json", encoding="utf-8"))     # thông tin chính thức của dataset
N_BANDS, NODATA = info["optical_bands"], info["nodata"]              # 7 band, nodata = -1
print(f"Dataset v{info['version']} | split: {info['split']} | band: {N_BANDS} | nodata nhãn: {NODATA} | đơn vị: {info['height_unit']}")

df = pd.concat([pd.read_csv(DATA / "metadata" / f"patches_{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
df = df.drop(columns=[c for c in df.columns if c.endswith("_sha256") or c.endswith("_bytes")])   # bỏ cột không cần

# Kiểm tra: số mẫu khớp dataset.json?
print("Số mẫu trong danh sách:", len(df), "| dataset.json ghi:", info["patches"])
print("Theo split:", df.split.value_counts().to_dict(), "| dataset.json ghi:", info["counts"])

# Chỉ giữ các mẫu có file zip thật (cho phép tải thiếu / chỉ tải vài thành phố để thử)
has_zip = df.archive.map(lambda a: (DATA / a).exists())
if (~has_zip).any():
    print(f"CẢNH BÁO: thiếu {df.loc[~has_zip, 'archive'].nunique()} file zip -> bỏ {int((~has_zip).sum())} mẫu của các thành phố đó.")
df = df[has_zip].reset_index(drop=True)
if df.empty:
    raise FileNotFoundError(f"Không có file zip nào trong {DATA / 'data'}. Chạy lại Phần 2 (DOWNLOAD = True).")

# Kiểm tra đường dẫn bên trong zip có thật không
bad = 0
for arc, sub in df.groupby("archive"):
    names = set(zipfile.ZipFile(DATA / arc).namelist())
    bad += int((~sub.optical_path.isin(names)).sum() + (~sub.height_path.isin(names)).sum())
print("Đường dẫn trong zip không tìm thấy:", bad)

if LIMIT:      # chạy thử nhanh: lấy ngẫu nhiên LIMIT mẫu
    df = df.sample(min(LIMIT, len(df)), random_state=SEED).reset_index(drop=True)
df["cache_idx"] = np.arange(len(df))        # vị trí của mẫu trong file cache
df.head()

# %%
# PHẦN 4: HÀM ĐỌC ẢNH/NHÃN TỪ BÊN TRONG ZIP (không cần giải nén) + MỞ THỬ 1 MẪU
# =====================================================================
def read_tif_bytes(b):
    """Giải mã nội dung file .tif (dạng bytes) thành mảng numpy."""
    if rasterio is not None:
        from rasterio.io import MemoryFile
        with MemoryFile(b) as mf, mf.open() as src:
            return src.read()                        # (band, H, W)
    return tifffile.imread(io.BytesIO(b))            # có thể ra (H, W, band) hoặc (band, H, W)


def read_image(zf, path):
    """Ảnh 7 band từ zip -> (C, H, W) float32."""
    a = read_tif_bytes(zf.read(path)).astype("float32")
    if a.ndim == 2:
        a = a[None]
    elif a.ndim == 3 and a.shape[-1] <= 16 and a.shape[0] > 16:    # dạng (H,W,C) -> đổi sang (C,H,W)
        a = a.transpose(2, 0, 1)
    return a


def read_label(zf, path):
    """Nhãn chiều cao (m) từ zip -> (H, W) float32.
    Giữ NGUYÊN quy ước dataset: pixel = NODATA (-1) là KHÔNG CÓ DỮ LIỆU nhãn; 0 là nền (không có công trình); >0 là chiều cao."""
    a = read_tif_bytes(zf.read(path)).astype("float32")
    if a.ndim == 3:
        a = a[0] if a.shape[0] <= 4 else a[..., 0]
    a = np.nan_to_num(a, nan=NODATA, posinf=NODATA, neginf=NODATA)
    a[a < 0] = NODATA                                  # mọi giá trị âm đều coi là nodata
    return a


# Mở thử mẫu đầu tiên
r0 = df.iloc[0]
with zipfile.ZipFile(DATA / r0.archive) as z:
    img0, lbl0 = read_image(z, r0.optical_path), read_label(z, r0.height_path)
print(f"Mẫu: {r0.sample_id} | nguồn nhãn: {r0.reference_source}")
print(f"Ảnh {img0.shape} {img0.dtype} | min {np.nanmin(img0):.3f} max {np.nanmax(img0):.3f} | NaN {int(np.isnan(img0).sum())}")
print(f"Nhãn {lbl0.shape} | các giá trị khác nhau (10 đầu): {np.unique(lbl0)[:10]} | max {lbl0.max()} | pixel nodata: {(lbl0 == NODATA).mean():.1%}")
assert img0.shape[0] == N_BANDS, f"số band đọc được {img0.shape[0]} khác {N_BANDS}"


def rgb_view(x):
    """Ảnh (H,W,C) -> ảnh màu 3 band, kéo giãn theo phân vị 2-98% để dễ nhìn."""
    x = np.asarray(x, dtype="float32")[..., RGB_IDX]
    lo, hi = np.nanpercentile(x, (2, 98))
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(rgb_view(img0.transpose(1, 2, 0))); ax[0].set_title("Ảnh (RGB tạm)")
im = ax[1].imshow(np.ma.masked_equal(lbl0, NODATA), cmap="magma"); ax[1].set_title("Nhãn chiều cao (m), nodata để trống")
for a in ax: a.axis("off")
plt.colorbar(im, ax=ax[1], fraction=0.046); plt.show()

# %%
# PHẦN 5: ĐỌC MỘT LẦN – TÍNH THỐNG KÊ + TẠO CACHE (song song theo từng file zip)
# Mỗi mẫu được đọc đúng 1 lần, trong cùng lần đọc ta: (1) tính thống kê ảnh/nhãn (2) resize rồi lưu cache (3) tính hash.
# Mỗi luồng xử lý trọn 1 file zip (tự mở ZipFile riêng) nên không tranh nhau.
# Cache: 1 file .npy kiểu float16, hình dạng (số mẫu, IMG_SIZE, IMG_SIZE, 7), giữ NGUYÊN giá trị gốc (chưa chuẩn hóa).
# =====================================================================
CACHE_FILE = CACHE / f"x_raw_{IMG_SIZE}.npy"
STATS_FILE = OUT / "stats.csv"
INDEX_FILE = OUT / "index.csv"


def resize_raw(img_chw, size=IMG_SIZE):
    """(C,H,W) -> (size,size,C) float32, thay NaN bằng 0, resize INTER_AREA (giữ chi tiết khi thu nhỏ)."""
    x = np.ascontiguousarray(np.nan_to_num(img_chw, nan=0.0).transpose(1, 2, 0))
    return cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)


def img_hash(small_hwc):
    """Hash ảnh 256 bit: thu ảnh xám về 16x16 rồi so với trung bình. Ảnh giống nhau -> cùng chuỗi."""
    g = cv2.resize(small_hwc[..., RGB_IDX].mean(axis=2).astype("float32"), (16, 16), interpolation=cv2.INTER_AREA)
    return "".join("1" if v else "0" for v in (g > g.mean()).ravel())


def height_stats(lbl):
    """Từ bản đồ chiều cao (H,W):
    - valid_frac: tỷ lệ pixel có nhãn hợp lệ (khác nodata)
    - bld_frac  : tỷ lệ pixel có công trình (>0) TRONG vùng hợp lệ
    - h_max / h_mean / h_p90: 3 cách quy ra 1 số cho patch (chỉ tính trên pixel công trình)"""
    valid = lbl != NODATA
    b = lbl[lbl > 0]
    out = {"valid_frac": float(valid.mean()), "bld_frac": float(b.size / max(valid.sum(), 1))}
    out.update({"h_max": float(b.max()), "h_mean": float(b.mean()), "h_p90": float(np.percentile(b, 90))} if b.size
               else {"h_max": 0.0, "h_mean": 0.0, "h_p90": 0.0})
    return out


def build_cache():
    cache = np.lib.format.open_memmap(CACHE_FILE, mode="w+", dtype=np.float16,
                                      shape=(len(df), IMG_SIZE, IMG_SIZE, N_BANDS))
    pbar = tqdm(total=len(df))

    def process_archive(item):                        # xử lý trọn 1 file zip (chạy trong 1 luồng)
        arc, sub = item
        recs = []
        with zipfile.ZipFile(DATA / arc) as zf:
            for r in sub.itertuples():
                i = r.cache_idx
                try:
                    img, lbl = read_image(zf, r.optical_path), read_label(zf, r.height_path)   # <<< ĐỌC: duy nhất 1 lần
                    if img.shape[0] != N_BANDS:
                        raise ValueError(f"số band = {img.shape[0]}")
                    small = resize_raw(img)
                    cache[i] = np.clip(small, -6e4, 6e4).astype(np.float16)   # float16 tối đa ~65504 nên kẹp lại
                    rec = {"cache_idx": i, "ok": True, "H": img.shape[1], "W": img.shape[2],
                           "label_shape_match": lbl.shape == img.shape[1:],
                           "nan_img": int(np.isnan(img).sum()), "all_const": bool(np.nanmax(img) == np.nanmin(img)),
                           "img_mean": float(np.nanmean(img)), "img_std": float(np.nanstd(img)), "hash": img_hash(small)}
                    rec.update(height_stats(lbl))
                    recs.append(rec)
                except Exception as e:
                    recs.append({"cache_idx": i, "ok": False, "err": str(e)})
                pbar.update(1)
        return recs

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        results = [r for part in ex.map(process_archive, list(df.groupby("archive"))) for r in part]
    pbar.close(); cache.flush()
    out = df.merge(pd.DataFrame(results), on="cache_idx")
    out.to_csv(STATS_FILE, index=False); df.to_csv(INDEX_FILE, index=False)
    return out


if CACHE_FILE.exists() and STATS_FILE.exists() and not FORCE_REBUILD and pd.read_csv(INDEX_FILE).sample_id.tolist() == df.sample_id.tolist():
    st_all = pd.read_csv(STATS_FILE)                  # có cache rồi -> dùng lại, không đọc lại zip
    print("Dùng lại cache:", CACHE_FILE)
else:
    st_all = build_cache()
cache = np.load(CACHE_FILE, mmap_mode="r")            # mở cache dạng memmap (không nạp hết vào RAM)
print("Cache:", cache.shape, cache.dtype, f"~{cache.nbytes / 1e9:.2f} GB | mẫu lỗi:", int((~st_all.ok).sum()))
if (~st_all.ok).any():
    print(st_all.loc[~st_all.ok, ["sample_id", "err"]].head())
st = st_all[st_all.ok & (st_all.valid_frac >= MIN_VALID_FRAC)].copy()     # bảng thống kê các mẫu dùng được

# %%
# PHẦN 6: EDA (dùng bảng thống kê và cache, không đọc lại zip)
# =====================================================================
def to_class(h):
    "Chiều cao (m) -> lớp 0..3 theo THRESHOLDS."
    return int(np.searchsorted(THRESHOLDS, h, side="right"))


# 6.1 Chất lượng dữ liệu
print("Kích thước gốc khác nhau:", st.groupby(["H", "W"]).size().to_dict())
print(f"NaN: {(st.nan_img > 0).sum()} | ảnh toàn 1 giá trị: {st.all_const.sum()} | nhãn lệch kích thước: {(~st.label_shape_match).sum()}")
print(f"Patch không có công trình: {(st.bld_frac == 0).sum()} / {len(st)}")
print(f"Patch có <50% pixel nhãn hợp lệ (nhiều nodata): {(st.valid_frac < 0.5).sum()} | không có pixel hợp lệ nào: {(st.valid_frac == 0).sum()}")

# 6.2 Thử 3 cách quy nhãn -> bảng đếm số mẫu mỗi lớp (gửi TV1 để chốt)
tab = pd.DataFrame({m: st[m].map(to_class).value_counts().reindex(range(4), fill_value=0) for m in ("h_max", "h_mean", "h_p90")})
tab.index = CLASS_NAMES
print(tab); tab.to_csv(OUT / "class_counts_by_label_mode.csv")
st["cls"] = st[LABEL_MODE].map(to_class)

# 6.3 Biểu đồ phân bố
fig, ax = plt.subplots(1, 4, figsize=(20, 4))
ax[0].hist(st.loc[st.h_max > 0, "h_max"], bins=60); ax[0].set_yscale("log")
for t in THRESHOLDS: ax[0].axvline(t, color="red", ls="--")          # vạch đỏ = mốc chia lớp
ax[0].set_title("Chiều cao lớn nhất mỗi patch (m)")
st.groupby(["split", "cls"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0).plot(kind="bar", ax=ax[1])
ax[1].set_title(f"Số patch theo lớp và split ({LABEL_MODE})"); ax[1].legend(CLASS_NAMES, fontsize=7)
st.groupby(["reference_source", "cls"]).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0).plot(kind="bar", ax=ax[2])
ax[2].set_title("Số patch theo lớp và nguồn nhãn"); ax[2].legend(CLASS_NAMES, fontsize=7)
ax[3].hist(st.valid_frac, bins=40); ax[3].set_title("Tỷ lệ pixel nhãn hợp lệ (khác nodata)")
plt.tight_layout(); plt.savefig(FIG / "eda_distributions.png", dpi=150, bbox_inches="tight"); plt.show()

# 6.4 Kiểm tra chia tập: dataset ghi split là "city-disjoint" -> không thành phố nào nằm ở >1 split
cs = st.groupby("city_id")["split"].nunique()
print(f"Số thành phố: {cs.size} | thành phố nằm trong >1 split: {(cs > 1).sum()}  (mong đợi 0)")
print("Số thành phố theo split:", st.groupby("split")["city_id"].nunique().to_dict(), "| dataset.json ghi:", info["city_counts"])
print("Chiều cao lớn nhất trung bình theo split (m):", st.groupby("split")["h_max"].mean().round(1).to_dict())

# 6.5 Ảnh trùng (hash đã tính sẵn ở Phần 5) – nhóm trùng nằm ở nhiều split = nghi rò rỉ
g = st.groupby("hash")
dup = g.filter(lambda x: len(x) > 1)
print(f"Mẫu thuộc nhóm trùng: {len(dup)} | nhóm trùng xuyên split: {(g['split'].nunique() > 1).sum()}")

# 6.6 Lưới ảnh mẫu theo lớp (lấy từ cache, rất nhanh)
n = 6
fig, axes = plt.subplots(4, n, figsize=(2.4 * n, 10))
for c in range(4):
    pool = st[st.cls == c]
    pick = pool.sample(min(n, len(pool)), random_state=SEED) if len(pool) else pool
    for j in range(n):
        axes[c, j].axis("off")
        if j < len(pick):
            axes[c, j].imshow(rgb_view(cache[pick.iloc[j].cache_idx])); axes[c, j].set_title(f"{pick.iloc[j].h_max:.0f} m", fontsize=8)
    axes[c, 0].text(-0.1, 0.5, CLASS_NAMES[c], transform=axes[c, 0].transAxes, ha="right", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "samples_by_class.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
# PHẦN 6B: KHẢO SÁT BÓNG ĐỔ (checklist việc 3 yêu cầu khảo sát bóng đổ)
# Dataset KHÔNG có mask bóng đổ, nên ta dùng ĐỊNH LƯỢNG GẦN ĐÚNG (proxy): tỷ lệ pixel tối hơn hẳn mức sáng điển hình của patch.
#   shadow_frac = tỷ lệ pixel có độ sáng < SHADOW_REL x độ sáng trung vị của patch
# Hạn chế (ghi vào báo cáo): pixel tối cũng có thể là mặt nước, mái tối, cây rậm; ảnh nhiều tầng sáng tối cũng bị tính.
# Tính trực tiếp từ cache nên rất nhanh, không đọc lại zip.
# =====================================================================
SHADOW_REL = 0.75                                # ngưỡng tương đối so với độ sáng trung vị (0.5 quá chặt: gần như không bắt được pixel nào trên dữ liệu thật)


def shadow_mask(x_hwc, rel=None):
    "Ảnh (H,W,C) -> mặt nạ bool (H,W) các pixel nghi là bóng đổ."
    rel = SHADOW_REL if rel is None else rel
    g = np.asarray(x_hwc, dtype="float32")[..., RGB_IDX].mean(axis=2)     # độ sáng = trung bình 3 band RGB
    valid = (g > 0) & (g != NODATA)                                       # pixel = 0 hoặc -1 là thiếu dữ liệu, bỏ qua
    if not valid.any():
        return np.zeros(g.shape, bool)
    return (g < rel * np.median(g[valid])) & valid


# Bảng hiệu chỉnh: thử vài ngưỡng trên 500 patch ngẫu nhiên để chọn SHADOW_REL (kết hợp xem ảnh overlay phía dưới)
_calib = st.sample(min(500, len(st)), random_state=SEED).cache_idx
print("Hiệu chỉnh ngưỡng bóng đổ (trung bình tỷ lệ pixel bị coi là bóng đổ trên 500 patch):")
for _r in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
    _f = np.array([shadow_mask(cache[i], _r).mean() for i in _calib])
    print(f"  SHADOW_REL={_r:<4} -> trung bình {_f.mean():.4f} | trung vị {np.median(_f):.4f} | phân vị 90: {np.quantile(_f, .9):.4f}")

st["shadow_frac"] = [float(shadow_mask(cache[i]).mean()) for i in tqdm(st.cache_idx)]
st[["sample_id", "split", "city_id", "h_max", "cls", "bld_frac", "shadow_frac"]].to_csv(OUT / "shadow_stats.csv", index=False)

# Bảng theo lớp + tương quan Spearman giữa bóng đổ và chiều cao (chỉ patch có công trình)
by_cls = st.groupby("cls")["shadow_frac"].agg(["count", "mean", "median"]).round(4)
by_cls.index = [CLASS_NAMES[c] for c in by_cls.index]
print("Tỷ lệ bóng đổ (proxy) theo lớp chiều cao:\n", by_cls)
has_b = st[st.h_max > 0]
rho = has_b["shadow_frac"].corr(has_b["h_max"], method="spearman")
print(f"Tương quan Spearman(shadow_frac, h_max) trên patch có công trình: {rho:.3f}  (gần 0 = bóng đổ gần như không liên quan chiều cao)")
by_cls.to_csv(OUT / "shadow_by_class.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].boxplot([st.loc[st.cls == c, "shadow_frac"].dropna() for c in range(4)])
ax[0].set_xticks([1, 2, 3, 4]); ax[0].set_xticklabels(["Thấp", "TB", "Cao", "Rất cao"])
ax[0].set_title("Tỷ lệ pixel bóng đổ (proxy) theo lớp chiều cao"); ax[0].set_ylabel("shadow_frac")
ax[1].hexbin(has_b["h_max"], has_b["shadow_frac"], gridsize=40, bins="log", mincnt=1)
ax[1].set_xlabel("h_max (m)"); ax[1].set_ylabel("shadow_frac"); ax[1].set_title(f"Spearman = {rho:.3f}")
plt.tight_layout(); plt.savefig(FIG / "shadow_vs_height.png", dpi=150, bbox_inches="tight"); plt.show()

# Ảnh overlay để KIỂM TRA bằng mắt proxy có bắt đúng bóng không (đỏ = pixel bị coi là bóng đổ). Nếu sai, chỉnh SHADOW_REL.
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for c in range(4):
    pool = st[st.cls == c]
    for r in range(2):
        a = axes[r, c]; a.axis("off")
        if len(pool) > r:
            row = pool.sample(min(2, len(pool)), random_state=SEED).iloc[r]
            x = np.asarray(cache[int(row.cache_idx)], dtype="float32")
            v = rgb_view(x).copy()
            m = shadow_mask(x)
            v[m] = 0.6 * v[m] + 0.4 * np.array([1, 0, 0])
            a.imshow(v); a.set_title(f"{CLASS_NAMES[c]} | shadow={row.shadow_frac:.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "shadow_overlay.png", dpi=150, bbox_inches="tight"); plt.show()

# %%
# PHẦN 7: TIỀN XỬ LÝ CƠ BẢN
# Quy ước CỐ ĐỊNH của nhóm:  preprocess = resize (INTER_AREA)  ->  chuẩn hóa từng band theo phân vị (tính từ TRAIN) -> [0,1]
# Vì resize làm TRƯỚC chuẩn hóa nên cache (đã resize) chỉ cần chuẩn hóa là ra đúng kết quả pipeline đầy đủ.
# =====================================================================
train_idx = st[st.split == "train"].cache_idx.to_numpy()
sel = np.sort(rng.choice(train_idx, size=min(300, len(train_idx)), replace=False))
vals = np.asarray(cache[sel], dtype="float32").reshape(-1, N_BANDS)           # gom pixel từ 300 patch train
vals = vals[rng.choice(len(vals), size=min(1_000_000, len(vals)), replace=False)]
p_lo, p_hi = np.percentile(vals, [P_LOW, P_HIGH], axis=0)                     # mỗi band 1 cặp (thấp, cao)
json.dump({"p_lo": p_lo.tolist(), "p_hi": p_hi.tolist(), "img_size": IMG_SIZE, "rgb_idx": RGB_IDX,
           "p_low_pct": P_LOW, "p_high_pct": P_HIGH, "seed": SEED}, open(CFG / "norm_stats.json", "w"), indent=2)  # TV3, TV5 dùng lại
print("p_lo:", np.round(p_lo, 3), "\np_hi:", np.round(p_hi, 3))
# Tác giả có kèm thống kê chuẩn hóa chính thức; in ra để tham khảo/đối chiếu (chưa dùng thay thế)
_nf = DATA / "normalization" / "normalization.json"
if _nf.exists():
    print("normalization.json (tác giả):", _nf.read_text(encoding="utf-8")[:600])


def normalize(x):
    "Chuẩn hóa mảng (..., C) về [0,1] theo p_lo/p_hi từng band."
    return np.clip((np.asarray(x, dtype="float32") - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)


def preprocess(img_chw):
    "HÀM TIỀN XỬ LÝ CHUNG cho train/val/test/demo. Vào: ảnh gốc (C,H,W). Ra: (IMG_SIZE,IMG_SIZE,C) float32 trong [0,1]."
    return normalize(resize_raw(img_chw))


# Kiểm tra: đi đường đầy đủ (đọc file -> preprocess) phải khớp đường nhanh (cache -> normalize)
i = int(st.cache_idx.iloc[0]); ri = df.iloc[i]
with zipfile.ZipFile(DATA / ri.archive) as _z:
    full = preprocess(read_image(_z, ri.optical_path))
fast = normalize(cache[i])
assert full.shape == (IMG_SIZE, IMG_SIZE, N_BANDS) and full.min() >= 0 and full.max() <= 1
print("Sai khác lớn nhất giữa 2 đường (do float16):", float(np.abs(full - fast).max()))
assert np.abs(full - fast).max() < 0.02

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(rgb_view(img0.transpose(1, 2, 0))); ax[0].set_title(f"Trước ({img0.shape[1]}x{img0.shape[2]})")
ax[1].imshow(full[..., RGB_IDX]); ax[1].set_title(f"Sau ({IMG_SIZE}x{IMG_SIZE})")
for a in ax: a.axis("off")
plt.show()

# %%
# PHẦN 8: CÁC PHIÊN BẢN ẢNH (tính ngay từ cache, không cần lưu hàng nghìn file)
# Các phép làm việc trên ảnh RGB uint8 (0-255). Tham số lưu trong PARAMS để tái lập.
# =====================================================================
PARAMS = {"contrast": {"lo": 2, "hi": 98}, "clahe": {"clip": 2.0, "grid": 8},
          "denoise": {"k": 3}, "sharpen": {"sigma": 1.0, "amount": 1.0}}


def to_rgb_u8(x01):
    "(H,W,C) trong [0,1] -> 3 band RGB uint8."
    return (np.clip(x01[..., RGB_IDX], 0, 1) * 255).astype("uint8")


def contrast_stretch(u8):
    a, b = np.percentile(u8, (PARAMS["contrast"]["lo"], PARAMS["contrast"]["hi"]))
    return np.clip((u8.astype("float32") - a) / (b - a + 1e-8) * 255, 0, 255).astype("uint8")


def clahe(u8):
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)                 # chỉ chỉnh kênh sáng L, giữ màu
    lab[..., 0] = cv2.createCLAHE(clipLimit=PARAMS["clahe"]["clip"],
                                  tileGridSize=(PARAMS["clahe"]["grid"],) * 2).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise(u8):
    return cv2.medianBlur(u8, PARAMS["denoise"]["k"])


def sharpen(u8):
    blur = cv2.GaussianBlur(u8, (0, 0), PARAMS["sharpen"]["sigma"])
    a = PARAMS["sharpen"]["amount"]
    return cv2.addWeighted(u8, 1 + a, blur, -a, 0)            # unsharp mask


VARIANTS = {"original": lambda u: u, "contrast": contrast_stretch, "clahe": clahe, "denoise": denoise,
            "sharpen": sharpen, "combo": lambda u: sharpen(clahe(denoise(u)))}   # combo: khử nhiễu -> CLAHE -> làm nét


def variant_batch(idx, name):
    """Trả về mảng uint8 (n, IMG_SIZE, IMG_SIZE, 3) của phiên bản `name` cho các mẫu có cache_idx trong `idx`.
    TV3 gọi hàm này để trích đặc trưng cho từng phiên bản."""
    fn = VARIANTS[name]
    return np.stack([fn(to_rgb_u8(normalize(cache[i]))) for i in idx])


# Lưới so sánh: mỗi hàng 1 lớp, mỗi cột 1 phiên bản
fig, axes = plt.subplots(4, len(VARIANTS), figsize=(2.4 * len(VARIANTS), 10))
for c in range(4):
    pool = st[st.cls == c]
    if pool.empty:
        continue
    u8 = to_rgb_u8(normalize(cache[int(pool.sample(1, random_state=SEED).cache_idx.iloc[0])]))
    for j, (name, fn) in enumerate(VARIANTS.items()):
        axes[c, j].imshow(fn(u8)); axes[c, j].axis("off")
        if c == 0: axes[c, j].set_title(name)
plt.tight_layout(); plt.savefig(FIG / "variants_grid.png", dpi=150, bbox_inches="tight"); plt.show()
json.dump({"variants": list(VARIANTS), "params": PARAMS, "img_size": IMG_SIZE, "rgb_idx": RGB_IDX},
          open(CFG / "transforms.json", "w"), indent=2)

if EXPORT_VARIANTS:       # xuất mỗi (phiên bản, split) thành 1 file .npy uint8 để TV3 nạp trực tiếp
    for name in VARIANTS:
        for sp in ("train", "val", "test"):
            idx = st[st.split == sp].cache_idx.to_numpy()
            np.save(PROC / f"{name}_{sp}.npy", variant_batch(idx, name))
            np.save(PROC / f"ids_{sp}.npy", st[st.split == sp].sample_id.to_numpy())
    print("Đã xuất vào", PROC)

# %%
# PHẦN 9: TĂNG CƯỜNG DỮ LIỆU (CHỈ TRAIN)
# Phép an toàn cho ảnh viễn thám: xoay 90 độ, lật, đổi sáng nhẹ. Không zoom/crop mạnh (làm đổi kích thước biểu kiến của công trình).
# Nhãn là 1 số cho cả patch nên xoay/lật không đổi nhãn.
# =====================================================================
AUG_CFG = {"rot90": "0/90/180/270 ngẫu nhiên", "flip_lr_prob": 0.5, "brightness_range": [0.95, 1.05], "seed": SEED}


def augment(x01, split, rng=rng):
    "x01: (H,W,C) trong [0,1]. Chốt an toàn: split khác 'train' -> báo lỗi."
    if split != "train":
        raise ValueError(f"CHỈ augment tập train, nhận split='{split}'")
    x = np.rot90(x01, k=int(rng.integers(0, 4)))
    if rng.random() < AUG_CFG["flip_lr_prob"]:
        x = np.fliplr(x)
    return np.clip(x * rng.uniform(*AUG_CFG["brightness_range"]), 0, 1).astype("float32")


base = normalize(cache[int(train_idx[0])])
fig, ax = plt.subplots(1, 6, figsize=(14, 3))
ax[0].imshow(base[..., RGB_IDX]); ax[0].set_title("gốc")
for k in range(1, 6):
    ax[k].imshow(augment(base, "train")[..., RGB_IDX]); ax[k].set_title(f"aug {k}")
for a in ax: a.axis("off")
plt.savefig(FIG / "augmentation_demo.png", dpi=150, bbox_inches="tight"); plt.show()
try:
    augment(base, "test")
except ValueError as e:
    print("OK, chốt an toàn hoạt động:", e)
json.dump(AUG_CFG, open(CFG / "augmentation.json", "w"), indent=2, ensure_ascii=False)

# %%
# PHẦN 10: SO SÁNH ẢNH GỐC VÀ ẢNH TIỀN XỬ LÝ (dùng kết quả của TV3 và TV4)
# Quy ước thư mục của nhóm: PROJECT_ROOT/outputs (TV2), PROJECT_ROOT/TV3_VSCode (TV3), PROJECT_ROOT/TV4_results (TV4).
# Vì vậy OUT ở Phần 1 phải là PROJECT_ROOT/outputs, ví dụ OUT = Path(r"E:\\outputs").
# =====================================================================
PROJECT_ROOT = OUT.parent
TV4_EXPERIMENTS = PROJECT_ROOT /"LENGOCPHUONGUYEN_2045240256_TV4"/"TV4_results" / "experiments.csv"    # TV4: DT/RF/XGBoost, tham số mặc định, 6 phiên bản ảnh
TV3_RESULTS = PROJECT_ROOT / "CAONGOCHUYEN_2045240113_TV3"/"TV3_VSCode" / "results_tv3.csv"            # TV3: Logistic/KNN, đã tune bằng GridSearchCV
EPS = 0.01                                                               # chênh Macro-F1 nhỏ hơn mức này coi là "không đáng kể"
VARIANT_ORDER = ["original", "contrast", "clahe", "denoise", "sharpen", "combo"]


def load_results():
    "Gộp kết quả của TV3 và TV4 về cùng dạng: preprocessing | model | macro_f1 (đều đánh giá trên VALIDATION, bộ đặc trưng 'all')."
    frames = []
    if TV4_EXPERIMENTS.exists():
        a = pd.read_csv(TV4_EXPERIMENTS)[["preprocessing", "model", "macro_f1"]]
        frames.append(a)
    if TV3_RESULTS.exists():
        b = pd.read_csv(TV3_RESULTS)
        b = b[b["features"] == "all"].rename(columns={"variant": "preprocessing", "val_f1_macro": "macro_f1"})
        frames.append(b[["preprocessing", "model", "macro_f1"]])
    if not frames:
        raise FileNotFoundError(f"Chưa có kết quả: {TV4_EXPERIMENTS} và {TV3_RESULTS}")
    return pd.concat(frames, ignore_index=True)


def verdict(d):
    "Kết luận theo Δ Macro-F1 so với ảnh gốc."
    if pd.isna(d):
        return "chưa có dữ liệu"
    return "không đáng kể" if abs(d) < EPS else ("cải thiện" if d > 0 else "làm giảm")


def compare_preprocessing():
    res = load_results()
    table = res.pivot_table(index="preprocessing", columns="model", values="macro_f1", aggfunc="mean")
    table = table.reindex([v for v in VARIANT_ORDER if v in table.index])
    if "original" not in table.index:
        raise KeyError("Thiếu dòng 'original' (ảnh gốc) làm đối chứng trong kết quả")
    delta = table.subtract(table.loc["original"], axis=1)             # Δ so với ảnh gốc, từng mô hình
    # Trung bình chỉ tính trên các mô hình có kết quả ở MỌI phiên bản ảnh (nếu không, các hàng có số mô hình khác nhau sẽ không so sánh được)
    common = table.columns[table.notna().all()]
    if len(common) == 0:
        raise ValueError("Chưa có mô hình nào được chạy đủ cho cả các phiên bản ảnh; chờ TV3/TV4 chạy thêm")
    print("Mô hình dùng để lấy trung bình (có đủ mọi phiên bản):", list(common))
    summary = pd.DataFrame({"macro_f1_trung_binh": table[common].mean(axis=1), "delta_trung_binh": delta[common].mean(axis=1),
                            "so_mo_hinh_co_ket_qua": table.notna().sum(axis=1)})
    summary["ket_luan"] = summary["delta_trung_binh"].map(verdict)
    summary.loc["original", "ket_luan"] = "đối chứng"
    print("Macro-F1 (validation):\n", table.round(4), "\n\nΔ so với original:\n", delta.round(4), "\n\n", summary.round(4))
    table.round(4).to_csv(OUT / "preprocessing_macro_f1.csv", encoding="utf-8-sig")
    delta.round(4).to_csv(OUT / "preprocessing_delta.csv", encoding="utf-8-sig")
    summary.round(4).to_csv(OUT / "preprocessing_comparison.csv", encoding="utf-8-sig")
    ax = delta.drop(index="original").plot(kind="bar", figsize=(10, 4))
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-EPS, EPS, color="grey", alpha=0.2, label=f"|Δ| < {EPS} (không đáng kể)")
    ax.set_ylabel("Δ Macro-F1 so với ảnh gốc"); ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "preprocessing_delta.png", dpi=150, bbox_inches="tight"); plt.show()
    return table, delta, summary


if TV4_EXPERIMENTS.exists() or TV3_RESULTS.exists():
    try:
        compare_preprocessing()
    except Exception as e:                       # lỗi ở đây không được làm dừng Phần 11
        print("PHẦN 10 BÁO LỖI (đã bỏ qua để chạy tiếp):", repr(e))
else:
    print("Chưa có kết quả của TV3/TV4 tại:", PROJECT_ROOT, "-> kiểm tra OUT ở Phần 1 và vị trí các thư mục")

# %%
# PHẦN 11: SO SÁNH KẾT QUẢ CÓ / KHÔNG AUGMENTATION
# Cách làm: cùng phiên bản ảnh, cùng đặc trưng (hàm extract_features của TV3), cùng mô hình, cùng validation.
#   (a) KHÔNG augment: huấn luyện trên ảnh train gốc
#   (b) CÓ augment   : huấn luyện trên ảnh train gốc + AUG_COPIES bản augment mỗi ảnh (chỉ train; val/test không đụng tới)
# Chỉ đánh giá trên VALIDATION. Không dùng test để quyết định có augment hay không.
# =====================================================================
AUG_VARIANT = "original"          # phiên bản ảnh dùng để so sánh
AUG_COPIES = 1                    # số bản augment tạo thêm cho mỗi ảnh train
AUG_MODEL_SEEDS = [42, 43, 44]    # chạy nhiều seed mô hình để biết chênh lệch có lớn hơn độ dao động ngẫu nhiên không
AUG_MAX_TRAIN = None              # None = dùng hết ảnh train; đặt 1500 để chạy thử nhanh
# Tìm tv3_features.py ở vài chỗ thường gặp; nếu vẫn không thấy, gán trực tiếp: TV3_FEATURES_PY = Path(r"E:\\doan\\TV3_VSCode\\tv3_features.py")
_candidates = [Path(r"E:\XLAS\CAONGOCHUYEN_2045240113_TV3\TV3_VSCode\tv3_features.py"), BASE / "TV3_VSCode" / "tv3_features.py",
               BASE / "tv3_features.py", Path.cwd() / "tv3_features.py", Path.cwd() / "TV3_VSCode" / "tv3_features.py",
               Path.home() / "Downloads" / "tv3_features.py"]
TV3_FEATURES_PY = next((p for p in _candidates if p.exists()), _candidates[0])


def load_tv3_extractor(path):
    "Nạp hàm extract_features từ file tv3_features.py của TV3 để dùng ĐÚNG cùng đặc trưng với nhóm."
    spec = importlib.util.spec_from_file_location("tv3_features", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_features


def features_of(images_u8, extract):
    "Danh sách ảnh RGB uint8 -> ma trận đặc trưng (hist + hog + glcm + edge ghép theo đúng thứ tự TV3)."
    def one(img):
        f = extract(img)
        return np.concatenate([f["hist"], f["hog"], f["glcm"], f["edge"]]).astype("float32")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        return np.stack(list(tqdm(ex.map(one, images_u8), total=len(images_u8))))


def augment_u8(img_u8, rng_local):
    "Augment ảnh RGB uint8 bằng đúng hàm augment() ở Phần 9 (xoay 90 độ, lật, đổi sáng nhẹ); vẫn có chốt chỉ-train."
    out = augment(img_u8.astype("float32") / 255.0, "train", rng_local)
    return (out * 255.0).round().astype("uint8")


def compare_augmentation():
    if RandomForestClassifier is None:
        raise ImportError("Cần cài scikit-learn: pip install scikit-learn scikit-image")
    if not TV3_FEATURES_PY.exists():
        raise FileNotFoundError(f"Không thấy {TV3_FEATURES_PY}. Cần file tv3_features.py của TV3 (xem PROJECT_ROOT ở Phần 10).")
    extract = load_tv3_extractor(TV3_FEATURES_PY)

    tr = st[st.split == "train"]
    if AUG_MAX_TRAIN:
        tr = tr.sample(min(AUG_MAX_TRAIN, len(tr)), random_state=SEED)
    va = st[st.split == "val"]
    y_tr = np.digitize(tr[LABEL_MODE].to_numpy(), THRESHOLDS)
    y_va = np.digitize(va[LABEL_MODE].to_numpy(), THRESHOLDS)
    print(f"Train: {len(tr)} ảnh | Val: {len(va)} ảnh | phiên bản ảnh: {AUG_VARIANT} | lớp train: {np.bincount(y_tr, minlength=4).tolist()}")

    img_tr = variant_batch(tr.cache_idx.to_numpy(), AUG_VARIANT)
    img_va = variant_batch(va.cache_idx.to_numpy(), AUG_VARIANT)
    X_tr, X_va = features_of(img_tr, extract), features_of(img_va, extract)

    rng_aug = np.random.default_rng(SEED)                   # seed riêng để bản augment tái lập được
    aug_imgs = [augment_u8(im, rng_aug) for _ in range(AUG_COPIES) for im in img_tr]
    X_aug = features_of(np.stack(aug_imgs), extract)
    y_aug = np.tile(y_tr, AUG_COPIES)                       # nhãn của bản augment = nhãn ảnh gốc (xoay/lật không đổi chiều cao)
    X_tr_aug, y_tr_aug = np.concatenate([X_tr, X_aug]), np.concatenate([y_tr, y_aug])

    def make(name, seed):
        if name == "RandomForest":
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
            return Pipeline([("clf", clf)])
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])

    rows = []
    for name in ("RandomForest", "LogisticRegression"):
        for setting, (X, y) in {"khong_augment": (X_tr, y_tr), "co_augment": (X_tr_aug, y_tr_aug)}.items():
            scores = [f1_score(y_va, make(name, sd).fit(X, y).predict(X_va), labels=[0, 1, 2, 3], average="macro", zero_division=0)
                      for sd in AUG_MODEL_SEEDS]
            rows.append({"model": name, "setting": setting, "n_train": len(y), "macro_f1_mean": np.mean(scores), "macro_f1_std": np.std(scores)})
            print(f"{name:18s} {setting:14s} n_train={len(y):6d}  Macro-F1 = {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    res = pd.DataFrame(rows)
    piv = res.pivot(index="model", columns="setting", values="macro_f1_mean")
    piv["delta"] = piv["co_augment"] - piv["khong_augment"]
    piv["ket_luan"] = piv["delta"].map(verdict)              # dùng cùng ngưỡng EPS như Phần 10
    print("\n", piv.round(4))
    res.round(4).to_csv(OUT / "augmentation_comparison_detail.csv", index=False, encoding="utf-8-sig")
    piv.round(4).to_csv(OUT / "augmentation_comparison.csv", encoding="utf-8-sig")
    return piv


if not TV3_FEATURES_PY.exists():
    print("!!! PHẦN 11 KHÔNG CHẠY: không thấy tv3_features.py. Đã tìm ở:", *[str(p) for p in _candidates], sep="\n    ")
    print("    -> gán đường dẫn đúng cho TV3_FEATURES_PY ở trên rồi chạy lại ô này.")
elif RandomForestClassifier is None:
    print("!!! PHẦN 11 KHÔNG CHẠY: chưa cài scikit-learn. Mở cmd và gõ: pip install scikit-learn scikit-image")
else:
    aug_result = compare_augmentation()
    print("Đã lưu:", OUT / "augmentation_comparison.csv")

