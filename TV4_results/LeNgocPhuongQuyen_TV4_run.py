"""
TV4 - Mô hình & đánh giá (checklist #11, #13, #14, #16, #18, #27) trên feature cache của TV3.

Chạy cả quy trình bằng một lệnh (đây là đúng script sinh ra mọi số liệu trong báo cáo TV4):
    python TV4_run.py all
hoặc từng giai đoạn, KHÔNG đụng test cho tới giai đoạn cuối:
    python TV4_run.py explore   # #11 baseline + log; #13 DT/RF/XGB mặc định; so sánh 6 phiên bản ảnh bằng
                                #     CV nhóm theo city_id trên train (trung bình ± độ lệch chuẩn qua 5 fold)
    python TV4_run.py tune      # #14 GridSearchCV (cùng 5 fold) trên phiên bản ảnh tốt nhất, chọn mô hình
    python TV4_run.py final     # #16 test đúng một lần + #18 phân tích lỗi + số liệu cho #27
    python TV4_run.py export    # hình tham chiếu TV3 + TV4_ket_qua.xlsx (chỉ đọc kết quả đã lưu)

Đầu vào (ROOT = thư mục chứa file này, hoặc biến môi trường TV4_ROOT):
    outputs/stats.csv                               (TV2: nhãn h_max, split, city_id)
    TV3_VSCode/feature_cache/<variant>_<split>.npz  (TV3: hist, hog, glcm, edge, sample_id, optical_valid_frac)
    TV3_VSCode/results_tv3.csv                      (TV3: mô hình Logistic/KNN tốt nhất, làm tham chiếu)
Đầu ra: TV4_results/ (hoặc biến môi trường TV4_OUT). Thư viện: numpy pandas matplotlib scikit-learn xgboost joblib openpyxl
"""
import json, os, platform, sys, time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(os.environ.get("TV4_ROOT", Path(__file__).resolve().parent))
STATS = ROOT / "outputs" / "stats.csv"
FEAT_DIR = ROOT / "TV3_VSCode" / "feature_cache"
TV3_RESULTS = ROOT / "TV3_VSCode" / "results_tv3.csv"
OUT = Path(os.environ.get("TV4_OUT", ROOT / "TV4_results"))
FIG, MODELS = OUT / "figures", OUT / "models"
for d in (OUT, FIG, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# ---- Cấu hình phải khớp TV1/TV2/TV3 ----
SEED = 42
HEIGHT_COLUMN, THRESHOLDS = "h_max", [10.0, 30.0, 60.0]
CLASS_NAMES = ["Thấp (<10m)", "TB (10-30m)", "Cao (30-60m)", "Rất cao (>=60m)"]
SHORT = ["Thấp", "TB", "Cao", "Rất cao"]
LABELS = [0, 1, 2, 3]
VARIANTS = ["original", "contrast", "clahe", "denoise", "sharpen", "combo"]
FEATURE_PARTS = ["hist", "hog", "glcm", "edge"]          # ghép theo thứ tự này (giống TV3 'all')
N_JOBS = int(os.environ.get("TV4_JOBS", 2))
ENV = {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
       "numpy": np.__version__, "pandas": pd.__version__}

LOG_FILE = OUT / "experiment_log.csv"
LOG_COLUMNS = ["run_id", "timestamp", "member", "stage", "split_eval", "preprocessing", "features", "feature_dim",
               "model", "params", "cv_scheme", "seed", "n_train", "n_eval", "fit_time_s", "accuracy",
               "precision_macro", "recall_macro", "macro_f1", "weighted_f1", "roc_auc_ovr", "notes"]


def log(**row):
    if not LOG_FILE.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    row.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    row.setdefault("member", "TV4")
    row.setdefault("seed", SEED)
    row.setdefault("notes", json.dumps(ENV))
    if isinstance(row.get("params"), dict):
        row["params"] = json.dumps(row["params"], ensure_ascii=False, default=str)
    pd.DataFrame([{c: row.get(c, "") for c in LOG_COLUMNS}]).to_csv(
        LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")


def metrics(y, pred, proba=None):
    m = {"accuracy": accuracy_score(y, pred),
         "precision_macro": precision_score(y, pred, labels=LABELS, average="macro", zero_division=0),
         "recall_macro": recall_score(y, pred, labels=LABELS, average="macro", zero_division=0),
         "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
         "weighted_f1": f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0),
         "roc_auc_ovr": np.nan}
    if proba is not None and len(np.unique(y)) == 4:
        m["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=LABELS)
    return {k: round(float(v), 4) for k, v in m.items()}


# ---------------------------------------------------------------- dữ liệu
_stats = pd.read_csv(STATS).set_index("sample_id")


def load(variant, split):
    # allow_pickle: cache TV3 (25/10) lưu sample_id dạng object; file là dữ liệu nội bộ của nhóm
    d = np.load(FEAT_DIR / f"{variant}_{split}.npz", allow_pickle=True)
    ids = d["sample_id"].astype(str)
    assert (d["optical_valid_frac"] >= 0.5).all(), "cache chưa lọc optical_valid_frac >= 0.5"
    meta = _stats.reindex(ids)
    assert not meta[HEIGHT_COLUMN].isna().any(), "sample_id không khớp stats.csv"
    meta = meta.reset_index()
    y = np.digitize(meta[HEIGHT_COLUMN].to_numpy(), THRESHOLDS).astype(int)
    X = np.concatenate([d[p].astype("float32") for p in FEATURE_PARTS], axis=1)
    return X, y, meta


_cache = {}


def get(variant):
    if variant not in _cache:
        _cache[variant] = {sp: load(variant, sp) for sp in ("train", "val", "test")}
    return _cache[variant]


def proba(model, X):
    p, out = model.predict_proba(X), np.zeros((len(X), 4))
    out[:, np.asarray(model.classes_).astype(int)] = p
    return out


def build_models(grid="full"):
    dt = (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
          {"clf__max_depth": [5, 10, 20, None], "clf__min_samples_leaf": [1, 5, 10]})
    rf = (RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS),
          {"clf__max_depth": [10, 20, None], "clf__max_features": ["sqrt", 0.05]})
    xg = (XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", tree_method="hist", n_estimators=100,
                        learning_rate=0.1, max_depth=4, colsample_bytree=0.3, max_bin=64, random_state=SEED, n_jobs=N_JOBS),
          {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5], "clf__learning_rate": [0.05, 0.1]})
    return {k: (Pipeline([("clf", e)]), g) for k, (e, g) in {"DecisionTree": dt, "RandomForest": rf, "XGBoost": xg}.items()}


def fit_kw(name, y):
    return {"clf__sample_weight": compute_sample_weight("balanced", y)} if name == "XGBoost" else {}


def cv_folds(y, groups):
    """5 fold nhóm theo city_id trên train, cố định seed -> MỌI so sánh (phiên bản ảnh, GridSearch) dùng chung fold."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y, groups)), f"StratifiedGroupKFold-5(city_id, {len(np.unique(groups))} thành phố)"


# ---------------------------------------------------------------- giai đoạn 1
def explore():
    D0 = get("original")
    ytr, yva = D0["train"][1], D0["val"][1]
    ids0 = D0["train"][2].sample_id.to_numpy()
    counts = pd.DataFrame({sp: np.bincount(D0[sp][1], minlength=4) for sp in ("train", "val", "test")}, index=CLASS_NAMES)
    counts.to_csv(OUT / "class_counts_tv4.csv", encoding="utf-8-sig")
    print(counts)
    cities = {sp: set(D0[sp][2].city_id) for sp in ("train", "val", "test")}
    print("số thành phố train/val/test:", [len(c) for c in cities.values()], "| thành phố giao nhau giữa các split:",
          len(cities["train"] & cities["val"]) + len(cities["train"] & cities["test"]) + len(cities["val"] & cities["test"]))
    folds, cv_name = cv_folds(ytr, D0["train"][2].city_id.to_numpy())
    print(cv_name, "| số mẫu theo lớp trong từng fold kiểm định:", [np.bincount(ytr[v], minlength=4).tolist() for _, v in folds])

    # #11 baseline Majority: trên validation (giữ để đối chiếu) và trên cùng 5 fold CV
    base = DummyClassifier(strategy="most_frequent").fit(D0["train"][0], ytr)
    mb = metrics(yva, base.predict(D0["val"][0]), proba(base, D0["val"][0]))
    log(run_id="baseline_majority_val", stage="baseline", split_eval="val", preprocessing="-", features="-", feature_dim=0,
        model="Majority", params={"strategy": "most_frequent"}, cv_scheme="none", n_train=len(ytr), n_eval=len(yva), **mb)
    rb = cross_validate(DummyClassifier(strategy="most_frequent"), D0["train"][0], ytr, cv=folds,
                        scoring=["f1_macro", "accuracy"])
    rows = [{"preprocessing": "-", "model": "Majority", "fold": k, "macro_f1": f, "accuracy": a, "fit_time_s": 0.0}
            for k, (f, a) in enumerate(zip(rb["test_f1_macro"], rb["test_accuracy"]))]
    log(run_id="baseline_majority_cv", stage="baseline", split_eval="cv-train", preprocessing="-", features="-",
        feature_dim=0, model="Majority", params={"strategy": "most_frequent"}, cv_scheme=cv_name, n_train=len(ytr),
        macro_f1=round(rb["test_f1_macro"].mean(), 4), accuracy=round(rb["test_accuracy"].mean(), 4),
        notes=f"macro_f1_std={rb['test_f1_macro'].std():.4f}")
    print(f"Baseline: val macro_f1={mb['macro_f1']:.4f} | CV macro_f1={rb['test_f1_macro'].mean():.4f}±{rb['test_f1_macro'].std():.4f}")

    # #13 + so sánh 6 phiên bản ảnh: DT/RF/XGB mặc định, CÙNG 5 fold, cùng đặc trưng, cùng metric
    for v in VARIANTS:
        Xv, yv, mv = get(v)["train"]
        assert (mv.sample_id.to_numpy() == ids0).all() and (yv == ytr).all(), f"{v}: thứ tự mẫu khác original"
        for name, (pipe, _) in build_models().items():
            t0 = time.time()
            r = cross_validate(pipe, Xv, ytr, cv=folds, scoring=["f1_macro", "accuracy"], params=fit_kw(name, ytr))
            dt = round(time.time() - t0, 1)
            rows += [{"preprocessing": v, "model": name, "fold": k, "macro_f1": f, "accuracy": a, "fit_time_s": dt / len(folds)}
                     for k, (f, a) in enumerate(zip(r["test_f1_macro"], r["test_accuracy"]))]
            f1m, f1s = r["test_f1_macro"].mean(), r["test_f1_macro"].std()
            log(run_id=f"{name}_{v}_cv_default", stage="cv_variant", split_eval="cv-train", preprocessing=v,
                features="hist+hog+glcm+edge (TV3)", feature_dim=Xv.shape[1], model=name,
                params=pipe.named_steps["clf"].get_params(), cv_scheme=cv_name, n_train=len(ytr), fit_time_s=dt,
                macro_f1=round(f1m, 4), accuracy=round(r["test_accuracy"].mean(), 4),
                notes=json.dumps({"macro_f1_std": round(f1s, 4), "folds": np.round(r["test_f1_macro"], 4).tolist(), **ENV}))
            print(f"{v:9s} {name:13s} CV macro_f1={f1m:.4f}±{f1s:.4f} ({dt}s)", flush=True)

    folds_tab = pd.DataFrame(rows)
    folds_tab.to_csv(OUT / "cv_variant_folds.csv", index=False, encoding="utf-8-sig")
    m = folds_tab[folds_tab.model != "Majority"]
    per_model = m.groupby(["preprocessing", "model"]).agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                                                         accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
    per_model = per_model.reindex(VARIANTS, level=0).round(4)
    per_model.to_csv(OUT / "cv_variant_by_model.csv", encoding="utf-8-sig")
    # trung bình 3 mô hình theo TỪNG fold, rồi trung bình ± độ lệch chuẩn qua 5 fold; chênh lệch ghép cặp so với ảnh gốc
    fold_avg = m.pivot_table(index="fold", columns="preprocessing", values="macro_f1", aggfunc="mean")[VARIANTS]
    delta = fold_avg.sub(fold_avg["original"], axis=0)
    summ = pd.DataFrame({"macro_f1_mean": fold_avg.mean(), "macro_f1_std": fold_avg.std(),
                         "delta_vs_original_mean": delta.mean(), "delta_vs_original_std": delta.std(),
                         "so_fold_tot_hon_goc": (delta > 0).sum()})
    summ["ket_luan"] = np.where(summ.index == "original", "đối chứng",
                        np.where(summ.delta_vs_original_mean.abs() <= summ.delta_vs_original_std, "không khác biệt rõ",
                        np.where(summ.delta_vs_original_mean > 0, "cải thiện", "làm giảm")))
    summ = summ.round(4)
    summ.index.name = "preprocessing"
    summ.to_csv(OUT / "cv_variant_summary.csv", encoding="utf-8-sig")
    print(per_model); print(summ)
    # file cho TV2 (compare_preprocessing): cột preprocessing, model, accuracy, macro_f1 = trung bình CV
    per_model.reset_index().rename(columns={"macro_f1_mean": "macro_f1", "accuracy_mean": "accuracy"})[
        ["preprocessing", "model", "accuracy", "macro_f1", "macro_f1_std"]].to_csv(
        OUT / "experiments.csv", index=False, encoding="utf-8-sig")

    piv_m = per_model["macro_f1_mean"].unstack(); piv_s = per_model["macro_f1_std"].unstack()
    ax = piv_m.plot(kind="bar", yerr=piv_s, capsize=2, figsize=(9, 4), rot=0)
    ax.axhline(rb["test_f1_macro"].mean(), color="red", ls="--", lw=1, label="Majority baseline")
    ax.set_ylabel("Macro-F1 (CV 5 fold theo city_id)"); ax.set_xlabel(""); ax.legend(fontsize=8, ncol=4)
    plt.title("Phiên bản tiền xử lý × mô hình: trung bình ± độ lệch chuẩn qua 5 fold"); plt.tight_layout()
    plt.savefig(FIG / "cv_macro_f1_by_variant.png", dpi=150); plt.close()
    best = summ["macro_f1_mean"].idxmax()
    json.dump({"best_variant": best, "rule": "macro_f1 trung bình 3 mô hình, CV nhóm 5 fold trên train"},
              open(OUT / "best_variant.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("Phiên bản ảnh tốt nhất (CV):", best)


# ---------------------------------------------------------------- giai đoạn 2
def tv3_candidate(Dv, variant):
    """Mô hình tốt nhất của TV3 (Logistic/KNN) trên validation, fit lại trên train bằng tham số TV3 đã tune."""
    r = pd.read_csv(TV3_RESULTS)
    key = "cv_f1_macro_mean" if "cv_f1_macro_mean" in r.columns else "cv_f1_macro"
    r = r.loc[r[key].idxmax()]                     # TV3 chọn tham số/mô hình bằng CV trên train
    params = eval(r.best_params)
    est = (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED) if r.model == "logistic"
           else KNeighborsClassifier(n_jobs=N_JOBS))
    feats = FEATURE_PARTS if r.features == "all" else r.features.split("_")
    d = np.load(FEAT_DIR / f"{r.variant}_train.npz"); dv = np.load(FEAT_DIR / f"{r.variant}_val.npz")
    Xtr = np.concatenate([d[p] for p in feats], axis=1)
    Xva = np.concatenate([dv[p] for p in feats], axis=1)
    ytr = load(r.variant, "train")[1]
    pipe = Pipeline([("scale", StandardScaler()), ("model", est)]).set_params(**params).fit(Xtr, ytr)
    return f"TV3-{r.model}({r.variant}/{r.features})", pipe, Xva, load(r.variant, "val")[1], r.variant, r.features


def tune():
    variant = json.load(open(OUT / "best_variant.json"))["best_variant"]
    Dv = get(variant)
    Xtr, ytr, mtr = Dv["train"]; Xva, yva, _ = Dv["val"]
    folds, cv_name = cv_folds(ytr, mtr.city_id.to_numpy())
    print(cv_name, "| lớp trong từng fold validation:", [np.bincount(ytr[v], minlength=4).tolist() for _, v in folds])
    summ, cands = [], {}
    for name, (pipe, grid) in build_models().items():
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=folds, n_jobs=1, refit=True, error_score="raise")
        gs.fit(Xtr, ytr, **fit_kw(name, ytr))
        mv = metrics(yva, gs.predict(Xva), proba(gs.best_estimator_, Xva))
        dt = round(time.time() - t0, 1)
        i = gs.best_index_
        log(run_id=f"{name}_{variant}_gridsearch", stage="gridsearch", split_eval="val", preprocessing=variant,
            features="hist+hog+glcm+edge (TV3)", feature_dim=Xtr.shape[1], model=name, params=gs.best_params_,
            cv_scheme=cv_name, n_train=len(ytr), n_eval=len(yva), fit_time_s=dt, **mv)
        pd.DataFrame(gs.cv_results_).drop(columns=["params"]).to_csv(
            OUT / f"cv_results_{name}.csv", index=False, encoding="utf-8-sig")
        joblib.dump(gs.best_estimator_, MODELS / f"{name}_{variant}_best.joblib", compress=3)
        summ.append({"model": name, "preprocessing": variant, "best_params": json.dumps(gs.best_params_),
                     "cv_macro_f1": round(gs.best_score_, 4), "cv_std": round(gs.cv_results_["std_test_score"][i], 4),
                     "n_combos": len(gs.cv_results_["params"]), "fit_time_s": dt, **{f"val_{k}": v for k, v in mv.items()}})
        cands[name] = (gs.best_estimator_, mv["macro_f1"])
        print(f"{name:13s} CV={gs.best_score_:.4f}±{summ[-1]['cv_std']:.4f} | val macro_f1={mv['macro_f1']:.4f} | "
              f"{gs.best_params_} ({dt}s)", flush=True)

    pd.DataFrame(summ).to_csv(OUT / "gridsearch_tv4_only.csv", index=False, encoding="utf-8-sig")
    select()


def select():
    """Chọn mô hình CHÍNH bằng validation trong 3 mô hình TV4 đã tune (DT/RF/XGBoost) trên phiên bản ảnh tốt nhất.
    Mô hình tốt nhất của TV3 (Logistic/KNN, TV3 chọn bằng CV trên train) được fit lại bằng đúng tham số của TV3 và
    giữ làm THAM CHIẾU trên test, không tham gia chọn mô hình chính."""
    variant = json.load(open(OUT / "best_variant.json"))["best_variant"]
    t4 = pd.read_csv(OUT / "gridsearch_tv4_only.csv")
    primary = t4.sort_values("val_macro_f1", ascending=False).iloc[0]
    tname, tpipe, tXva, tyva, tvar, tfeat = tv3_candidate(get(variant), variant)
    mt = metrics(tyva, tpipe.predict(tXva), proba(tpipe, tXva))
    ref = {"model": tname, "preprocessing": tvar, "feature_set": tfeat,
           **{f"val_{k}": v for k, v in mt.items()}}
    full = pd.concat([t4, pd.DataFrame([{"model": tname, "preprocessing": tvar, "best_params": "(tham số TV3)",
                                         **{f"val_{k}": v for k, v in mt.items()}}])], ignore_index=True)
    full.sort_values("val_macro_f1", ascending=False).to_csv(OUT / "gridsearch_summary.csv", index=False, encoding="utf-8-sig")
    print(full[["model", "preprocessing", "cv_macro_f1", "val_macro_f1", "val_accuracy", "val_roc_auc_ovr"]])
    joblib.dump(joblib.load(MODELS / f"{primary.model}_{variant}_best.joblib"), MODELS / "best_model.joblib", compress=3)
    joblib.dump(tpipe, MODELS / "reference_tv3.joblib", compress=3)
    json.dump({"model": primary.model, "preprocessing": variant, "feature_set": "all", "label_mode": HEIGHT_COLUMN,
               "thresholds": THRESHOLDS, "class_names": CLASS_NAMES, "feature_order": FEATURE_PARTS, "env": ENV,
               "best_params": primary.best_params, "cv_macro_f1": float(primary.cv_macro_f1),
               "val_macro_f1": float(primary.val_macro_f1), "reference": ref},
              open(OUT / "selection.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("MÔ HÌNH CHÍNH (theo validation, trong DT/RF/XGB):", primary.model, "| ảnh:", variant, "| tham chiếu TV3:", tname)


# ---------------------------------------------------------------- giai đoạn 3
def final():
    sel = json.load(open(OUT / "selection.json", encoding="utf-8"))
    if LOG_FILE.exists() and (pd.read_csv(LOG_FILE).stage == "final_test").any() and "--force" not in sys.argv:
        sys.exit("Test đã được chạy (xem experiment_log.csv). Thêm --force chỉ khi có lý do và ghi rõ trong báo cáo.")
    model = joblib.load(MODELS / "best_model.joblib")
    var, fs = sel["preprocessing"], sel["feature_set"]
    d = np.load(FEAT_DIR / f"{var}_test.npz")
    parts = FEATURE_PARTS if fs == "all" else [fs]
    Xte = np.concatenate([d[p].astype("float32") for p in parts], axis=1)
    _, yte, mte = load(var, "test")
    ytr = load(var, "train")[1]
    pred, prob = model.predict(Xte), proba(model, Xte)
    m = metrics(yte, pred, prob)
    base_pred = np.full(len(yte), np.bincount(ytr).argmax())
    mb = metrics(yte, base_pred)
    log(run_id="FINAL_TEST", stage="final_test", split_eval="test", preprocessing=var, features=fs, feature_dim=Xte.shape[1],
        model=sel["model"], params=sel, cv_scheme="selected on validation", n_train=len(ytr), n_eval=len(yte), **m)
    print("TEST:", m, "\nBaseline Majority (test):", mb)

    # Tham chiếu: mô hình tốt nhất của TV3, chỉ để so sánh (KHÔNG dùng để chọn mô hình)
    r = sel["reference"]; rm = joblib.load(MODELS / "reference_tv3.joblib")
    rd = np.load(FEAT_DIR / f"{r['preprocessing']}_test.npz")
    Xr = np.concatenate([rd[p].astype("float32") for p in (FEATURE_PARTS if r["feature_set"] == "all" else r["feature_set"].split("_"))], axis=1)
    yr = load(r["preprocessing"], "test")[1]
    rpred = rm.predict(Xr); mr = metrics(yr, rpred, proba(rm, Xr))
    log(run_id="FINAL_TEST_REFERENCE", stage="final_test_reference", split_eval="test", preprocessing=r["preprocessing"],
        features=r["feature_set"], feature_dim=Xr.shape[1], model=r["model"], params="tham chiếu TV3",
        cv_scheme="selected by TV3 (CV on train)", n_train=len(ytr), n_eval=len(yr), **mr)
    json.dump({**r, "test": mr, "confusion_matrix": confusion_matrix(yr, rpred, labels=LABELS).tolist()},
              open(OUT / "test_reference_tv3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Tham chiếu TV3 (test):", mr)

    rep = pd.DataFrame(classification_report(yte, pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0,
                                             output_dict=True)).T.round(4)
    rep.to_csv(OUT / "test_classification_report.csv", encoding="utf-8-sig"); print(rep)
    cm = confusion_matrix(yte, pred, labels=LABELS)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(OUT / "test_confusion_matrix.csv", encoding="utf-8-sig")
    auc_cls = {CLASS_NAMES[k]: round(float(roc_auc_score((yte == k).astype(int), prob[:, k])), 4) for k in LABELS}
    json.dump({"macro_ovr": m["roc_auc_ovr"], "per_class": auc_cls}, open(OUT / "test_roc_auc.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("ROC-AUC từng lớp:", auc_cls)

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(LABELS, SHORT); ax.set_yticks(LABELS, SHORT)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix (test) – {sel['model']}", fontsize=9); fig.colorbar(im, fraction=.046)
    plt.tight_layout(); plt.savefig(FIG / "test_confusion_matrix.png", dpi=150); plt.close()

    # ---------- #18 phân tích lỗi ----------
    te = mte.copy(); te["y"], te["pred"] = yte, pred
    te["conf"] = prob[np.arange(len(te)), pred]; te["wrong"] = te.y != te.pred
    pairs = pd.DataFrame([(CLASS_NAMES[i], CLASS_NAMES[j], int(cm[i, j]), round(cm[i, j] / max(cm[i].sum(), 1), 4))
                          for i in LABELS for j in LABELS if i != j and cm[i, j] > 0],
                         columns=["thuc_te", "du_doan", "so_mau", "ty_le_trong_lop"]).sort_values("so_mau", ascending=False)
    pairs.to_csv(OUT / "error_confused_pairs.csv", index=False, encoding="utf-8-sig")
    per_class = pd.DataFrame({"lop": CLASS_NAMES, "so_mau_train": np.bincount(ytr, minlength=4), "so_mau_test": cm.sum(1),
                              "dung": np.diag(cm), "recall": np.round(np.diag(cm) / cm.sum(1).clip(min=1), 4),
                              "precision": np.round(np.diag(cm) / cm.sum(0).clip(min=1), 4)})
    per_class.to_csv(OUT / "error_per_class.csv", index=False, encoding="utf-8-sig")
    h = te[HEIGHT_COLUMN].to_numpy()
    near = np.min([np.abs(h - t) / t for t in THRESHOLDS], axis=0) <= 0.2
    w = te.wrong.to_numpy()
    summary = {"model": sel["model"], "preprocessing": var, **{f"test_{k}": v for k, v in m.items()},
               **{f"baseline_{k}": v for k, v in mb.items()},
               "chenh_weighted_tru_macro": round(m["weighted_f1"] - m["macro_f1"], 4),
               "so_loi": int(w.sum()), "ty_le_loi": round(float(w.mean()), 4),
               "loi_lech_1_bac": round(float((np.abs(te.y - te.pred)[te.wrong] == 1).mean()), 4),
               "loi_lech_2_bac_tro_len": round(float((np.abs(te.y - te.pred)[te.wrong] >= 2).mean()), 4),
               "loi_gan_nguong_20pct": round(float(near[w].mean()), 4),
               "mau_gan_nguong_trong_test": round(float(near.mean()), 4),
               "acc_mau_gan_nguong": round(float((~te.wrong[near]).mean()), 4),
               "acc_mau_xa_nguong": round(float((~te.wrong[~near]).mean()), 4)}
    json.dump(summary, open(OUT / "error_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    def seg(col, name, bins=None, labels=None):
        s = pd.cut(te[col], bins=bins, labels=labels, include_lowest=True) if bins is not None else te[col]
        r = te.groupby(s, observed=True).apply(lambda d: pd.Series({
            "so_mau": len(d), "accuracy": round(float((d.y == d.pred).mean()), 4),
            "macro_f1": round(f1_score(d.y, d.pred, labels=sorted(set(d.y) | set(d.pred)), average="macro", zero_division=0), 4),
            "ty_le_lop_Thap": round(float((d.y == 0).mean()), 3)}), include_groups=False)
        r.index.name = name; r.to_csv(OUT / f"error_by_{name}.csv", encoding="utf-8-sig"); print(r)
    seg("city_id", "city"); seg("reference_source", "nguon_nhan")
    seg("bld_frac", "mat_do_cong_trinh", bins=[-1e-9, 0, .05, .15, 1.0], labels=["0 (không có CT)", "0–5%", "5–15%", ">15%"])
    te["_sd_q"] = pd.qcut(te.img_std, 3, labels=["tương phản thấp", "vừa", "tương phản cao"])
    te["_mn_q"] = pd.qcut(te.img_mean, 3, labels=["tối", "vừa", "sáng"])
    seg("_sd_q", "do_tuong_phan_anh"); seg("_mn_q", "do_sang_anh")

    cols = ["sample_id", "city_id", "reference_source", HEIGHT_COLUMN, "bld_frac", "y", "pred", "conf"]
    te[te.wrong].sort_values("conf", ascending=False)[cols].to_csv(OUT / "error_worst_samples.csv", index=False,
                                                                   encoding="utf-8-sig")
    imp = getattr(getattr(model, "named_steps", {}).get("clf", None), "feature_importances_", None)
    if imp is not None and fs == "all":
        sizes = [np.load(FEAT_DIR / f"{var}_test.npz")[p].shape[1] for p in FEATURE_PARTS]
        edges = np.cumsum([0] + sizes)
        gi = pd.DataFrame({"nhom": ["Histogram màu", "HOG", "GLCM", "Edge"],
                           "so_chieu": sizes,
                           "tong_importance": [float(imp[edges[i]:edges[i + 1]].sum()) for i in range(4)]})
        gi["importance_tren_moi_chieu"] = gi.tong_importance / gi.so_chieu
        gi.round(5).to_csv(OUT / "feature_group_importance.csv", index=False, encoding="utf-8-sig"); print(gi.round(5))
    # figure phân bố dự đoán vs thực tế
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(4)
    ax.bar(x - .2, np.bincount(yte, minlength=4), .4, label="Thực tế"); ax.bar(x + .2, np.bincount(pred, minlength=4), .4, label="Dự đoán")
    ax.set_xticks(x, SHORT); ax.legend(); ax.set_title("Phân bố lớp trên test"); plt.tight_layout()
    plt.savefig(FIG / "test_class_distribution.png", dpi=150); plt.close()


def export():
    """Chỉ đọc kết quả đã lưu: hình confusion matrix của mô hình tham chiếu TV3 + file Excel tổng hợp. Không train, không chạy test."""
    r = json.load(open(OUT / "test_reference_tv3.json", encoding="utf-8"))
    cm = np.array(r["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(LABELS, SHORT); ax.set_yticks(LABELS, SHORT)
    for i in LABELS:
        for j in LABELS:
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix (test) – tham chiếu {r['model']}", fontsize=9); fig.colorbar(im, fraction=.046)
    plt.tight_layout(); plt.savefig(FIG / "test_confusion_matrix_reference_tv3.png", dpi=150); plt.close()
    sheets = {"experiment_log": "experiment_log.csv", "cv_phien_ban_tong": "cv_variant_summary.csv",
              "cv_phien_ban_mo_hinh": "cv_variant_by_model.csv", "cv_tung_fold": "cv_variant_folds.csv",
              "gridsearch": "gridsearch_summary.csv", "test_report": "test_classification_report.csv",
              "test_confusion": "test_confusion_matrix.csv", "loi_cap_nham": "error_confused_pairs.csv",
              "loi_theo_lop": "error_per_class.csv", "loi_theo_city": "error_by_city.csv",
              "loi_mat_do_CT": "error_by_mat_do_cong_trinh.csv", "loi_nguon_nhan": "error_by_nguon_nhan.csv",
              "nhom_dac_trung": "feature_group_importance.csv", "so_mau_lop": "class_counts_tv4.csv"}
    with pd.ExcelWriter(OUT / "TV4_ket_qua.xlsx", engine="openpyxl") as w:
        for name, f in sheets.items():
            pd.read_csv(OUT / f, encoding="utf-8-sig").to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]; ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(40, max(10, max(len(str(c.value or "")) for c in col[:30]) + 2))
    print("Đã xuất:", OUT / "TV4_ket_qua.xlsx")


if __name__ == "__main__":
    stages = {"explore": explore, "tune": tune, "select": select, "final": final, "export": export}
    for st in (["explore", "tune", "final", "export"] if sys.argv[1] == "all" else [sys.argv[1]]):
        print(f"\n========== {st} ==========", flush=True)
        stages[st]()
