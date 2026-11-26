from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PATH_DATASET = Path(
    r"D:\Kuliah Semester 9\Skripsi (Data Baru)\jumlah-penumpang-angkutan-umum-yang-terlayani-per-hari-komponen-data.xlsx"
)

FOLDER_OUTPUT = Path(
    r"D:\Kuliah Semester 9\Skripsi (Data Baru)\hasil_xgboost"
)

KOLOM_TANGGAL = "tanggal"
KOLOM_TARGET = "jumlah_penumpang_per_hari"

FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)

if not PATH_DATASET.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {PATH_DATASET}")

df = pd.read_excel(PATH_DATASET)

if df.empty:
    raise ValueError("Dataset kosong.")

if KOLOM_TANGGAL not in df.columns:
    raise KeyError(f"Kolom tanggal '{KOLOM_TANGGAL}' tidak ditemukan.")

if KOLOM_TARGET not in df.columns:
    raise KeyError(f"Kolom target '{KOLOM_TARGET}' tidak ditemukan.")

print("Dataset berhasil dibaca.")
print("Jumlah baris:", len(df))
print("Jumlah kolom:", len(df.columns))
print("Kolom:", list(df.columns))
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.isna().sum())

df[KOLOM_TANGGAL] = pd.to_datetime(
    df[KOLOM_TANGGAL],
    errors="coerce"
)

if df[KOLOM_TANGGAL].isna().any():
    raise ValueError("Terdapat tanggal tidak valid.")

df = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)

if df[KOLOM_TANGGAL].duplicated().any():
    raise ValueError("Terdapat tanggal duplikat dalam dataset.")

df.insert(0, "nomor_urut", range(1, len(df) + 1))

df.to_csv(
    FOLDER_OUTPUT / "02_data_terurut.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Data berhasil diurutkan.")
print(df.head())
print(df.tail())

target_asli = df[KOLOM_TARGET].copy()
target_teks = target_asli.astype(str).str.strip()

target_teks = target_teks.str.replace(
    r"\.0+$",
    "",
    regex=True
)

target_teks = target_teks.str.replace(
    r"[.,\s]",
    "",
    regex=True
)

mask_valid = target_teks.str.fullmatch(
    r"\d+",
    na=False
)

df[KOLOM_TARGET] = pd.to_numeric(
    target_teks.where(mask_valid),
    errors="coerce"
)

mask_invalid = df[KOLOM_TARGET].isna()

if mask_invalid.any():
    data_invalid = df.loc[
        mask_invalid,
        ["nomor_urut", KOLOM_TANGGAL]
    ].copy()

    data_invalid["nilai_asli"] = target_asli[mask_invalid]

    print(data_invalid.to_string(index=False))
    raise ValueError("Terdapat nilai target yang tidak valid.")

df.to_csv(
    FOLDER_OUTPUT / "03_target_validasi.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Seluruh target berhasil dikonversi.")
print("Tipe target:", df[KOLOM_TARGET].dtype)

df["tahun"] = df[KOLOM_TANGGAL].dt.year
df["bulan"] = df[KOLOM_TANGGAL].dt.month
df["hari"] = df[KOLOM_TANGGAL].dt.day
df["hari_dalam_minggu"] = df[KOLOM_TANGGAL].dt.dayofweek
df["minggu_ke"] = df[KOLOM_TANGGAL].dt.isocalendar().week.astype(int)
df["akhir_pekan"] = (
    df["hari_dalam_minggu"] >= 5
).astype(int)

kolom_fitur_kalender = [
    "nomor_urut",
    KOLOM_TANGGAL,
    "tahun",
    "bulan",
    "hari",
    "hari_dalam_minggu",
    "minggu_ke",
    "akhir_pekan"
]

print("Fitur kalender:")
print(df[kolom_fitur_kalender].head())
print(df[kolom_fitur_kalender].tail())

df.to_csv(
    FOLDER_OUTPUT / "04_data_fitur_kalender.csv",
    index=False,
    encoding="utf-8-sig"
)

df = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)

df["lag_1"] = df[KOLOM_TARGET].shift(1)
df["lag_7"] = df[KOLOM_TARGET].shift(7)

df["rolling_mean_7"] = (
    df[KOLOM_TARGET]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["rolling_mean_14"] = (
    df[KOLOM_TARGET]
    .shift(1)
    .rolling(window=14)
    .mean()
)

kolom_fitur_historis = [
    "nomor_urut",
    KOLOM_TANGGAL,
    KOLOM_TARGET,
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14"
]

print("Fitur historis:")
print(df[kolom_fitur_historis].head())
print(df[kolom_fitur_historis].tail())
print(df[kolom_fitur_historis].isna().sum())

df.to_csv(
    FOLDER_OUTPUT / "05_data_fitur_historis.csv",
    index=False,
    encoding="utf-8-sig"
)

FITUR_KALENDER = [
    "tahun",
    "bulan",
    "hari",
    "hari_dalam_minggu",
    "minggu_ke",
    "akhir_pekan"
]

FITUR_HISTORIS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14"
]

FITUR_MODEL = FITUR_KALENDER + FITUR_HISTORIS

df_model = df.dropna(
    subset=FITUR_MODEL + [KOLOM_TARGET]
).copy()

df_model = df_model.sort_values(
    KOLOM_TANGGAL
).reset_index(drop=True)

X = df_model[FITUR_MODEL].copy()
y = df_model[KOLOM_TARGET].copy()

batas_pembagian = int(len(df_model) * 0.80)

X_train = X.iloc[:batas_pembagian].copy()
X_test = X.iloc[batas_pembagian:].copy()

y_train = y.iloc[:batas_pembagian].copy()
y_test = y.iloc[batas_pembagian:].copy()

data_train = df_model.iloc[:batas_pembagian].copy()
data_test = df_model.iloc[batas_pembagian:].copy()

if data_train[KOLOM_TANGGAL].max() >= data_test[KOLOM_TANGGAL].min():
    raise ValueError("Pembagian data tidak kronologis.")

if X_train.isna().any().any() or X_test.isna().any().any():
    raise ValueError("Terdapat nilai kosong pada fitur.")

if y_train.isna().any() or y_test.isna().any():
    raise ValueError("Terdapat nilai kosong pada target.")

print("Jumlah data pemodelan:", len(df_model))
print("Jumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))
print("Tanggal latih:", data_train[KOLOM_TANGGAL].min(), "sampai", data_train[KOLOM_TANGGAL].max())
print("Tanggal uji:", data_test[KOLOM_TANGGAL].min(), "sampai", data_test[KOLOM_TANGGAL].max())
print("Fitur model:", FITUR_MODEL)

df_model.to_csv(
    FOLDER_OUTPUT / "06_data_pemodelan.csv",
    index=False
)

data_train.to_csv(
    FOLDER_OUTPUT / "06_data_latih.csv",
    index=False
)

data_test.to_csv(
    FOLDER_OUTPUT / "06_data_uji.csv",
    index=False
)

model_awal = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model_awal.fit(X_train, y_train)

y_pred_awal = model_awal.predict(X_test)

mse_awal = mean_squared_error(y_test, y_pred_awal)
rmse_awal = np.sqrt(mse_awal)
mae_awal = mean_absolute_error(y_test, y_pred_awal)
r2_awal = r2_score(y_test, y_pred_awal)

print("Evaluasi model awal:")
print("RMSE:", rmse_awal)
print("MAE:", mae_awal)
print("MSE:", mse_awal)
print("R2:", r2_awal)

hasil_prediksi_awal = data_test[
    ["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]
].copy()

hasil_prediksi_awal["prediksi_xgboost_awal"] = y_pred_awal
hasil_prediksi_awal["selisih_prediksi"] = (
    hasil_prediksi_awal[KOLOM_TARGET]
    - hasil_prediksi_awal["prediksi_xgboost_awal"]
)

hasil_prediksi_awal.to_csv(
    FOLDER_OUTPUT / "07_prediksi_model_awal.csv",
    index=False
)

metrik_model_awal = {
    "RMSE": float(rmse_awal),
    "MAE": float(mae_awal),
    "MSE": float(mse_awal),
    "R2": float(r2_awal)
}

with open(
    FOLDER_OUTPUT / "07_evaluasi_model_awal.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrik_model_awal, file, indent=4, ensure_ascii=False)

joblib.dump(
    model_awal,
    FOLDER_OUTPUT / "07_model_xgboost_awal.joblib"
)

parameter_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0]
}

cv_time_series = TimeSeriesSplit(n_splits=3)

model_tuning = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)

grid_search = GridSearchCV(
    estimator=model_tuning,
    param_grid=parameter_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv_time_series,
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=True,
    error_score="raise"
)

grid_search.fit(X_train, y_train)

model_terbaik = grid_search.best_estimator_
y_pred_tuning = model_terbaik.predict(X_test)

rmse_cv_terbaik = -grid_search.best_score_

mse_tuning = mean_squared_error(y_test, y_pred_tuning)
rmse_tuning = np.sqrt(mse_tuning)
mae_tuning = mean_absolute_error(y_test, y_pred_tuning)
r2_tuning = r2_score(y_test, y_pred_tuning)

print("Parameter terbaik:", grid_search.best_params_)
print("RMSE validasi:", rmse_cv_terbaik)
print("Evaluasi model tuning:")
print("RMSE:", rmse_tuning)
print("MAE:", mae_tuning)
print("MSE:", mse_tuning)
print("R2:", r2_tuning)

hasil_prediksi_tuning = data_test[
    ["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]
].copy()

hasil_prediksi_tuning["prediksi_xgboost_tuning"] = y_pred_tuning
hasil_prediksi_tuning["selisih_prediksi"] = (
    hasil_prediksi_tuning[KOLOM_TARGET]
    - hasil_prediksi_tuning["prediksi_xgboost_tuning"]
)

hasil_prediksi_tuning.to_csv(
    FOLDER_OUTPUT / "08_prediksi_model_tuning.csv",
    index=False
)

hasil_evaluasi_tuning = {
    "parameter_terbaik": grid_search.best_params_,
    "RMSE_validasi_terbaik": float(rmse_cv_terbaik),
    "RMSE": float(rmse_tuning),
    "MAE": float(mae_tuning),
    "MSE": float(mse_tuning),
    "R2": float(r2_tuning)
}

with open(
    FOLDER_OUTPUT / "08_evaluasi_model_tuning.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(hasil_evaluasi_tuning, file, indent=4, ensure_ascii=False)

pd.DataFrame(grid_search.cv_results_).to_csv(
    FOLDER_OUTPUT / "08_hasil_grid_search.csv",
    index=False
)

joblib.dump(
    model_terbaik,
    FOLDER_OUTPUT / "08_model_xgboost_terbaik.joblib"
)

df_prediksi_semua = df.sort_values(
    KOLOM_TANGGAL
).reset_index(drop=True).copy()

tanggal_awal_test = data_test[KOLOM_TANGGAL].min()

X_semua = df_prediksi_semua[FITUR_MODEL].copy()
prediksi_semua = model_terbaik.predict(X_semua)

mask_sebelum_test = (
    df_prediksi_semua[KOLOM_TANGGAL] < tanggal_awal_test
)

riwayat = (
    df_prediksi_semua.loc[mask_sebelum_test, KOLOM_TARGET]
    .astype(float)
    .tolist()
)

prediksi_recursive = []

for indeks in range(len(data_test)):
    baris_test = data_test.iloc[indeks]

    fitur_hari_ini = {
        "tahun": baris_test["tahun"],
        "bulan": baris_test["bulan"],
        "hari": baris_test["hari"],
        "hari_dalam_minggu": baris_test["hari_dalam_minggu"],
        "minggu_ke": baris_test["minggu_ke"],
        "akhir_pekan": baris_test["akhir_pekan"],
        "lag_1": riwayat[-1],
        "lag_7": riwayat[-7],
        "rolling_mean_7": sum(riwayat[-7:]) / 7,
        "rolling_mean_14": sum(riwayat[-14:]) / 14
    }

    X_hari_ini = pd.DataFrame(
        [fitur_hari_ini],
        columns=FITUR_MODEL
    )

    hasil_hari_ini = float(
        model_terbaik.predict(X_hari_ini)[0]
    )

    prediksi_recursive.append(hasil_hari_ini)
    riwayat.append(hasil_hari_ini)

    if (indeks + 1) % 30 == 0 or indeks == 0:
        print(
            f"Prediksi ke-{indeks + 1} | "
            f"Tanggal: {baris_test[KOLOM_TANGGAL].date()} | "
            f"Prediksi: {hasil_hari_ini:.2f}"
        )

indeks_data_uji = df_prediksi_semua.index[
    df_prediksi_semua[KOLOM_TANGGAL].isin(
        data_test[KOLOM_TANGGAL]
    )
]

if len(indeks_data_uji) != len(prediksi_recursive):
    raise ValueError("Jumlah prediksi recursive tidak sesuai.")

prediksi_semua[indeks_data_uji] = prediksi_recursive

hasil_prediksi_semua = df_prediksi_semua[
    ["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]
].copy()

hasil_prediksi_semua["prediksi_xgboost_terbaik"] = prediksi_semua
hasil_prediksi_semua["status_prediksi"] = "Prediksi in-sample"

hasil_prediksi_semua.loc[
    hasil_prediksi_semua[KOLOM_TANGGAL] < data_train[KOLOM_TANGGAL].min(),
    "status_prediksi"
] = "Fitur historis belum lengkap"

hasil_prediksi_semua.loc[
    hasil_prediksi_semua[KOLOM_TANGGAL].isin(data_test[KOLOM_TANGGAL]),
    "status_prediksi"
] = "Prediksi berantai"

hasil_prediksi_semua["selisih_prediksi"] = (
    hasil_prediksi_semua[KOLOM_TARGET]
    - hasil_prediksi_semua["prediksi_xgboost_terbaik"]
)

hasil_prediksi_semua["kesalahan_absolut"] = (
    hasil_prediksi_semua["selisih_prediksi"].abs()
)

if len(hasil_prediksi_semua) != len(df):
    raise ValueError("Jumlah hasil prediksi tidak sama dengan data awal.")

if hasil_prediksi_semua["prediksi_xgboost_terbaik"].isna().any():
    raise ValueError("Terdapat hasil prediksi kosong.")

hasil_prediksi_semua.to_csv(
    FOLDER_OUTPUT / "09_prediksi_seluruh_data_recursive.csv",
    index=False
)

y_aktual_recursive = data_test[KOLOM_TARGET].reset_index(drop=True)
y_pred_recursive = pd.Series(prediksi_recursive)

mse_recursive = mean_squared_error(
    y_aktual_recursive,
    y_pred_recursive
)

rmse_recursive = np.sqrt(mse_recursive)
mae_recursive = mean_absolute_error(
    y_aktual_recursive,
    y_pred_recursive
)

r2_recursive = r2_score(
    y_aktual_recursive,
    y_pred_recursive
)

print("Evaluasi prediksi berantai:")
print("RMSE:", rmse_recursive)
print("MAE:", mae_recursive)
print("MSE:", mse_recursive)
print("R2:", r2_recursive)

hasil_evaluasi_recursive = data_test[
    ["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]
].copy()

hasil_evaluasi_recursive["prediksi_recursive"] = y_pred_recursive.values

hasil_evaluasi_recursive["selisih_prediksi"] = (
    hasil_evaluasi_recursive[KOLOM_TARGET]
    - hasil_evaluasi_recursive["prediksi_recursive"]
)

hasil_evaluasi_recursive["kesalahan_absolut"] = (
    hasil_evaluasi_recursive["selisih_prediksi"].abs()
)

metrik_recursive = {
    "RMSE": float(rmse_recursive),
    "MAE": float(mae_recursive),
    "MSE": float(mse_recursive),
    "R2": float(r2_recursive)
}

with open(
    FOLDER_OUTPUT / "10_evaluasi_prediksi_recursive.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrik_recursive, file, indent=4, ensure_ascii=False)

hasil_evaluasi_recursive.to_csv(
    FOLDER_OUTPUT / "10_hasil_evaluasi_recursive.csv",
    index=False
)

df_analisis = hasil_evaluasi_recursive.copy()

df_analisis[KOLOM_TANGGAL] = pd.to_datetime(
    df_analisis[KOLOM_TANGGAL]
)

df_analisis["bias_prediksi"] = (
    df_analisis["prediksi_recursive"]
    - df_analisis[KOLOM_TARGET]
)

df_analisis["bulan"] = (
    df_analisis[KOLOM_TANGGAL].dt.to_period("M").astype(str)
)

analisis_bulanan = (
    df_analisis.groupby("bulan")
    .agg(
        jumlah_data=(KOLOM_TANGGAL, "count"),
        rata_rata_aktual=(KOLOM_TARGET, "mean"),
        rata_rata_prediksi=("prediksi_recursive", "mean"),
        MAE=("kesalahan_absolut", "mean"),
        RMSE=(
            "bias_prediksi",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        rata_rata_bias=("bias_prediksi", "mean")
    )
    .reset_index()
)

print("Analisis kesalahan per bulan:")
print(
    analisis_bulanan.to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)

kesalahan_terbesar = (
    df_analisis.sort_values(
        "kesalahan_absolut",
        ascending=False
    )
    .head(10)
    [[
        KOLOM_TANGGAL,
        KOLOM_TARGET,
        "prediksi_recursive",
        "bias_prediksi",
        "kesalahan_absolut"
    ]]
    .copy()
)

print("Kesalahan prediksi terbesar:")
print(
    kesalahan_terbesar.to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)

bias_rata_rata = float(df_analisis["bias_prediksi"].mean())
bias_absolut_rata_rata = float(df_analisis["bias_prediksi"].abs().mean())

jumlah_overpredict = int(
    (df_analisis["bias_prediksi"] > 0).sum()
)

jumlah_underpredict = int(
    (df_analisis["bias_prediksi"] < 0).sum()
)

jumlah_prediksi_tepat = int(
    (df_analisis["bias_prediksi"] == 0).sum()
)

if bias_rata_rata > 0:
    kecenderungan = "Overpredict"
elif bias_rata_rata < 0:
    kecenderungan = "Underpredict"
else:
    kecenderungan = "Tidak memiliki bias rata-rata"

ringkasan_bias = {
    "jumlah_data_evaluasi": int(len(df_analisis)),
    "rata_rata_bias_prediksi": bias_rata_rata,
    "rata_rata_bias_absolut": bias_absolut_rata_rata,
    "jumlah_overpredict": jumlah_overpredict,
    "jumlah_underpredict": jumlah_underpredict,
    "jumlah_prediksi_tepat": jumlah_prediksi_tepat,
    "kecenderungan_model": kecenderungan
}

analisis_bulanan.to_csv(
    FOLDER_OUTPUT / "11_analisis_kesalahan_recursive.csv",
    index=False
)

kesalahan_terbesar.to_csv(
    FOLDER_OUTPUT / "11_kesalahan_prediksi_terbesar.csv",
    index=False
)

with open(
    FOLDER_OUTPUT / "11_ringkasan_kesalahan_recursive.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(ringkasan_bias, file, indent=4, ensure_ascii=False)

file_model_final = (
    FOLDER_OUTPUT / "12_model_xgboost_final.joblib"
)

joblib.dump(model_terbaik, file_model_final)

fitur_model = list(X_train.columns)
parameter_final = model_terbaik.get_params()

metadata_model = {
    "nama_model": "XGBoost Regression",
    "nama_file_model": file_model_final.name,
    "jenis_prediksi": "Recursive time-series prediction",
    "jumlah_fitur": len(fitur_model),
    "daftar_fitur": fitur_model,
    "parameter_model": {
        "n_estimators": int(parameter_final["n_estimators"]),
        "max_depth": int(parameter_final["max_depth"]),
        "learning_rate": float(parameter_final["learning_rate"]),
        "subsample": float(parameter_final["subsample"]),
        "objective": parameter_final["objective"]
    },
    "hasil_evaluasi_recursive": {
        "RMSE": float(rmse_recursive),
        "MAE": float(mae_recursive),
        "MSE": float(mse_recursive),
        "R2": float(r2_recursive)
    },
    "periode_pengujian": {
        "tanggal_awal": str(data_test[KOLOM_TANGGAL].min().date()),
        "tanggal_akhir": str(data_test[KOLOM_TANGGAL].max().date()),
        "jumlah_data_uji": int(len(data_test))
    }
}

with open(
    FOLDER_OUTPUT / "12_metadata_model_final.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata_model, file, indent=4, ensure_ascii=False)

model_final_verifikasi = joblib.load(file_model_final)

if model_final_verifikasi.n_features_in_ != len(X_train.columns):
    raise ValueError("Jumlah fitur model final tidak sesuai.")

X_verifikasi = X_test.iloc[:5]

prediksi_model_asli = model_terbaik.predict(X_verifikasi)
prediksi_model_simpan = model_final_verifikasi.predict(X_verifikasi)

if not np.isfinite(prediksi_model_simpan).all():
    raise ValueError("Prediksi model final mengandung NaN atau infinity.")

if not np.allclose(
    prediksi_model_asli,
    prediksi_model_simpan,
    rtol=1e-5,
    atol=1e-5
):
    raise ValueError("Prediksi model tersimpan berbeda dari model asli.")

file_wajib = [
    "02_data_terurut.csv",
    "03_target_validasi.csv",
    "04_data_fitur_kalender.csv",
    "05_data_fitur_historis.csv",
    "06_data_pemodelan.csv",
    "06_data_latih.csv",
    "06_data_uji.csv",
    "07_prediksi_model_awal.csv",
    "07_evaluasi_model_awal.json",
    "07_model_xgboost_awal.joblib",
    "08_prediksi_model_tuning.csv",
    "08_evaluasi_model_tuning.json",
    "08_hasil_grid_search.csv",
    "08_model_xgboost_terbaik.joblib",
    "09_prediksi_seluruh_data_recursive.csv",
    "10_evaluasi_prediksi_recursive.json",
    "10_hasil_evaluasi_recursive.csv",
    "11_analisis_kesalahan_recursive.csv",
    "11_kesalahan_prediksi_terbesar.csv",
    "11_ringkasan_kesalahan_recursive.json",
    "12_model_xgboost_final.joblib",
    "12_metadata_model_final.json"
]

file_ada = [
    nama_file
    for nama_file in file_wajib
    if (FOLDER_OUTPUT / nama_file).exists()
]

file_tidak_ada = [
    nama_file
    for nama_file in file_wajib
    if not (FOLDER_OUTPUT / nama_file).exists()
]

laporan_pemeriksaan = {
    "jumlah_file_wajib": len(file_wajib),
    "jumlah_file_tersedia": len(file_ada),
    "jumlah_file_tidak_tersedia": len(file_tidak_ada),
    "file_tidak_tersedia": file_tidak_ada,
    "jumlah_fitur_model": int(model_final_verifikasi.n_features_in_),
    "jumlah_fitur_data": int(len(X_train.columns)),
    "verifikasi_model": "Berhasil",
    "verifikasi_prediksi": "Berhasil",
    "status_pemeriksaan": (
        "Lengkap" if not file_tidak_ada else "Belum lengkap"
    )
}

with open(
    FOLDER_OUTPUT / "13_laporan_pemeriksaan_final.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(laporan_pemeriksaan, file, indent=4, ensure_ascii=False)

print("Pemeriksaan akhir:")
print("File wajib:", len(file_wajib))
print("File tersedia:", len(file_ada))
print("File tidak tersedia:", len(file_tidak_ada))
print("Status:", laporan_pemeriksaan["status_pemeriksaan"])

if file_tidak_ada:
    print("File yang belum tersedia:")
    for nama_file in file_tidak_ada:
        print("-", nama_file)

print("Program selesai.")