from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from io import BytesIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import io
import zipfile
import shutil

# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="XGBoost Regression Transjakarta",
    page_icon="🚌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
FOLDER_DATASET = BASE_DIR / "dataset"

file_dataset = FOLDER_DATASET / "dataset_transjakarta.xlsx"

FOLDER_OUTPUT = BASE_DIR / "hasil_xgboost"

KOLOM_TANGGAL = "tanggal"
KOLOM_TARGET = "jumlah_penumpang_per_hari"


# ============================================================
# DAFTAR FILE HASIL TRAINING
# ============================================================

FILE_HASIL = {
    "Data terurut": "02_data_terurut.csv",
    "Target validasi": "03_target_validasi.csv",
    "Fitur kalender": "04_data_fitur_kalender.csv",
    "Fitur historis": "05_data_fitur_historis.csv",
    "Data pemodelan": "06_data_pemodelan.csv",
    "Data latih": "06_data_latih.csv",
    "Data uji": "06_data_uji.csv",
    "Prediksi model awal": "07_prediksi_model_awal.csv",
    "Evaluasi model awal": "07_evaluasi_model_awal.json",
    "Model XGBoost awal": "07_model_xgboost_awal.joblib",
    "Prediksi model tuning": "08_prediksi_model_tuning.csv",
    "Evaluasi model tuning": "08_evaluasi_model_tuning.json",
    "Hasil Grid Search": "08_hasil_grid_search.csv",
    "Model XGBoost terbaik": "08_model_xgboost_terbaik.joblib",
    "Prediksi seluruh data": "09_prediksi_seluruh_data_recursive.csv",
    "Evaluasi recursive": "10_evaluasi_prediksi_recursive.json",
    "Hasil evaluasi recursive": "10_hasil_evaluasi_recursive.csv",
    "Analisis kesalahan bulanan": "11_analisis_kesalahan_recursive.csv",
    "Kesalahan terbesar": "11_kesalahan_prediksi_terbesar.csv",
    "Ringkasan kesalahan": "11_ringkasan_kesalahan_recursive.json",
    "Model final": "12_model_xgboost_final.joblib",
    "Metadata model final": "12_metadata_model_final.json",
    "Laporan pemeriksaan": "13_laporan_pemeriksaan_final.json",
}


# ============================================================
# FUNGSI MEMBACA FILE HASIL
# ============================================================

def baca_csv(nama_file):
    path = FOLDER_OUTPUT / nama_file

    if not path.exists():
        return None

    return pd.read_csv(path)


def baca_json(nama_file):
    path = FOLDER_OUTPUT / nama_file

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def tampilkan_csv(nama_file):
    data = baca_csv(nama_file)

    if data is None:
        st.warning(f"File belum tersedia: {nama_file}")
        return

    st.caption(
        f"Jumlah baris: {len(data):,} | "
        f"Jumlah kolom: {len(data.columns):,}"
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label=f"Unduh {nama_file}",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name=nama_file,
        mime="text/csv",
        key=f"download_{nama_file}",
    )


def tampilkan_json(nama_file):
    data = baca_json(nama_file)

    if data is None:
        st.warning(f"File belum tersedia: {nama_file}")
        return

    st.json(data)


def tampilkan_metrik(data):
    if not data:
        st.warning("Data evaluasi belum tersedia.")
        return

    kolom = st.columns(4)

    daftar_metrik = ["RMSE", "MAE", "MSE", "R2"]

    for kolom_ui, nama_metrik in zip(kolom, daftar_metrik):
        nilai = data.get(nama_metrik)

        if nilai is not None:
            kolom_ui.metric(
                label=nama_metrik,
                value=f"{nilai:,.4f}",
            )



# ============================================================
# FUNGSI PIPELINE TRAINING (TERINTEGRASI DI APP.PY)
# ============================================================

FITUR_KALENDER = [
    "hari_dalam_minggu_sin",
    "hari_dalam_minggu_cos",
    "hari_tahun_sin",
    "hari_tahun_cos",
    "akhir_pekan"
]

FITUR_HISTORIS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14"
]

FITUR_MODEL = FITUR_KALENDER + FITUR_HISTORIS

def buat_fitur_kalender(tanggal):
    tanggal = pd.Timestamp(tanggal)

    hari_dalam_minggu = tanggal.dayofweek
    hari_ke = tanggal.dayofyear

    return {
        "hari_dalam_minggu_sin": np.sin(
            2 * np.pi * hari_dalam_minggu / 7
        ),
        "hari_dalam_minggu_cos": np.cos(
            2 * np.pi * hari_dalam_minggu / 7
        ),
        "hari_tahun_sin": np.sin(
            2 * np.pi * hari_ke / 365.25
        ),
        "hari_tahun_cos": np.cos(
            2 * np.pi * hari_ke / 365.25
        ),
        "akhir_pekan": int(hari_dalam_minggu >= 5)
    }


def simpan_csv(data, nama_file):
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(FOLDER_OUTPUT / nama_file, index=False, encoding="utf-8-sig")


def simpan_json(data, nama_file):
    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(FOLDER_OUTPUT / nama_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def hitung_metrik(y_asli, y_prediksi):
    mse = mean_squared_error(y_asli, y_prediksi)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_asli, y_prediksi)),
        "MSE": float(mse),
        "R2": float(r2_score(y_asli, y_prediksi)),
    }


def jalankan_prapemrosesan(df_asli):
    df = df_asli.copy()

    if df.empty:
        raise ValueError("Dataset kosong.")

    if KOLOM_TANGGAL not in df.columns:
        raise KeyError(
            f"Kolom tanggal '{KOLOM_TANGGAL}' tidak ditemukan."
        )

    if KOLOM_TARGET not in df.columns:
        raise KeyError(
            f"Kolom target '{KOLOM_TARGET}' tidak ditemukan."
        )

    if "jenis_moda" not in df.columns:
        raise KeyError(
            "Kolom 'jenis_moda' tidak ditemukan."
        )

    df[KOLOM_TANGGAL] = pd.to_datetime(df[KOLOM_TANGGAL], errors="coerce")
    if df[KOLOM_TANGGAL].isna().any():
        raise ValueError("Terdapat tanggal tidak valid.")

    df = df.sort_values(KOLOM_TANGGAL).reset_index(drop=True)
    if df[KOLOM_TANGGAL].duplicated().any():
        raise ValueError("Terdapat tanggal duplikat dalam dataset.")

    df.insert(0, "nomor_urut", range(1, len(df) + 1))

    # ========================================================
    # VALIDASI JENIS MODA
    # ========================================================

    df["jenis_moda"] = (
        df["jenis_moda"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["jenis_moda"] == "transjakarta"
    ].copy()

    if df.empty:
        raise ValueError(
            "Tidak terdapat data dengan jenis_moda Transjakarta."
        )

    df = df.sort_values(
        KOLOM_TANGGAL
    ).reset_index(drop=True)

    simpan_csv(df, "02_data_terurut.csv")

    target_asli = df[KOLOM_TARGET].copy()
    target_teks = target_asli.astype(str).str.strip()
    target_teks = target_teks.str.replace(r"\.0+$", "", regex=True)
    target_teks = target_teks.str.replace(r"[.,\s]", "", regex=True)
    mask_valid = target_teks.str.fullmatch(r"\d+", na=False)
    df[KOLOM_TARGET] = pd.to_numeric(target_teks.where(mask_valid), errors="coerce")

    if df[KOLOM_TARGET].isna().any():
        invalid = df.loc[df[KOLOM_TARGET].isna(), ["nomor_urut", KOLOM_TANGGAL]].copy()
        invalid["nilai_asli"] = target_asli[df[KOLOM_TARGET].isna()]
        raise ValueError(
            "Terdapat nilai target tidak valid. Contoh baris:\n"
            + invalid.head(10).to_string(index=False)
        )

    simpan_csv(df, "03_target_validasi.csv")

    fitur_kalender = pd.DataFrame(
        [
            buat_fitur_kalender(tanggal)
            for tanggal in df[KOLOM_TANGGAL]
        ],
            index=df.index
    )

    df = pd.concat(
        [df, fitur_kalender],
        axis=1
    )

    selisih_tanggal = (
        df[KOLOM_TANGGAL]
        .diff()
        .dropna()
    )

    if not selisih_tanggal.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Dataset harus memiliki catatan harian berurutan "
            "tanpa tanggal yang hilang."
        )

    simpan_csv(df, "04_data_fitur_kalender.csv")

    df["lag_1"] = df[KOLOM_TARGET].shift(1)
    df["lag_7"] = df[KOLOM_TARGET].shift(7)
    df["rolling_mean_7"] = df[KOLOM_TARGET].shift(1).rolling(window=7).mean()
    df["rolling_mean_14"] = df[KOLOM_TARGET].shift(1).rolling(window=14).mean()
    simpan_csv(df, "05_data_fitur_historis.csv")

    df_model = df.dropna(subset=FITUR_MODEL + [KOLOM_TARGET]).copy()
    df_model = df_model.sort_values(KOLOM_TANGGAL).reset_index(drop=True)

    batas = int(len(df_model) * 0.80)
    data_train = df_model.iloc[:batas].copy()
    data_test = df_model.iloc[batas:].copy()

    if data_train.empty or data_test.empty:
        raise ValueError("Data tidak cukup untuk pembagian 80:20.")
    if data_train[KOLOM_TANGGAL].max() >= data_test[KOLOM_TANGGAL].min():
        raise ValueError("Pembagian data tidak kronologis.")

    X_train = data_train[FITUR_MODEL].copy()
    X_test = data_test[FITUR_MODEL].copy()
    y_train = data_train[KOLOM_TARGET].copy()
    y_test = data_test[KOLOM_TARGET].copy()

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Terdapat nilai kosong pada fitur.")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Terdapat nilai kosong pada target.")

    simpan_csv(df_model, "06_data_pemodelan.csv")
    simpan_csv(data_train, "06_data_latih.csv")
    simpan_csv(data_test, "06_data_uji.csv")

    return {
        "df": df,
        "df_model": df_model,
        "data_train": data_train,
        "data_test": data_test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def jalankan_model_awal(data):
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
    model.fit(data["X_train"], data["y_train"])
    pred = model.predict(data["X_test"])
    metrik = hitung_metrik(data["y_test"], pred)

    hasil = data["data_test"][["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]].copy()
    hasil["prediksi_xgboost_awal"] = pred
    hasil["selisih_prediksi"] = hasil[KOLOM_TARGET] - hasil["prediksi_xgboost_awal"]

    simpan_csv(hasil, "07_prediksi_model_awal.csv")
    simpan_json(metrik, "07_evaluasi_model_awal.json")
    joblib.dump(model, FOLDER_OUTPUT / "07_model_xgboost_awal.joblib")
    return model, metrik



def jalankan_tuning(data):
    # ========================================================
    # GRID SEARCH UNTUK PREDIKSI ONE-STEP-AHEAD
    # ========================================================

    parameter_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [1, 3],
    }

    cv = TimeSeriesSplit(n_splits=3)

    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
        error_score="raise"
    )

    # Tuning hanya menggunakan data training
    grid.fit(
        data["X_train"],
        data["y_train"]
    )

    # Model terbaik hasil validasi silang
    model = grid.best_estimator_

    # ========================================================
    # EVALUASI ONE-STEP-AHEAD PADA DATA TEST
    # ========================================================

    pred = model.predict(data["X_test"])

    metrik = hitung_metrik(
        data["y_test"],
        pred
    )

    metrik["jenis_evaluasi"] = "Recursive multi-step"
    metrik["jenis_evaluasi"] = "One-step-ahead"
    metrik["metode_tuning"] = "GridSearchCV"
    metrik["metode_prediksi"] = "Prediksi berantai"
    metrik["metode_validasi"] = "TimeSeriesSplit"
    metrik["jumlah_fold"] = 3
    metrik["jumlah_kombinasi_parameter"] = len(
        grid.cv_results_["params"]
    )
    metrik["RMSE_validasi_terbaik"] = float(
        -grid.best_score_
    )
    metrik["parameter_terbaik"] = grid.best_params_
    metrik["sumber_fitur_historis"] = (
    "Nilai aktual pada awal periode uji "
    "dan hasil prediksi pada periode berikutnya"
    )

    hasil = data["data_test"][
        ["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]
    ].copy()

    hasil["prediksi_xgboost_tuning"] = pred
    hasil["jenis_evaluasi"] = "One-step-ahead"
    hasil["selisih_prediksi"] = (
        hasil[KOLOM_TARGET]
        - hasil["prediksi_xgboost_tuning"]
    )

    # ========================================================
    # MENYIMPAN HASIL
    # ========================================================

    simpan_csv(
        hasil,
        "08_prediksi_model_tuning.csv"
    )

    simpan_json(
        metrik,
        "08_evaluasi_model_tuning.json"
    )

    hasil_grid = pd.DataFrame(
        grid.cv_results_
    ).sort_values(
        "rank_test_score"
    )

    simpan_csv(
        hasil_grid,
        "08_hasil_grid_search.csv"
    )

    joblib.dump(
        model,
        FOLDER_OUTPUT / "08_model_xgboost_terbaik.joblib"
    )

    return grid, model, metrik

# ============================================================
# FUNGSI PREDIKSI 7 HARI KE DEPAN
# ============================================================


def prediksi_7_hari_ke_depan(
    model,
    df_historis,
    tanggal_mulai=None
):
    df_hist = df_historis[
        [KOLOM_TANGGAL, KOLOM_TARGET]
    ].copy()

    df_hist[KOLOM_TANGGAL] = pd.to_datetime(
        df_hist[KOLOM_TANGGAL],
        errors="coerce"
    )

    df_hist[KOLOM_TARGET] = pd.to_numeric(
        df_hist[KOLOM_TARGET],
        errors="coerce"
    )

    df_hist = df_hist.dropna(
        subset=[KOLOM_TANGGAL, KOLOM_TARGET]
    )

    df_hist = df_hist.sort_values(
        KOLOM_TANGGAL
    ).reset_index(drop=True)

    if df_hist.empty:
        raise ValueError("Data historis kosong.")

    if df_hist[KOLOM_TANGGAL].duplicated().any():
        raise ValueError(
            "Terdapat tanggal duplikat pada data historis."
        )

    # Memastikan tanggal berurutan setiap hari
    selisih_tanggal = (
        df_hist[KOLOM_TANGGAL]
        .diff()
        .dropna()
    )

    if not selisih_tanggal.eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "Data historis harus berurutan setiap hari "
            "tanpa tanggal yang hilang."
        )

    riwayat = (
        df_hist[KOLOM_TARGET]
        .astype(float)
        .tolist()
    )

    if len(riwayat) < 14:
        raise ValueError(
            "Data historis minimal 14 hari diperlukan."
        )

    # Menentukan tanggal awal prediksi
    tanggal_terakhir = df_hist[KOLOM_TANGGAL].max()

    if tanggal_mulai is None:
        tanggal_mulai = (
            tanggal_terakhir + pd.Timedelta(days=1)
        )
    else:
        tanggal_mulai = pd.Timestamp(tanggal_mulai)

    if tanggal_mulai <= tanggal_terakhir:
        raise ValueError(
            "Tanggal awal prediksi harus setelah "
            "tanggal terakhir dataset."
        )


    hasil_prediksi = []

    # Menghitung jumlah hari yang harus diprediksi
    # sebelum mencapai tanggal awal yang diminta
    hari_pemanasan = (
        tanggal_mulai - tanggal_terakhir
    ).days - 1

    total_hari_prediksi = hari_pemanasan + 7

    # Prediksi seluruh hari secara berurutan.
    # Prediksi sebelum tanggal mulai hanya digunakan
    # untuk membentuk fitur historis.
    for hari_ke in range(total_hari_prediksi):

        tanggal = (
            tanggal_terakhir
            + pd.Timedelta(days=hari_ke + 1)
        )

        fitur_hari_ini = buat_fitur_kalender(
            tanggal
        )

        fitur_hari_ini.update({
            "lag_1": riwayat[-1],
            "lag_7": riwayat[-7],
            "rolling_mean_7": np.mean(riwayat[-7:]),
            "rolling_mean_14": np.mean(riwayat[-14:])
        })

        X_hari_ini = pd.DataFrame(
            [fitur_hari_ini],
            columns=FITUR_MODEL
        )

        prediksi_mentah = float(
            model.predict(X_hari_ini)[0]
        )

        nilai_prediksi = max(
            0,
            prediksi_mentah
        )

        nilai_prediksi = int(
            round(nilai_prediksi)
        )

        # Hanya menyimpan tujuh hari sesuai
        # tanggal yang diminta pengguna
        if tanggal >= tanggal_mulai:
            hasil_prediksi.append({
                "tanggal": tanggal,
                "hari": tanggal.strftime("%A"),
                "prediksi_mentah": prediksi_mentah,
                "prediksi_jumlah_penumpang": nilai_prediksi
            })

        # Prediksi hari ini menjadi histori
        # untuk pembentukan fitur hari berikutnya
        riwayat.append(nilai_prediksi)

    hasil = pd.DataFrame(
        hasil_prediksi
    )

    simpan_csv(
        hasil,
        "14_prediksi_7_hari_ke_depan.csv"
    )

    return hasil


def jalankan_model_terbaik(data, model_terbaik):
    df = data["df"].sort_values(KOLOM_TANGGAL).reset_index(drop=True).copy()
    data_train = data["data_train"]
    data_test = data["data_test"]

    tanggal_awal_test = data_test[KOLOM_TANGGAL].min()
    X_semua = df[FITUR_MODEL].copy()
    prediksi_semua = model_terbaik.predict(X_semua)

    mask_sebelum_test = df[KOLOM_TANGGAL] < tanggal_awal_test
    riwayat = df.loc[mask_sebelum_test, KOLOM_TARGET].astype(float).tolist()
    if len(riwayat) < 14:
        raise ValueError("Riwayat sebelum data uji kurang dari 14 hari.")

    prediksi_recursive = []
    for indeks in range(len(data_test)):
        baris = data_test.iloc[indeks]
        fitur_hari_ini = buat_fitur_kalender(
            baris[KOLOM_TANGGAL]
        )

        fitur_hari_ini.update({
            "lag_1": riwayat[-1],
            "lag_7": riwayat[-7:][0],
            "rolling_mean_7": np.mean(riwayat[-7:]),
            "rolling_mean_14": np.mean(riwayat[-14:])
        })
        X_hari_ini = pd.DataFrame([fitur_hari_ini], columns=FITUR_MODEL)
        nilai_prediksi = float(model_terbaik.predict(X_hari_ini)[0])
        prediksi_recursive.append(nilai_prediksi)
        riwayat.append(nilai_prediksi)

    indeks_uji = df.index[df[KOLOM_TANGGAL].isin(data_test[KOLOM_TANGGAL])]
    if len(indeks_uji) != len(prediksi_recursive):
        raise ValueError("Jumlah prediksi recursive tidak sesuai.")
    prediksi_semua[indeks_uji] = prediksi_recursive

    hasil_semua = df[["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]].copy()
    hasil_semua["prediksi_xgboost_terbaik"] = prediksi_semua
    hasil_semua["status_prediksi"] = "Prediksi in-sample"
    hasil_semua.loc[
        hasil_semua[KOLOM_TANGGAL] < data_train[KOLOM_TANGGAL].min(),
        "status_prediksi"
    ] = "Fitur historis belum lengkap"
    hasil_semua.loc[
        hasil_semua[KOLOM_TANGGAL].isin(data_test[KOLOM_TANGGAL]),
        "status_prediksi"
    ] = "Prediksi berantai"
    hasil_semua["selisih_prediksi"] = (
        hasil_semua[KOLOM_TARGET] - hasil_semua["prediksi_xgboost_terbaik"]
    )
    hasil_semua["kesalahan_absolut"] = hasil_semua["selisih_prediksi"].abs()

    if len(hasil_semua) != len(df) or hasil_semua["prediksi_xgboost_terbaik"].isna().any():
        raise ValueError("Hasil prediksi seluruh data tidak valid.")
    simpan_csv(hasil_semua, "09_prediksi_seluruh_data_recursive.csv")

    y_aktual = data_test[KOLOM_TARGET].reset_index(drop=True)
    y_pred = pd.Series(prediksi_recursive)
    metrik = hitung_metrik(y_aktual, y_pred)

    evaluasi = data_test[["nomor_urut", KOLOM_TANGGAL, KOLOM_TARGET]].copy()
    evaluasi["prediksi_recursive"] = y_pred.values
    evaluasi["selisih_prediksi"] = evaluasi[KOLOM_TARGET] - evaluasi["prediksi_recursive"]
    evaluasi["kesalahan_absolut"] = evaluasi["selisih_prediksi"].abs()
    simpan_json(metrik, "10_evaluasi_prediksi_recursive.json")
    simpan_csv(evaluasi, "10_hasil_evaluasi_recursive.csv")

    analisis = evaluasi.copy()
    analisis[KOLOM_TANGGAL] = pd.to_datetime(analisis[KOLOM_TANGGAL])
    analisis["bias_prediksi"] = analisis["prediksi_recursive"] - analisis[KOLOM_TARGET]
    analisis["bulan"] = analisis[KOLOM_TANGGAL].dt.to_period("M").astype(str)

    bulanan = (
        analisis.groupby("bulan")
        .agg(
            jumlah_data=(KOLOM_TANGGAL, "count"),
            rata_rata_aktual=(KOLOM_TARGET, "mean"),
            rata_rata_prediksi=("prediksi_recursive", "mean"),
            MAE=("kesalahan_absolut", "mean"),
            RMSE=("bias_prediksi", lambda x: np.sqrt(np.mean(x ** 2))),
            rata_rata_bias=("bias_prediksi", "mean"),
        )
        .reset_index()
    )
    terbesar = (
        analisis.sort_values("kesalahan_absolut", ascending=False)
        .head(10)[[
            KOLOM_TANGGAL, KOLOM_TARGET, "prediksi_recursive",
            "bias_prediksi", "kesalahan_absolut"
        ]]
        .copy()
    )

    bias_rata = float(analisis["bias_prediksi"].mean())
    if bias_rata > 0:
        kecenderungan = "Overpredict"
    elif bias_rata < 0:
        kecenderungan = "Underpredict"
    else:
        kecenderungan = "Tidak memiliki bias rata-rata"

    ringkasan = {
        "jumlah_data_evaluasi": int(len(analisis)),
        "rata_rata_bias_prediksi": bias_rata,
        "rata_rata_bias_absolut": float(analisis["bias_prediksi"].abs().mean()),
        "jumlah_overpredict": int((analisis["bias_prediksi"] > 0).sum()),
        "jumlah_underpredict": int((analisis["bias_prediksi"] < 0).sum()),
        "jumlah_prediksi_tepat": int((analisis["bias_prediksi"] == 0).sum()),
        "kecenderungan_model": kecenderungan,
    }
    simpan_csv(bulanan, "11_analisis_kesalahan_recursive.csv")
    simpan_csv(terbesar, "11_kesalahan_prediksi_terbesar.csv")
    simpan_json(ringkasan, "11_ringkasan_kesalahan_recursive.json")

    # ============================================================
    # PELATIHAN MODEL FINAL MENGGUNAKAN SELURUH DATA
    # ============================================================

    parameter = model_terbaik.get_params()

    model_final = XGBRegressor(**parameter)

    model_final.fit(
        data["df_model"][FITUR_MODEL],
        data["df_model"][KOLOM_TARGET]
    )

    file_model = FOLDER_OUTPUT / "12_model_xgboost_final.joblib"

    joblib.dump(model_final, file_model)
    metadata = {
            "nama_model": "XGBoost Regression",
            "nama_file_model": file_model.name,
            "jenis_prediksi": "Recursive time-series prediction",
            "jumlah_fitur": len(FITUR_MODEL),
            "daftar_fitur": FITUR_MODEL,
            "parameter_model": {
                "n_estimators": int(parameter["n_estimators"]),
                "max_depth": int(parameter["max_depth"]),
                "learning_rate": float(parameter["learning_rate"]),
                "subsample": float(parameter["subsample"]),
                "objective": parameter["objective"],
            },
            "hasil_evaluasi_recursive": metrik,
            "periode_pengujian": {
                "tanggal_awal": str(data_test[KOLOM_TANGGAL].min().date()),
                "tanggal_akhir": str(data_test[KOLOM_TANGGAL].max().date()),
                "jumlah_data_uji": int(len(data_test)),
            },
        }
    simpan_json(metadata, "12_metadata_model_final.json")

    model_simpan = joblib.load(file_model)
    if model_simpan.n_features_in_ != len(FITUR_MODEL):
        raise ValueError("Jumlah fitur model final tidak sesuai.")
    
    # Verifikasi model final yang telah disimpan
    if not np.allclose(
        model_final.predict(data["X_test"].iloc[:5]),
        model_simpan.predict(data["X_test"].iloc[:5]),
        rtol=1e-5,
        atol=1e-5
    ):
        raise ValueError(
            "Prediksi model tersimpan berbeda dari model final."
        )

    # ============================================================
    # PREDIKSI 7 HARI KE DEPAN
    # ============================================================

    file_wajib = [
        "02_data_terurut.csv", "03_target_validasi.csv",
        "04_data_fitur_kalender.csv", "05_data_fitur_historis.csv",
        "06_data_pemodelan.csv", "06_data_latih.csv", "06_data_uji.csv",
        "07_prediksi_model_awal.csv", "07_evaluasi_model_awal.json",
        "07_model_xgboost_awal.joblib", "08_prediksi_model_tuning.csv",
        "08_evaluasi_model_tuning.json", "08_hasil_grid_search.csv",
        "08_model_xgboost_terbaik.joblib",
        "09_prediksi_seluruh_data_recursive.csv",
        "10_evaluasi_prediksi_recursive.json", "10_hasil_evaluasi_recursive.csv",
        "11_analisis_kesalahan_recursive.csv",
        "11_kesalahan_prediksi_terbesar.csv",
        "11_ringkasan_kesalahan_recursive.json",
        "12_model_xgboost_final.joblib", "12_metadata_model_final.json",
    ]
    missing = [name for name in file_wajib if not (FOLDER_OUTPUT / name).exists()]
    laporan = {
        "jumlah_file_wajib": len(file_wajib),
        "jumlah_file_tersedia": len(file_wajib) - len(missing),
        "jumlah_file_tidak_tersedia": len(missing),
        "file_tidak_tersedia": missing,
        "jumlah_fitur_model": int(model_simpan.n_features_in_),
        "jumlah_fitur_data": len(FITUR_MODEL),
        "verifikasi_model": "Berhasil",
        "verifikasi_prediksi": "Berhasil",
        "status_pemeriksaan": "Lengkap" if not missing else "Belum lengkap",
    }
    simpan_json(laporan, "13_laporan_pemeriksaan_final.json")
    return metrik, laporan

# ============================================================
# FUNGSI REKOMENDASI LAYANAN
# ============================================================

def hitung_rekomendasi_layanan(
    prediksi_penumpang,
    kapasitas_bus,
    headway_menit,
    waktu_putaran_menit,
    armada_tersedia
):

    if prediksi_penumpang < 0:
        raise ValueError(
            "Prediksi jumlah penumpang tidak boleh negatif."
        )

    if kapasitas_bus <= 0:
        raise ValueError(
            "Kapasitas bus harus lebih dari 0."
        )

    if headway_menit <= 0:
        raise ValueError(
            "Headway harus lebih dari 0."
        )

    if waktu_putaran_menit <= 0:
        raise ValueError(
            "Waktu putaran harus lebih dari 0."
        )

    if armada_tersedia < 0:
        raise ValueError(
            "Jumlah armada tersedia tidak boleh negatif."
        )

    # ========================================================
    # KEBUTUHAN ARMADA BERDASARKAN WAKTU PUTARAN DAN HEADWAY
    # ========================================================

    kebutuhan_armada = int(
        np.ceil(
            waktu_putaran_menit
            / headway_menit
        )
    )

    selisih_armada = (
        armada_tersedia
        - kebutuhan_armada
    )

    if selisih_armada >= 0:
        status_armada = (
            "Armada tersedia mencukupi"
        )
    else:
        status_armada = (
            "Armada tersedia belum mencukupi"
        )

    # ========================================================
    # PERHITUNGAN KAPASITAS LAYANAN
    # ========================================================

    jumlah_perjalanan_per_bus = (
        waktu_putaran_menit
        / headway_menit
    )

    estimasi_kapasitas_layanan = (
        kebutuhan_armada
        * kapasitas_bus
    )

    if (
        prediksi_penumpang
        <= estimasi_kapasitas_layanan
    ):
        status_kapasitas = (
            "Kapasitas layanan mencukupi"
        )
    else:
        status_kapasitas = (
            "Kapasitas layanan belum mencukupi"
        )

    return {
        "prediksi_penumpang": int(
            prediksi_penumpang
        ),
        "kapasitas_bus": int(
            kapasitas_bus
        ),
        "headway_menit": float(
            headway_menit
        ),
        "waktu_putaran_menit": float(
            waktu_putaran_menit
        ),
        "armada_tersedia": int(
            armada_tersedia
        ),
        "kebutuhan_armada": int(
            kebutuhan_armada
        ),
        "selisih_armada": int(
            selisih_armada
        ),
        "estimasi_kapasitas_layanan": int(
            estimasi_kapasitas_layanan
        ),
        "status_armada": status_armada,
        "status_kapasitas": status_kapasitas,
        "peran_xgboost": (
            "XGBoost digunakan untuk memprediksi "
            "jumlah penumpang. Kebutuhan armada "
            "dihitung menggunakan parameter "
            "operasional."
        ),
    }

# ============================================================
# UPLOAD DATASET SEBELUM NAVIGASI
# ============================================================

if st.session_state.get("df_dataset_upload") is None:

    st.title("XGBoost Regression")
    st.subheader("Prediksi Jumlah Penumpang Transjakarta")

    st.write(
        "Silakan upload dataset Excel terlebih dahulu "
        "untuk membuka navigasi dan memulai proses training."
    )

    file_dataset_awal = st.file_uploader(
        "Upload Dataset Excel",
        type=["xlsx", "xls"],
        key="upload_dataset_awal"
    )

    # ============================================================
    # DOWNLOAD DATASET CONTOH
    # ============================================================

    file_dataset = FOLDER_DATASET / "dataset_transjakarta.xlsx"

    if file_dataset.exists():
        with open(file_dataset, "rb") as file:
            st.markdown(
                """
                ### Belum memiliki dataset?

                Anda dapat mengunduh template excel Transjakarta
                untuk mencoba proses training dan prediksi
                jumlah penumpang selama 7 hari ke depan.

                Unduh dataset di bawah ini, kemudian membaca catatan yang berada di
                excel ini, terakhir unggah kembali melalui kolom Upload Dataset Excel.

                Download disini:
                """
            )

            file_dataset = FOLDER_DATASET / "dataset_transjakarta.xlsx"
            st.download_button(
                label="Download Template Dataset Transjakarta",
                data=file.read(),
                file_name="dataset_transjakarta.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("File dataset contoh belum tersedia.")

    if file_dataset_awal is not None:

        try:
            upload_bytes_awal = file_dataset_awal.getvalue()
            df_upload_awal = pd.read_excel(file_dataset_awal)

            kolom_wajib = [
                KOLOM_TANGGAL,
                KOLOM_TARGET,
                "jenis_moda"
            ]

            kolom_tidak_ada = [
                kolom
                for kolom in kolom_wajib
                if kolom not in df_upload_awal.columns
            ]

            if df_upload_awal.empty:
                st.error("Dataset Excel tidak memiliki data.")

            elif kolom_tidak_ada:
                st.error(
                    "Dataset tidak sesuai. Kolom yang belum tersedia: "
                    + ", ".join(kolom_tidak_ada)
                )

            else:
                # Bersihkan hasil training dari dataset sebelumnya
                shutil.rmtree(FOLDER_OUTPUT, ignore_errors=True)

                # Simpan dataset ke session state
                st.session_state["dataset_uploaded"] = upload_bytes_awal
                st.session_state["dataset_name"] = file_dataset_awal.name
                st.session_state["df_dataset_upload"] = df_upload_awal

                # Reset status seluruh tahapan training
                for kunci in [
                    "tahap_preprocessing_selesai",
                    "tahap_model_awal_selesai",
                    "tahap_tuning_selesai",
                    "tahap_model_terbaik_selesai",
                ]:
                    st.session_state[kunci] = False

                # Hapus data dan model dari proses sebelumnya
                for kunci in [
                    "data_pipeline",
                    "model_awal",
                    "metrik_model_awal",
                    "grid_search",
                    "model_tuning",
                    "metrik_model_tuning",
                    "metrik_model_final",
                    "laporan_model_final",
                ]:
                    st.session_state.pop(kunci, None)

                st.success(
                    f"Dataset berhasil dimuat: {file_dataset_awal.name}"
                )

                st.rerun()

        except Exception as error:
            st.error(f"Dataset tidak dapat dibaca: {error}")

    st.stop()

# ============================================================
# UI KARTU KONFIGURASI FITUR MODEL
# ============================================================

def tampilkan_kartu_fitur():

    st.markdown("""
    <style>
    .feature-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 17px;
        min-height: 125px;
        margin-bottom: 12px;
        transition: .2s ease;
    }

    .feature-card:hover {
        border-color: #00AEEF;
        transform: translateY(-3px);
    }

    .feature-number {
        font-size: 11px;
        color: #00AEEF;
        font-weight: 700;
    }

    .feature-name {
        font-size: 16px;
        font-weight: 700;
        margin: 9px 0;
    }

    .feature-code {
        font-size: 11px;
        opacity: .7;
        overflow-wrap: anywhere;
    }

    .feature-category {
        font-size: 11px;
        margin-top: 10px;
        opacity: .8;
    }
    </style>
    """, unsafe_allow_html=True)

    fitur = [
        ("Hari Sin", "hari_dalam_minggu_sin", "Kalender"),
        ("Hari Cos", "hari_dalam_minggu_cos", "Kalender"),
        ("Tahun Sin", "hari_tahun_sin", "Kalender"),
        ("Tahun Cos", "hari_tahun_cos", "Kalender"),
        ("Akhir Pekan", "akhir_pekan", "Kalender"),
        ("Lag 1", "lag_1", "Historis"),
        ("Lag 7", "lag_7", "Historis"),
        ("Rolling Mean 7", "rolling_mean_7", "Historis"),
        ("Rolling Mean 14", "rolling_mean_14", "Historis"),
    ]

    st.subheader("Konfigurasi Fitur Model")
    st.caption(
        "Sembilan fitur prediktor yang digunakan "
        "dalam model XGBoost Regression."
    )

    kolom = st.columns(3)

    for i, (nama, kode, kategori) in enumerate(fitur):

        with kolom[i % 3]:

            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-number">
                    FEATURE {i + 1:02d}
                </div>

                <div class="feature-name">
                    {nama}
                </div>

                <div class="feature-code">
                    {kode}
                </div>

                <div class="feature-category">
                    ● {kategori}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.info("Total fitur model: 9 fitur aktif")

# ============================================================
# SIDEBAR DAN NAVIGASI
# ============================================================

st.sidebar.title("XGBoost Regression")
st.sidebar.caption("Prediksi Jumlah Penumpang Transjakarta")

st.markdown("""
<style>

/* Hilangkan elemen lingkaran radio BaseWeb */
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
    width: 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Hilangkan radio dari struktur label */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radio"] {
    display: none !important;
}

/* Rapikan posisi teks menu */
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    gap: 0 !important;
}

/* Background sidebar biru Transjakarta */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #002B50 0%,
        #003B70 55%,
        #00529B 100%
    ) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

/* Warna teks sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

/* Jarak antar menu */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 7px;
}

/* Menu tidak aktif */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background-color: rgba(180, 180, 180, 0.50) !important;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 10px;
    padding: 10px 14px;
    transition: all 0.2s ease;
}

/* Hilangkan lingkaran radio */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Hilangkan input radio */
[data-testid="stSidebar"] input[type="radio"] {
    display: none !important;
}

/* Menu aktif */
[data-testid="stSidebar"] label:has(input:checked) {
    background: #00AEEF !important;
    border: 1px solid #00AEEF !important;
    border-left: 4px solid #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.20);
}

/* Hover menu */
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background-color: rgba(255, 255, 255, 0.20) !important;
}

/* Hover menu aktif */
[data-testid="stSidebar"] label:has(input:checked):hover {
    background: #008FC7 !important;
}

/* ==========================================
   UI UTAMA - XGBOOST REGRESSION
   Navigasi sidebar tidak diubah
========================================== */

/* Background halaman utama */
[data-testid="stMain"] {
    background:
        radial-gradient(
            circle at top right,
            rgba(0, 174, 239, 0.10),
            transparent 35%
        ),
        #0E1117;
}

/* Area konten */
[data-testid="stMain"] .block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Judul halaman */
[data-testid="stMain"] h1 {
    color: #F5F9FF;
    font-size: 2.2rem;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin-bottom: 0.5rem;
}

/* Subjudul */
[data-testid="stMain"] h2 {
    color: #E2ECF7;
    font-weight: 650;
    margin-top: 1.5rem;
}

/* Judul bagian */
[data-testid="stMain"] h3 {
    color: #D7E8F8;
    font-weight: 600;
}

/* Teks utama */
[data-testid="stMain"] p {
    color: #B8C7D9;
    line-height: 1.7;
}

/* Kartu metrik */
[data-testid="stMain"] [data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #1A2635,
        #151D29
    );
    border: 1px solid rgba(0, 174, 239, 0.20);
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
}

/* Label metrik */
[data-testid="stMain"] [data-testid="stMetricLabel"] {
    color: #9FB5CC;
    font-size: 0.9rem;
}

/* Nilai metrik */
[data-testid="stMain"] [data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-weight: 700;
}

/* Kotak informasi */
[data-testid="stMain"] [data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid rgba(0, 174, 239, 0.25);
}

/* Tombol konten utama saja */
[data-testid="stMain"] .stButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    transition: all 0.2s ease;
}

/* Tombol utama */
[data-testid="stMain"] .stButton > button[kind="primary"] {
    background: #00AEEF;
    color: #FFFFFF;
    border: 1px solid #00AEEF;
}

/* Hover tombol utama */
[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
    background: #008FC7;
    border-color: #008FC7;
    box-shadow: 0 4px 14px rgba(0, 174, 239, 0.25);
}

/* Tabel */
[data-testid="stMain"] [data-testid="stDataFrame"] {
    border: 1px solid #293747;
    border-radius: 12px;
    overflow: hidden;
}

/* Kotak kode dan path */
[data-testid="stMain"] [data-testid="stCode"] {
    border: 1px solid #293747;
    border-radius: 10px;
}

/* Garis pemisah */
[data-testid="stMain"] hr {
    border-color: #293747;
}

/* Selectbox */
[data-testid="stMain"] [data-testid="stSelectbox"] {
    margin-bottom: 0.5rem;
}

/* Download button */
[data-testid="stMain"] [data-testid="stDownloadButton"] button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER DAN MENU NAVIGASI
# ==================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">🚌</div>
        <div class="sidebar-title">Transjakarta</div>
        <div class="sidebar-subtitle">
            Sistem Prediksi Penumpang Harian
        </div>
    </div>
    """, unsafe_allow_html=True)

    MENU_NAVIGASI = [
        ("🏠 Beranda", "Beranda"),
        ("⚙️ Jalankan Training", "Jalankan Training"),
        ("📁 Dataset Awal", "Dataset Awal"),
        ("🧹 Prapemrosesan", "Prapemrosesan"),
        ("📊 Pembagian Data 80:20", "Pembagian Data 80:20"),
        ("⊖ Model Awal", "Model Awal"),
        ("🔬 Tuning 3-Fold", "Tuning 3-Fold"),
        ("🏆 Model Terbaik", "Model Terbaik"),
        ("📊 Perbandingan Model", "Perbandingan Model"),
        ("🔮 Prediksi Recursive", "Prediksi Recursive"),
        ("🚌 Rekomendasi Layanan", "Rekomendasi Layanan"),
        ("📉 Analisis Kesalahan", "Analisis Kesalahan"),
        ("📂 File Hasil", "File Hasil"),
    ]

    if "navigasi_sidebar" not in st.session_state:
        st.session_state["navigasi_sidebar"] = "Beranda"

    with st.sidebar:
        for i, (label, nilai_menu) in enumerate(MENU_NAVIGASI):
            if st.button(
                label,
                key=f"nav_menu_{i}",
                use_container_width=True,
                type=(
                    "primary"
                    if st.session_state["navigasi_sidebar"] == nilai_menu
                    else "secondary"
                ),
            ):
                st.session_state["navigasi_sidebar"] = nilai_menu
                st.rerun()

    menu = st.session_state["navigasi_sidebar"]

# ============================================================
# BERANDA
# ============================================================

if menu == "Beranda":

    st.title("XGBoost Regression")
    st.subheader("Prediksi Jumlah Penumpang Transjakarta")

    st.write(
        "Aplikasi ini digunakan untuk menjalankan proses "
        "XGBoost Regression berdasarkan script train_model.py "
        "dan menampilkan hasil setiap tahap pemodelan."
    )

    st.info(
        "Alur: Dataset → Prapemrosesan → Fitur Historis → "
        "Split 80:20 → Model Awal → Evaluasi → "
        "Tuning 3-Fold → Model Terbaik → Evaluasi Recursive."
    )

    col1, = st.columns(1)

    with col1:
        st.write("**Dataset**")
        if st.session_state.get("dataset_uploaded") is not None:
            st.success(st.session_state.get("dataset_name", "Dataset sudah diunggah"))
        else:
            st.info("Dataset belum diunggah.")

    laporan = baca_json("13_laporan_pemeriksaan_final.json")

    if laporan:
        st.subheader("Status pemeriksaan terakhir")

        st.metric(
            "File tersedia",
            laporan.get("jumlah_file_tersedia", 0),
        )

        st.metric(
            "Status",
            laporan.get("status_pemeriksaan", "Belum tersedia"),
        )


# ============================================================
# JALANKAN TRAINING
# ============================================================

elif menu == "Jalankan Training":

    st.title("Jalankan XGBoost Regression")
    st.write(
        "Unggah dataset Excel, lalu jalankan setiap tahap secara berurutan. "
        "Model Awal, Tuning 3-Fold, dan Model Terbaik dijalankan terpisah."
    )

    df_dataset_upload = st.session_state["df_dataset_upload"]

    if st.session_state.get("df_dataset_upload") is None:
        st.info("Upload dataset Excel terlebih dahulu untuk memulai.")
    else:
        st.divider()
        st.subheader("Tahap 1 — Prapemrosesan dan Pembagian Data 80:20")
        if st.button("Jalankan Prapemrosesan & Split 80:20", type="primary", use_container_width=True):
            try:
                with st.spinner("Memproses dataset dan membagi data secara kronologis..."):
                    data_pipeline = jalankan_prapemrosesan(st.session_state["df_dataset_upload"])
                st.session_state["data_pipeline"] = data_pipeline
                st.session_state["tahap_preprocessing_selesai"] = True
                st.session_state["tahap_model_awal_selesai"] = False
                st.session_state["tahap_tuning_selesai"] = False
                st.session_state["tahap_model_terbaik_selesai"] = False
                st.success(
                    f"Prapemrosesan selesai. Data latih: {len(data_pipeline['X_train']):,}; "
                    f"data uji: {len(data_pipeline['X_test']):,}."
                )
            except Exception as error:
                st.exception(error)

        if st.session_state.get("tahap_preprocessing_selesai"):
            st.success("Tahap 1 selesai — data pemodelan dan split 80:20 tersedia.")

        st.divider()
        st.subheader("Tahap 2 — Model Awal")
        if not st.session_state.get("tahap_preprocessing_selesai"):
            st.warning("Selesaikan Tahap 1 terlebih dahulu.")
        elif st.button("Jalankan Model Awal", use_container_width=True):
            try:
                with st.spinner("Melatih dan mengevaluasi Model Awal..."):
                    model_awal, metrik_awal = jalankan_model_awal(st.session_state["data_pipeline"])
                st.session_state["model_awal"] = model_awal
                st.session_state["metrik_model_awal"] = metrik_awal
                st.session_state["tahap_model_awal_selesai"] = True
                st.session_state["tahap_tuning_selesai"] = False
                st.session_state["tahap_model_terbaik_selesai"] = False
                st.success("Model Awal selesai.")
                tampilkan_metrik(metrik_awal)
            except Exception as error:
                st.exception(error)

        if st.session_state.get("tahap_model_awal_selesai"):
            st.success("Tahap 2 selesai — hasil Model Awal tersimpan.")

        st.divider()
        st.subheader("Tahap 3 — Tuning 3-Fold")
        if not st.session_state.get("tahap_model_awal_selesai"):
            st.warning("Jalankan Model Awal terlebih dahulu.")
        elif st.button("Jalankan Tuning 3-Fold", use_container_width=True):
            try:
                with st.spinner("Tuning berjalan. Grid Search mengevaluasi kombinasi parameter dengan 3 fold..."):
                    grid, model_tuning, metrik_tuning = jalankan_tuning(st.session_state["data_pipeline"])
                st.session_state["grid_search"] = grid
                st.session_state["model_tuning"] = model_tuning
                st.session_state["metrik_model_tuning"] = metrik_tuning
                st.session_state["tahap_tuning_selesai"] = True
                st.session_state["tahap_model_terbaik_selesai"] = False
                st.success("Tuning 3-Fold selesai.")

                tampilkan_metrik(metrik_tuning)

                st.subheader("Parameter Terbaik")

                parameter_terbaik = grid.best_params_

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Learning Rate",
                        parameter_terbaik.get("learning_rate", "-")
                    )

                with col2:
                    st.metric(
                        "Max Depth",
                        parameter_terbaik.get("max_depth", "-")
                    )

                with col3:
                    st.metric(
                        "N Estimators",
                        parameter_terbaik.get("n_estimators", "-")
                    )

                with col4:
                    st.metric(
                        "Subsample",
                        parameter_terbaik.get("subsample", "-")
                    )

            except Exception as error:
                st.exception(error)


        if st.session_state.get("tahap_tuning_selesai"):
            st.success(
                "Tahap 3 selesai — model hasil tuning "
                "dan parameter terbaik tersimpan."
            )

            metrik_awal = st.session_state.get(
                "metrik_model_awal"
            )

            metrik_tuning = st.session_state.get(
                "metrik_model_tuning"
            )

            if (
                isinstance(metrik_awal, dict)
                and isinstance(metrik_tuning, dict)
            ):

                st.divider()
                st.subheader(
                    "Perbandingan Model Sebelum dan Sesudah Tuning"
                )

                daftar_metrik = [
                    "RMSE",
                    "MAE",
                    "MSE",
                    "R2",
                ]

                data_perbandingan = []

                for nama_metrik in daftar_metrik:

                    nilai_awal = float(
                        metrik_awal[nama_metrik]
                    )

                    nilai_tuning = float(
                        metrik_tuning[nama_metrik]
                    )

                    perubahan = nilai_tuning - nilai_awal

                    if nama_metrik == "R2":

                        status = (
                            "Membaik"
                            if perubahan > 0
                            else "Memburuk"
                            if perubahan < 0
                            else "Tidak berubah"
                        )

                    else:

                        status = (
                            "Membaik"
                            if perubahan < 0
                            else "Memburuk"
                            if perubahan > 0
                            else "Tidak berubah"
                        )

                    data_perbandingan.append({
                        "Metrik": nama_metrik,
                        "Model Awal": nilai_awal,
                        "Tuning 3-Fold": nilai_tuning,
                        "Perubahan": perubahan,
                        "Status": status,
                    })

                df_perbandingan = pd.DataFrame(
                    data_perbandingan
                )

                st.dataframe(
                    df_perbandingan.style.format({
                        "Model Awal": "{:.12f}",
                        "Tuning 3-Fold": "{:.12f}",
                        "Perubahan": "{:+.12f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "RMSE, MAE, dan MSE lebih baik jika nilainya "
                    "lebih rendah. R² lebih baik jika nilainya "
                    "lebih tinggi."
                )

                st.divider()

                # Grafik RMSE
                st.markdown("#### Perbandingan RMSE")

                st.bar_chart(
                    pd.DataFrame({
                        "Model Awal": [
                            float(metrik_awal["RMSE"])
                        ],
                        "Tuning 3-Fold": [
                            float(metrik_tuning["RMSE"])
                        ],
                    }, index=["RMSE"]),
                    use_container_width=True,
                )

                # Grafik MAE
                st.markdown("#### Perbandingan MAE")

                st.bar_chart(
                    pd.DataFrame({
                        "Model Awal": [
                            float(metrik_awal["MAE"])
                        ],
                        "Tuning 3-Fold": [
                            float(metrik_tuning["MAE"])
                        ],
                    }, index=["MAE"]),
                    use_container_width=True,
                )

                # Grafik MSE
                st.markdown("#### Perbandingan MSE")

                st.bar_chart(
                    pd.DataFrame({
                        "Model Awal": [
                            float(metrik_awal["MSE"])
                        ],
                        "Tuning 3-Fold": [
                            float(metrik_tuning["MSE"])
                        ],
                    }, index=["MSE"]),
                    use_container_width=True,
                )

                # Grafik R2
                st.markdown("#### Perbandingan R²")

                st.bar_chart(
                    pd.DataFrame({
                        "Model Awal": [
                            float(metrik_awal["R2"])
                        ],
                        "Tuning 3-Fold": [
                            float(metrik_tuning["R2"])
                        ],
                    }, index=["R²"]),
                    use_container_width=True,
                )

            else:
                st.info(
                    "Perbandingan belum tersedia. "
                    "Jalankan Model Awal dan Tuning 3-Fold "
                    "terlebih dahulu."
                )

        st.divider()
        st.subheader("Tahap 4 — Model Terbaik dan Prediksi Recursive")

        if not st.session_state.get("tahap_tuning_selesai"):
            st.warning("Selesaikan Tuning 3-Fold terlebih dahulu.")
        elif st.button("Jalankan Model Terbaik", use_container_width=True):
            try:
                with st.spinner("Menjalankan prediksi recursive, evaluasi, dan pemeriksaan model final..."):
                    metrik_final, laporan_final = jalankan_model_terbaik(
                        st.session_state["data_pipeline"],
                        st.session_state["model_tuning"]
                    )
                st.session_state["metrik_model_final"] = metrik_final
                st.session_state["laporan_model_final"] = laporan_final
                st.session_state["tahap_model_terbaik_selesai"] = True
                st.subheader("Laporan Pemeriksaan Model")

                if laporan_final:

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "File Wajib",
                            laporan_final.get("jumlah_file_wajib", "-")
                        )

                    with col2:
                        st.metric(
                            "File Tersedia",
                            laporan_final.get("jumlah_file_tersedia", "-")
                        )

                    with col3:
                        st.metric(
                            "File Tidak Tersedia",
                            laporan_final.get("jumlah_file_tidak_tersedia", "-")
                        )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Fitur Model",
                            laporan_final.get("jumlah_fitur_model", "-")
                        )

                    with col2:
                        st.metric(
                            "Fitur Data",
                            laporan_final.get("jumlah_fitur_data", "-")
                        )

                    with col3:
                        st.metric(
                            "Status Pemeriksaan",
                            laporan_final.get("status_pemeriksaan", "-")
                        )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(
                            f"Verifikasi Model: "
                            f"{laporan_final.get('verifikasi_model', '-')}"
                        )

                    with col2:
                        st.success(
                            f"Verifikasi Prediksi: "
                            f"{laporan_final.get('verifikasi_prediksi', '-')}"
                        )

                    file_tidak_tersedia = laporan_final.get(
                        "file_tidak_tersedia", []
                    )

                    if file_tidak_tersedia:
                        st.warning("Terdapat file yang belum tersedia.")

                        st.dataframe(
                            pd.DataFrame({
                                "File Tidak Tersedia": file_tidak_tersedia
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("Seluruh file yang dibutuhkan tersedia.")
                
            except Exception as error:
                st.exception(error)

        if st.session_state.get("tahap_model_terbaik_selesai"):
            st.success("Semua tahap selesai. File hasil dapat dilihat pada menu File Hasil.")

elif menu == "Model Terbaik":

    st.title("Model Terbaik")
    st.write(
        "Halaman ini menampilkan hasil evaluasi model XGBoost "
        "terbaik setelah proses Tuning 3-Fold dan prediksi recursive."
    )

    if not st.session_state.get("tahap_model_terbaik_selesai"):

        st.info(
            "Model Terbaik belum dijalankan. "
            "Buka menu Jalankan Training, selesaikan Tuning 3-Fold, "
            "kemudian klik Jalankan Model Terbaik."
        )

    else:

        st.subheader("Evaluasi Model Terbaik")

        metrik_final = st.session_state.get("metrik_model_final")

        if metrik_final is None:
            metrik_final = baca_json(
                "10_evaluasi_prediksi_recursive.json"
            )

        if metrik_final:
            tampilkan_metrik(metrik_final)
        else:
            st.warning("File evaluasi Model Terbaik belum tersedia.")

        st.divider()

        st.subheader("Parameter Model Terbaik")

        metadata = baca_json("12_metadata_model_final.json")

        if metadata:
            parameter = metadata.get("parameter_model", {})

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "N Estimators",
                    parameter.get("n_estimators", "-")
                )

            with col2:
                st.metric(
                    "Max Depth",
                    parameter.get("max_depth", "-")
                )

            with col3:
                st.metric(
                    "Learning Rate",
                    parameter.get("learning_rate", "-")
                )

            with col4:
                st.metric(
                    "Subsample",
                    parameter.get("subsample", "-")
                )

            st.caption(
                f"Objective: {parameter.get('objective', '-')}"
            )

        else:
            st.warning("Metadata model final belum tersedia.")

        st.divider()

        st.subheader("Laporan Pemeriksaan Model")

        laporan_final = st.session_state.get("laporan_model_final")

        if laporan_final is None:
            laporan_final = baca_json(
                "13_laporan_pemeriksaan_final.json"
            )

        if laporan_final:

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "File Wajib",
                    laporan_final.get(
                        "jumlah_file_wajib",
                        "-"
                    )
                )

            with col2:
                st.metric(
                    "File Tersedia",
                    laporan_final.get(
                        "jumlah_file_tersedia",
                        "-"
                    )
                )

            with col3:
                st.metric(
                    "File Tidak Tersedia",
                    laporan_final.get(
                        "jumlah_file_tidak_tersedia",
                        "-"
                    )
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Fitur Model",
                    laporan_final.get(
                        "jumlah_fitur_model",
                        "-"
                    )
                )

            with col2:
                st.metric(
                    "Fitur Data",
                    laporan_final.get(
                        "jumlah_fitur_data",
                        "-"
                    )
                )

            with col3:
                status = laporan_final.get(
                    "status_pemeriksaan",
                    "-"
                )

                if status == "Lengkap":
                    st.success(
                        f"Status Pemeriksaan: {status}"
                    )
                else:
                    st.warning(
                        f"Status Pemeriksaan: {status}"
                    )

            col1, col2 = st.columns(2)

            with col1:
                verifikasi_model = laporan_final.get(
                    "verifikasi_model",
                    "-"
                )

                if verifikasi_model == "Berhasil":
                    st.success(
                        f"Verifikasi Model: {verifikasi_model}"
                    )
                else:
                    st.warning(
                        f"Verifikasi Model: {verifikasi_model}"
                    )

            with col2:
                verifikasi_prediksi = laporan_final.get(
                    "verifikasi_prediksi",
                    "-"
                )

                if verifikasi_prediksi == "Berhasil":
                    st.success(
                        f"Verifikasi Prediksi: {verifikasi_prediksi}"
                    )
                else:
                    st.warning(
                        f"Verifikasi Prediksi: {verifikasi_prediksi}"
                    )

            file_tidak_tersedia = laporan_final.get(
                "file_tidak_tersedia",
                []
            )

            if file_tidak_tersedia:
                st.warning("Terdapat file yang belum tersedia.")

                st.dataframe(
                    pd.DataFrame({
                        "File Tidak Tersedia":
                            file_tidak_tersedia
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "Seluruh file yang dibutuhkan tersedia."
                )

        else:
            st.warning(
                "Laporan pemeriksaan belum tersedia."
            )

# ============================================================
# DATASET AWAL
# ============================================================

elif menu == "Dataset Awal":

    st.title("Dataset Awal")

    st.write(
        "Dataset awal merupakan data yang digunakan sebagai masukan "
        "sebelum masuk ke tahap prapemrosesan."
    )

    # Membuat template dataset kosong
    template_dataset = pd.DataFrame(columns=[
        "periode_data",
        "tanggal",
        "jenis_moda",
        "jumlah_penumpang_per_hari"
    ])

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        template_dataset.to_excel(
            writer,
            index=False,
            sheet_name="Data"
        )

    output.seek(0)

    st.subheader("Keterangan Kolom")

    keterangan_kolom = pd.DataFrame({
        "Kolom": [
            "periode_data",
            "tanggal",
            "jenis_moda",
            "jumlah_penumpang_per_hari"
        ],
        "Keterangan": [
            "Periode data yang menunjukkan periode atau bulan data.",
            "Tanggal data jumlah penumpang.",
            "Jenis moda transportasi yang digunakan.",
            "Jumlah penumpang Transjakarta pada tanggal tersebut."
        ]
    })

    st.dataframe(
        keterangan_kolom,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset yang Telah Diunggah")

    df_awal = st.session_state.get("df_dataset_upload")

    if df_awal is None:

        st.info(
            "Dataset belum diunggah. Download template di atas, "
            "isi dataset, kemudian upload melalui menu Jalankan Training."
        )

    else:

        st.write("Data sebelum prapemrosesan")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Jumlah baris",
            f"{len(df_awal):,}"
        )

        col2.metric(
            "Jumlah kolom",
            f"{len(df_awal.columns):,}"
        )

        col3.metric(
            "Nilai kosong",
            f"{int(df_awal.isna().sum().sum()):,}"
        )

        st.subheader("Seluruh data awal")

        st.dataframe(
            df_awal,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Tipe data")

        tipe_data = pd.DataFrame({
            "Kolom": df_awal.columns,
            "Tipe Data": df_awal.dtypes.astype(str).values,
            "Jumlah Kosong": df_awal.isna().sum().values,
        })

        st.dataframe(
            tipe_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PRAPEMROSESAN
# ============================================================

elif menu == "Prapemrosesan":

    st.title("Prapemrosesan Data")

    tahap = st.selectbox(
        "Pilih hasil tahap prapemrosesan",
        [
            "Data terurut",
            "Target validasi",
            "Fitur kalender",
            "Fitur historis",
            "Data pemodelan",
        ],
    )

    file_tahap = {
        "Data terurut": "02_data_terurut.csv",
        "Target validasi": "03_target_validasi.csv",
        "Fitur kalender": "04_data_fitur_kalender.csv",
        "Fitur historis": "05_data_fitur_historis.csv",
        "Data pemodelan": "06_data_pemodelan.csv",
    }

    tampilkan_csv(file_tahap[tahap])


# ============================================================
# PEMBAGIAN DATA 80:20
# ============================================================

elif menu == "Pembagian Data 80:20":

    st.title("Pembagian Data 80:20")

    data_latih = baca_csv("06_data_latih.csv")
    data_uji = baca_csv("06_data_uji.csv")

    if data_latih is None or data_uji is None:
        st.warning("Data latih dan data uji belum tersedia.")

    else:
        jumlah_latih = len(data_latih)
        jumlah_uji = len(data_uji)
        jumlah_total = jumlah_latih + jumlah_uji

        col1, col2, col3 = st.columns(3)

        col1.metric("Data latih", f"{jumlah_latih:,}")
        col2.metric("Data uji", f"{jumlah_uji:,}")
        col3.metric("Total data", f"{jumlah_total:,}")

        st.write(
            f"Proporsi data latih: "
            f"{jumlah_latih / jumlah_total * 100:.2f}%"
        )

        st.write(
            f"Proporsi data uji: "
            f"{jumlah_uji / jumlah_total * 100:.2f}%"
        )

        st.subheader("Data Latih")
        tampilkan_csv("06_data_latih.csv")

        st.subheader("Data Uji")
        tampilkan_csv("06_data_uji.csv")


# ============================================================
# MODEL AWAL
# ============================================================

elif menu == "Model Awal":

    st.title("Pelatihan dan Evaluasi Model Awal")

    st.subheader("Evaluasi Model Awal")

    metrik_awal = baca_json(
        "07_evaluasi_model_awal.json"
    )

    tampilkan_metrik(metrik_awal)

    st.subheader("Detail metrik")

    st.subheader("Hasil Prediksi Model Awal")

    tampilkan_csv(
        "07_prediksi_model_awal.csv"
    )


# ============================================================
# PERBANDINGAN MODEL
# ============================================================

elif menu == "Perbandingan Model":

    st.title("Perbandingan Model Sebelum dan Sesudah Tuning")

    st.write(
        "Halaman ini membandingkan performa Model Awal "
        "dengan model hasil Tuning 3-Fold berdasarkan "
        "metrik evaluasi pada data pengujian yang sama."
    )

    metrik_awal = baca_json(
        "07_evaluasi_model_awal.json"
    )

    metrik_tuning = baca_json(
        "08_evaluasi_model_tuning.json"
    )

    if metrik_awal is None or metrik_tuning is None:

        st.warning(
            "Hasil evaluasi belum tersedia. "
            "Jalankan Model Awal dan Tuning 3-Fold "
            "terlebih dahulu."
        )

    else:

        daftar_metrik = [
            "RMSE",
            "MAE",
            "MSE",
            "R2",
        ]

        nama_model = [
            "Model Awal",
            "Model Tuning 3-Fold",
        ]

        # ----------------------------------------------------
        # RINGKASAN METRIK
        # ----------------------------------------------------

        st.subheader("Ringkasan Performa Model")

        kolom = st.columns(4)

        for kolom_ui, nama_metrik in zip(
            kolom,
            daftar_metrik
        ):

            nilai_awal = float(
                metrik_awal[nama_metrik]
            )

            nilai_tuning = float(
                metrik_tuning[nama_metrik]
            )

            kolom_ui.metric(
                label=nama_metrik,
                value=f"{nilai_tuning:,.4f}",
                delta=f"{nilai_tuning - nilai_awal:,.4f}",
                delta_color=(
                    "inverse"
                    if nama_metrik != "R2"
                    else "normal"
                ),
            )

        st.divider()

        # ----------------------------------------------------
        # TABEL PERBANDINGAN
        # ----------------------------------------------------

        st.subheader("Tabel Perbandingan Sebelum dan Sesudah Tuning")

        data_perbandingan = []

        for nama_metrik in daftar_metrik:

            nilai_awal = float(
                metrik_awal[nama_metrik]
            )

            nilai_tuning = float(
                metrik_tuning[nama_metrik]
            )

            perubahan = nilai_tuning - nilai_awal

            if nama_metrik in ["RMSE", "MAE", "MSE"]:

                if nilai_awal != 0:
                    persentase = (
                        (nilai_awal - nilai_tuning)
                        / nilai_awal
                    ) * 100
                else:
                    persentase = None

                if perubahan < 0:
                    status = "Membaik"
                elif perubahan > 0:
                    status = "Memburuk"
                else:
                    status = "Tidak berubah"

            else:

                persentase = None

                if perubahan > 0:
                    status = "Membaik"
                elif perubahan < 0:
                    status = "Memburuk"
                else:
                    status = "Tidak berubah"

            data_perbandingan.append({
                "Metrik": nama_metrik,
                "Model Awal": nilai_awal,
                "Model Tuning 3-Fold": nilai_tuning,
                "Perubahan": perubahan,
                "Persentase Perbaikan (%)": persentase,
                "Status": status,
            })

        df_perbandingan = pd.DataFrame(
            data_perbandingan
        )

        st.dataframe(
            df_perbandingan,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model Awal": st.column_config.NumberColumn(
                    format="%.12f"
                ),
                "Model Tuning 3-Fold": st.column_config.NumberColumn(
                    format="%.12f"
                ),
                "Perubahan": st.column_config.NumberColumn(
                    format="%.12f"
                ),
                "Persentase Perbaikan (%)": st.column_config.NumberColumn(
                    format="%.6f%%"
                ),
            },
        )

        st.caption(
            "Persentase perbaikan dihitung untuk RMSE, MAE, "
            "dan MSE. Nilai positif menunjukkan penurunan "
            "kesalahan. R² dibandingkan berdasarkan perubahan "
            "nilai koefisien determinasi."
        )

        st.divider()

        # ----------------------------------------------------
        # GRAFIK RMSE
        # ----------------------------------------------------

        st.subheader("Perbandingan RMSE")

        df_rmse = pd.DataFrame({
            "RMSE": [
                float(metrik_awal["RMSE"]),
                float(metrik_tuning["RMSE"]),
            ]
        }, index=nama_model)

        st.bar_chart(
            df_rmse,
            use_container_width=True,
        )

        st.caption(
            "RMSE yang lebih rendah menunjukkan "
            "kesalahan prediksi yang lebih kecil."
        )

        # ----------------------------------------------------
        # GRAFIK MAE
        # ----------------------------------------------------

        st.subheader("Perbandingan MAE")

        df_mae = pd.DataFrame({
            "MAE": [
                float(metrik_awal["MAE"]),
                float(metrik_tuning["MAE"]),
            ]
        }, index=nama_model)

        st.bar_chart(
            df_mae,
            use_container_width=True,
        )

        st.caption(
            "MAE menunjukkan rata-rata besar kesalahan "
            "prediksi tanpa memperhatikan arah kesalahan."
        )

        # ----------------------------------------------------
        # GRAFIK MSE
        # ----------------------------------------------------

        st.subheader("Perbandingan MSE")

        df_mse = pd.DataFrame({
            "MSE": [
                float(metrik_awal["MSE"]),
                float(metrik_tuning["MSE"]),
            ]
        }, index=nama_model)

        st.bar_chart(
            df_mse,
            use_container_width=True,
        )

        st.caption(
            "MSE memberikan penalti lebih besar "
            "terhadap kesalahan prediksi yang besar."
        )

        # ----------------------------------------------------
        # GRAFIK R2
        # ----------------------------------------------------

        st.subheader("Perbandingan R²")

        df_r2 = pd.DataFrame({
            "R²": [
                float(metrik_awal["R2"]),
                float(metrik_tuning["R2"]),
            ]
        }, index=nama_model)

        st.bar_chart(
            df_r2,
            use_container_width=True,
        )

        st.caption(
            "R² yang lebih tinggi menunjukkan "
            "kemampuan model menjelaskan variasi "
            "nilai aktual yang lebih baik."
        )

        st.divider()

        # ----------------------------------------------------
        # DETAIL METRIK LENGKAP
        # ----------------------------------------------------

        st.subheader("Detail Nilai Metrik")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Model Awal")

            metrik1, metrik2 = st.columns(2)

            with metrik1:
                st.metric(
                    "RMSE",
                    f"{metrik_awal['RMSE']:,.2f}"
                )

            with metrik2:
                st.metric(
                    "MAE",
                    f"{metrik_awal['MAE']:,.2f}"
                )

            metrik3, metrik4 = st.columns(2)

            with metrik3:
                st.metric(
                    "MSE",
                    f"{metrik_awal['MSE']:,.2f}"
                )

            with metrik4:
                st.metric(
                    "R²",
                    f"{metrik_awal['R2']:.4f}"
                )


        with col2:

            st.markdown("#### Model Tuning 3-Fold")

            metrik1, metrik2 = st.columns(2)

            with metrik1:
                st.metric(
                    "RMSE",
                    f"{metrik_tuning['RMSE']:,.2f}"
                )

            with metrik2:
                st.metric(
                    "MAE",
                    f"{metrik_tuning['MAE']:,.2f}"
                )

            metrik3, metrik4 = st.columns(2)

            with metrik3:
                st.metric(
                    "MSE",
                    f"{metrik_tuning['MSE']:,.2f}"
                )

            with metrik4:
                st.metric(
                    "R²",
                    f"{metrik_tuning['R2']:.4f}"
                )


# ============================================================
# TUNING 3-FOLD
# ============================================================

elif menu == "Tuning 3-Fold":

    st.title("Hyperparameter Tuning 3-Fold")

    st.write(
        "Tuning menggunakan GridSearchCV dengan "
        "TimeSeriesSplit sebanyak 3 fold. "
        "Parameter terbaik dipilih berdasarkan "
        "rata-rata RMSE validasi."
    )

    hasil_grid = baca_csv(
        "08_hasil_grid_search.csv"
    )

    if hasil_grid is None:
        st.warning("Hasil Grid Search belum tersedia.")

    else:
        st.metric(
            "Jumlah kombinasi parameter",
            f"{len(hasil_grid):,}",
        )

        st.metric(
            "Jumlah fold",
            "3",
        )

        st.metric(
            "Total evaluasi fold",
            f"{len(hasil_grid) * 3:,}",
        )

        # Mengubah skor negatif dari sklearn
        # menjadi nilai RMSE positif
        kolom_fold = [
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
        ]

        hasil_grid["RMSE_Rata_Rata_CV"] = (
            -hasil_grid["mean_test_score"]
        )

        hasil_grid["RMSE_Std_CV"] = (
            hasil_grid["std_test_score"]
        )

        for nomor, kolom in enumerate(kolom_fold, start=1):
            if kolom in hasil_grid.columns:
                hasil_grid[f"RMSE_Fold_{nomor}"] = (
                    -hasil_grid[kolom]
                )

        hasil_grid["Parameter"] = (
            hasil_grid["params"].astype(str)
        )

        tabel_ringkasan = hasil_grid[
            [
                "rank_test_score",
                "Parameter",
                "RMSE_Fold_1",
                "RMSE_Fold_2",
                "RMSE_Fold_3",
                "RMSE_Rata_Rata_CV",
                "RMSE_Std_CV",
            ]
        ].sort_values("rank_test_score")

        st.subheader("Seluruh kombinasi parameter")

        st.dataframe(
            tabel_ringkasan,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Unduh seluruh hasil tuning",
            data=tabel_ringkasan.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="hasil_tuning_3_fold.csv",
            mime="text/csv",
        )

        # Menampilkan setiap kombinasi dan fold
        st.subheader("Detail evaluasi setiap fold")

        detail_fold = []

        for _, baris in hasil_grid.iterrows():

            for nomor, kolom in enumerate(
                kolom_fold,
                start=1,
            ):

                if kolom in hasil_grid.columns:

                    detail_fold.append({
                        "Parameter": str(baris["params"]),
                        "Fold": nomor,
                        "RMSE": -float(baris[kolom]),
                    })

        df_detail_fold = pd.DataFrame(
            detail_fold
        )

        st.dataframe(
            df_detail_fold,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Evaluasi model hasil tuning")

        metrik_tuning = baca_json(
            "08_evaluasi_model_tuning.json"
        )

        tampilkan_metrik(metrik_tuning)

        st.subheader("Prediksi model hasil tuning")

        tampilkan_csv(
            "08_prediksi_model_tuning.csv"
        )


# ============================================================
# MODEL TERBAIK
# ============================================================

elif menu == "Model Terbaik":

    st.title("Model XGBoost Terbaik")

    metadata = baca_json("12_metadata_model_final.json")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Model",
            metadata.get("model", "XGBoost")
        )

    with col2:
        st.metric(
            "Jumlah Data",
            metadata.get("jumlah_data", "-")
        )

    with col3:
        st.metric(
            "Jumlah Fitur",
            metadata.get("jumlah_fitur", "-")
        )

    evaluasi = baca_json(
        "10_evaluasi_prediksi_recursive.json"
    )

    if metadata:
        st.subheader("Informasi Model")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Model",
                metadata.get("model", "XGBoost")
            )

        with col2:
            st.metric(
                "Jumlah Data",
                f"{metadata.get('jumlah_data', 0):,}"
                if isinstance(
                    metadata.get("jumlah_data"),
                    (int, float)
                )
                else metadata.get("jumlah_data", "-")
            )

        with col3:
            st.metric(
                "Jumlah Fitur",
                metadata.get("jumlah_fitur", "-")
            )
    
    st.subheader("Evaluasi prediksi recursive")

    tampilkan_metrik(evaluasi)

    st.subheader("Laporan pemeriksaan model")

    tampilkan_json(
        "13_laporan_pemeriksaan_final.json"
    )


# ============================================================
# PREDIKSI RECURSIVE
# ============================================================

elif menu == "Prediksi Recursive":

    st.title("Prediksi Recursive")

    st.success("UI konfigurasi fitur berhasil dipanggil!")

    tampilkan_kartu_fitur()

    st.write(
        "Halaman ini menampilkan hasil prediksi seluruh "
        "data dan hasil evaluasi pada data uji."
    )

    st.subheader("Prediksi Seluruh Data")

    tampilkan_csv(
        "09_prediksi_seluruh_data_recursive.csv"
    )

    st.subheader("Hasil Evaluasi Data Uji")

    tampilkan_csv(
        "10_hasil_evaluasi_recursive.csv"
    )

    st.subheader("Metrik Evaluasi Recursive")

    metrik_recursive = baca_json(
        "10_evaluasi_prediksi_recursive.json"
    )

    tampilkan_metrik(metrik_recursive)

    if metrik_recursive:
        # Kode metrik recursive yang sudah ada
        st.metric(
            "RMSE",
            f"{metrik_recursive['RMSE']:.2f}"
        )

        # ==========================================
        # PREDIKSI 7 HARI KE DEPAN
        # ==========================================
        st.divider()

        st.subheader("Prediksi 7 Hari ke Depan")

        tanggal_awal_prediksi = st.date_input(
            "Tanggal awal prediksi",
            value=(
                pd.Timestamp.today().date()
            ),
            help=(
                "Tanggal harus setelah tanggal "
                "terakhir pada dataset."
            )
        )

        if st.button(
            "Jalankan Prediksi 7 Hari Kerja",
            type="primary"
        ):
            try:

                model_final = joblib.load(
                    FOLDER_OUTPUT
                    / "12_model_xgboost_final.joblib"
                )

                st.write("Fitur model tersimpan:")
                st.write(model_final.get_booster().feature_names)

                st.write("Fitur yang digunakan aplikasi:")
                st.write(FITUR_MODEL)

                df_historis = baca_csv(
                    "03_target_validasi.csv"
                )

                if df_historis is None:
                    raise ValueError(
                        "Data historis belum tersedia."
                    )

                hasil_prediksi = (
                    prediksi_7_hari_ke_depan(
                        model_final,
                        df_historis,
                        tanggal_mulai=tanggal_awal_prediksi
                    )
                )

                st.success(
                    "Prediksi 7 hari kerja berhasil dibuat."
                )

            except Exception as error:
                st.error(
                    f"Prediksi gagal dijalankan: {error}"
                )

        tampilkan_csv(
            "14_prediksi_7_hari_ke_depan.csv"
        )

        # Tombol Download Prediksi 7 Hari
        file_prediksi = FOLDER_OUTPUT / "14_prediksi_7_hari_ke_depan.csv"

        if file_prediksi.exists():
            with open(file_prediksi, "rb") as file:
                st.download_button(
                    label="Download Prediksi 7 Hari ke Depan",
                    data=file,
                    file_name="prediksi_7_hari_ke_depan.csv",
                    mime="text/csv"
                )

# ============================================================
# ANALISIS KESALAHAN
# ============================================================

elif menu == "Analisis Kesalahan":

    st.title("Analisis Kesalahan Prediksi")

    st.subheader("Analisis Kesalahan Per Bulan")

    tampilkan_csv(
        "11_analisis_kesalahan_recursive.csv"
    )

    st.subheader("10 Kesalahan Prediksi Terbesar")

    tampilkan_csv(
        "11_kesalahan_prediksi_terbesar.csv"
    )

    st.subheader("Ringkasan Bias Prediksi")

    ringkasan_kesalahan = baca_json(
    "11_ringkasan_kesalahan_recursive.json"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MAE",
            f"{ringkasan_kesalahan.get('MAE', 0):,.2f}"
        )

    with col2:
        st.metric(
            "MSE",
            f"{ringkasan_kesalahan.get('MSE', 0):,.2f}"
        )

    with col3:
        st.metric(
            "RMSE",
            f"{ringkasan_kesalahan.get('RMSE', 0):,.2f}"
        )

    with col4:
        st.metric(
            "R²",
            f"{ringkasan_kesalahan.get('R2', 0):.4f}"
        )

# ============================================================
# REKOMENDASI LAYANAN
# ============================================================

elif menu == "Rekomendasi Layanan":

    st.title("Rekomendasi Penyediaan Layanan Bus")

    st.write(
        "Modul ini menggunakan hasil prediksi jumlah "
        "penumpang dan parameter operasional untuk "
        "menghitung kebutuhan layanan bus."
    )

    st.info(
        "XGBoost berperan pada prediksi jumlah penumpang. "
        "Perhitungan kebutuhan armada dilakukan menggunakan "
        "parameter operasional."
    )

    st.subheader("Parameter Operasional")

    kolom1, kolom2 = st.columns(2)

    with kolom1:

        prediksi_penumpang = st.number_input(
            "Prediksi jumlah penumpang",
            min_value=0,
            value=0,
            step=1
        )

        kapasitas_bus = st.number_input(
            "Kapasitas bus",
            min_value=1,
            value=80,
            step=1
        )

        headway_menit = st.number_input(
            "Headway (menit)",
            min_value=1.0,
            value=10.0,
            step=1.0
        )

    with kolom2:

        waktu_putaran_menit = st.number_input(
            "Waktu putaran (menit)",
            min_value=1.0,
            value=60.0,
            step=1.0
        )

        armada_tersedia = st.number_input(
            "Armada tersedia",
            min_value=0,
            value=1,
            step=1
        )

    if st.button(
        "Hitung Rekomendasi Layanan",
        type="primary",
        use_container_width=True
    ):

        try:

            hasil_rekomendasi = (
                hitung_rekomendasi_layanan(
                    prediksi_penumpang=(
                        prediksi_penumpang
                    ),
                    kapasitas_bus=(
                        kapasitas_bus
                    ),
                    headway_menit=(
                        headway_menit
                    ),
                    waktu_putaran_menit=(
                        waktu_putaran_menit
                    ),
                    armada_tersedia=(
                        armada_tersedia
                    )
                )
            )

            st.subheader(
                "Hasil Rekomendasi"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Prediksi Penumpang",
                f"{hasil_rekomendasi['prediksi_penumpang']:,}"
            )

            col2.metric(
                "Kebutuhan Armada",
                f"{hasil_rekomendasi['kebutuhan_armada']:,}"
            )

            col3.metric(
                "Armada Tersedia",
                f"{hasil_rekomendasi['armada_tersedia']:,}"
            )

            st.write(
                "**Status Armada:** "
                + hasil_rekomendasi[
                    "status_armada"
                ]
            )

            st.write(
                "**Status Kapasitas:** "
                + hasil_rekomendasi[
                    "status_kapasitas"
                ]
            )

            st.write(
                "**Estimasi Kapasitas Layanan:** "
                + f"{hasil_rekomendasi['estimasi_kapasitas_layanan']:,}"
                + " penumpang"
            )

        except Exception as error:

            st.error(
                f"Perhitungan gagal: {error}"
            )


# ============================================================
# FILE HASIL
# ============================================================

elif menu == "File Hasil":

    st.title("Daftar File Hasil Training")

    daftar_status = []

    for nama, file in FILE_HASIL.items():

        path = FOLDER_OUTPUT / file

        daftar_status.append({
            "Tahap": nama,
            "Nama File": file,
            "Status": (
                "Tersedia"
                if path.exists()
                else "Belum tersedia"
            ),
        })

    df_status = pd.DataFrame(
        daftar_status
    )

    st.dataframe(
        df_status,
        use_container_width=True,
        hide_index=True,
    )

    jumlah_ada = sum(
        (FOLDER_OUTPUT / file).exists()
        for file in FILE_HASIL.values()
    )

    st.metric(
        "File tersedia",
        f"{jumlah_ada} / {len(FILE_HASIL)}",
    )

    st.subheader("Download Hasil Training")

    if FOLDER_OUTPUT.exists():

        daftar_file = [
            file
            for file in FOLDER_OUTPUT.rglob("*")
            if file.is_file()
        ]

        if daftar_file:

            buffer_zip = io.BytesIO()

            with zipfile.ZipFile(
                buffer_zip,
                "w",
                zipfile.ZIP_DEFLATED
            ) as zip_file:

                for file in daftar_file:
                    zip_file.write(
                        file,
                        arcname=file.relative_to(FOLDER_OUTPUT)
                    )

            buffer_zip.seek(0)

            st.download_button(
                label="Download Folder Hasil Training",
                data=buffer_zip.getvalue(),
                file_name="hasil_xgboost.zip",
                mime="application/zip",
                use_container_width=True
            )

        else:
            st.info("Belum ada file hasil training.")

    else:
        st.info("Folder hasil training belum tersedia.")