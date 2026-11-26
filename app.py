from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
import streamlit as st


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="XGBoost Regression TransJakarta",
    page_icon="🚌",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# Mencari script training yang digunakan
KANDIDAT_SCRIPT = [
    BASE_DIR / "train_model.py",
    BASE_DIR / "train_model(3).py",
]

TRAIN_SCRIPT = next(
    (path for path in KANDIDAT_SCRIPT if path.exists()),
    None,
)

FOLDER_OUTPUT = Path(
    r"D:\Kuliah Semester 9\Skripsi (Data Baru)\hasil_xgboost"
)

PATH_DATASET = Path(
    r"D:\Kuliah Semester 9\Skripsi (Data Baru)\jumlah-penumpang-angkutan-umum-yang-terlayani-per-hari-komponen-data.xlsx"
)

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
# SIDEBAR DAN NAVIGASI
# ============================================================

st.sidebar.title("XGBoost Regression")
st.sidebar.caption("Prediksi Jumlah Penumpang TransJakarta")

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

/* Background sidebar biru TransJakarta */
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
        <div class="sidebar-title">TransJakarta</div>
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
        ("🔮 Prediksi Recursive", "Prediksi Recursive"),
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
    st.subheader("Prediksi Jumlah Penumpang TransJakarta")

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

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Script training**")

        if TRAIN_SCRIPT:
            st.success(TRAIN_SCRIPT.name)
        else:
            st.error("train_model.py belum ditemukan.")

    with col2:
        st.write("**Folder hasil**")
        st.code(str(FOLDER_OUTPUT))

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
        "Halaman ini menjalankan script training yang sama "
        "dengan train_model.py. Proses training tidak "
        "dijalankan ulang ketika kalian berpindah halaman."
    )

    if TRAIN_SCRIPT is None:
        st.error(
            "Script training tidak ditemukan. "
            "Simpan train_model.py di folder yang sama "
            "dengan app.py."
        )

    else:
        st.write("Script:")
        st.code(str(TRAIN_SCRIPT))

        st.write("Dataset:")
        st.code(str(PATH_DATASET))

        st.write("Folder output:")
        st.code(str(FOLDER_OUTPUT))

        if st.button(
            "Mulai Training",
            type="primary",
            use_container_width=True,
        ):

            if not PATH_DATASET.exists():
                st.error("Dataset Excel tidak ditemukan.")

            else:
                FOLDER_OUTPUT.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                area_log = st.empty()
                status = st.empty()

                log_baris = []

                status.info(
                    "Training sedang berjalan. "
                    "Jangan tutup aplikasi."
                )

                try:
                    proses = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            str(TRAIN_SCRIPT),
                        ],
                        cwd=str(BASE_DIR),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )

                    for baris in proses.stdout:
                        log_baris.append(baris.rstrip())

                        area_log.code(
                            "\n".join(log_baris[-100:]),
                            language="text",
                        )

                    kode_keluar = proses.wait()

                    if kode_keluar == 0:
                        status.success(
                            "Training selesai tanpa error."
                        )

                        st.success(
                            "Hasil training berhasil dibuat. "
                            "Buka halaman lain untuk melihat "
                            "hasil setiap tahap."
                        )

                    else:
                        status.error(
                            f"Training gagal. Exit code: {kode_keluar}"
                        )

                except Exception as error:
                    st.exception(error)


# ============================================================
# DATASET AWAL
# ============================================================

elif menu == "Dataset Awal":

    st.title("Dataset Awal")

    if not PATH_DATASET.exists():
        st.error("Dataset Excel tidak ditemukan.")

    else:
        df_awal = pd.read_excel(PATH_DATASET)

        st.write("Data sebelum prapemrosesan")

        col1, col2, col3 = st.columns(3)

        col1.metric("Jumlah baris", f"{len(df_awal):,}")
        col2.metric("Jumlah kolom", f"{len(df_awal.columns):,}")
        col3.metric(
            "Nilai kosong",
            f"{int(df_awal.isna().sum().sum()):,}",
        )

        st.subheader("Seluruh data awal")

        st.dataframe(
            df_awal,
            use_container_width=True,
            hide_index=True,
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
            hide_index=True,
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

    if metrik_awal:
        st.json(metrik_awal)

    st.subheader("Hasil Prediksi Model Awal")

    tampilkan_csv(
        "07_prediksi_model_awal.csv"
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

        if metrik_tuning:
            st.json(metrik_tuning)

        st.subheader("Prediksi model hasil tuning")

        tampilkan_csv(
            "08_prediksi_model_tuning.csv"
        )


# ============================================================
# MODEL TERBAIK
# ============================================================

elif menu == "Model Terbaik":

    st.title("Model XGBoost Terbaik")

    metadata = baca_json(
        "12_metadata_model_final.json"
    )

    evaluasi = baca_json(
        "10_evaluasi_prediksi_recursive.json"
    )

    if metadata:
        st.subheader("Informasi model")

        st.json(metadata)

    st.subheader("Evaluasi prediksi recursive")

    tampilkan_metrik(evaluasi)

    if evaluasi:
        st.json(evaluasi)

    st.subheader("Laporan pemeriksaan model")

    tampilkan_json(
        "13_laporan_pemeriksaan_final.json"
    )


# ============================================================
# PREDIKSI RECURSIVE
# ============================================================

elif menu == "Prediksi Recursive":

    st.title("Prediksi Recursive")

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
        st.json(metrik_recursive)


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

    tampilkan_json(
        "11_ringkasan_kesalahan_recursive.json"
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

    st.write("Folder output:")
    st.code(str(FOLDER_OUTPUT))