import streamlit as st
import pandas as pd
import io
import plotly.express as px
import os
import gdown
from src.preprocessing import load_and_prepare_data
from src.prediction import predict_future
from src.visualization import plot_interaktif

# ======================================================
# 🧭 KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Aplikasi Prediksi Permohonan",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Hilangkan menu dan footer Streamlit ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🎯 JUDUL UTAMA
# ======================================================
st.markdown("""
<h1 style='text-align:center; color: rgb(0,0,205); margin-bottom:-15px;'>
Aplikasi Prediksi Jumlah Permohonan
</h1>
""", unsafe_allow_html=True)

# ======================================================
# 🎨 TAMBAHAN CSS UNTUK TABEL & TAMPILAN
# ======================================================
st.markdown("""
<style>
h1 {
    margin-bottom: -10px !important;
    padding-bottom: 0 !important;
}
div[role="radiogroup"] {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
    padding-top: 0 !important;
}
div[role="radio"] {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}
h2 {
    margin-top: 4px;
    margin-bottom: 2px;
    color: rgb(0,0,205);
    font-weight:600;
    font-size:19px;
}
section[data-testid="stMarkdownContainer"] h3 {
    margin-top: -10px !important;
    margin-bottom: 2px !important;
}
.custom-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0px 1px 4px rgba(128,128,128,0.2);
    table-layout: fixed;
    word-wrap: break-word;
    margin-bottom: 6px;
}
.custom-table th {
    padding: 4px;
    font-size: 11px;
}
.custom-table td {
    padding: 3px;
    font-size: 11px;
}
.custom-table tr:nth-child(even){background-color:white;}
.custom-table tr:nth-child(odd){background-color: rgba(0,0,205,0.05);}
.custom-table tr:hover{background-color: rgba(34,139,34,0.08);}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 📂 LOAD DATA HISTORIS
# ======================================================
data_path = "data/tbl_permohonan_202507221101.csv"
df_harian = load_and_prepare_data(data_path)

# ======================================================
# 🧠 SIAPKAN MODEL NLP DARI GOOGLE DRIVE
# ======================================================
model_folder = "model_nlp"
os.makedirs(model_folder, exist_ok=True)

# --- File hasil fine-tuning kamu di Google Drive ---
files = {
    "model.safetensors": "15A8wnWNUrnMaYiRqS7m8DWk39otabKY5",
    "config.json": "1Bzfu0gz6l4tjnCqp2A_CajOnDaCaBgpp",
    "vocab.txt": "1fIk8GsRBg0dknuSCG5q4x7AnB1cCOpqa",
    "tokenizer_config.json": "1OG40ey5Eq53k6w-LiegBVq5ZbifYeFEl",
    "special_tokens_map.json": "1m-3nmTaaZ0kj2R2Peo1odlzOlGHBUryr"
}

# --- Unduh semua file model kalau belum ada (tanpa tulisan apa pun) ---
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(model_folder, filename)
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=True)

# ======================================================
# 📊 PILIH HALAMAN
# ======================================================
menu_choice = st.radio(
    label="", 
    options=["Analisis Data Historis", "Prediksi Jumlah Permohonan", "Analisis Sentimen"],
    horizontal=True,
    label_visibility="collapsed"
)

# ======================================================
# 🧩 TAMPILKAN HALAMAN SESUAI PILIHAN
# ======================================================
if menu_choice == "Analisis Data Historis":
    import pages.analisis as analisis
    analisis.show(df_harian)

elif menu_choice == "Prediksi Jumlah Permohonan":
    import pages.prediksi as prediksi
    prediksi.show(df_harian)

elif menu_choice == "Analisis Sentimen":
    import pages.sentimen as sentimen
    sentimen.show(df_harian)
