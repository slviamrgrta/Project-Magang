import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def show(df_harian=None):
    # =============================
    # 🎨 STYLE HALAMAN
    # =============================
    st.markdown("""
    <style>
    .divider {
        border: none;
        border-top: 1px solid rgba(0,0,205,0.2);
        margin: 24px 0px 24px 0px;
    }
    .stButton button {
        background: linear-gradient(90deg, rgb(0,0,205), rgb(65,105,225));
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        transition: 0.2s;
        margin-top: 8px;
        padding: 0.5rem 0rem;
    }
    .stButton button:hover {
        background: rgb(25,25,180);
        transform: scale(1.02);
    }
    .result-box {
        border: 2px solid;
        border-radius: 10px;
        padding: 12px 15px;
        margin-top: 12px;
        font-size: 15px;
        box-shadow: 0px 2px 6px rgba(128,128,128,0.2);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-positive {
        border-color: rgb(0,0,205);
        background-color: rgba(0,0,205,0.05);
    }
    .result-neutral {
        border-color: rgb(34,139,34);
        background-color: rgba(34,139,34,0.05);
    }
    .result-negative {
        border-color: rgb(220,20,60);
        background-color: rgba(220,20,60,0.05);
    }

    /* 🎨 Gaya tabel dengan header biru */
    .dataframe-table thead tr th {
        background-color: rgb(0,0,205) !important;
        color: white !important;
        font-weight: 700 !important;
        text-align: center !important;
        padding: 8px 4px !important;
    }
    .dataframe-table tbody tr td {
        text-align: center !important;
        background-color: white;
        padding: 6px 4px !important;
    }
    .dataframe-table {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
        border-collapse: collapse !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # =============================
    # 🧭 HEADER
    # =============================
    st.markdown("""
    <h2 style="
        text-align: left; 
        color: rgb(0,0,205);
        font-weight: 700;                     
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        font-size: 21px;
        margin-top: -15px;
        margin-bottom: 20px;  
    ">
    Halaman Prediksi 
    </h2>
    """, unsafe_allow_html=True)

    # =============================
    # 🧠 LOAD MODEL
    # =============================
    model_dir = os.path.abspath("model_nlp")

    if not os.path.exists(model_dir):
        st.error("❌ Folder 'model_nlp' tidak ditemukan. Pastikan sudah ada di direktori utama aplikasi.")
        st.stop()

    required_files = ["model.safetensors", "config.json", "tokenizer_config.json", "vocab.txt"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]

    if missing_files:
        st.error(f"❌ File berikut hilang di folder model_nlp: {', '.join(missing_files)}")
        st.stop()

    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.eval()
        return tokenizer, model

    try:
        tokenizer, model = load_model()
    except Exception as e:
        st.error(f"❌ Gagal memuat model lokal: {e}")
        st.stop()

    # =============================
    # 🧩 LAYOUT UTAMA
    # =============================
    col1, col2 = st.columns(2, gap="large")

    # =============================
    # ✏️ ANALISIS TEKS TUNGGAL
    # =============================
    with col1:
        st.markdown("""
        <h4 style="
            color: rgb(0,0,205); 
            font-size: 17px; 
            font-weight: 600; 
            margin-top: -6px; 
            margin-bottom: 8px;
        ">
        ✏️ Analisis Teks Tunggal
        </h4>
        """, unsafe_allow_html=True)

        user_input = st.text_area(
            "Masukkan teks di sini:",
            placeholder="Contoh: Pelayanan sangat memuaskan, petugasnya ramah sekali!",
            height=95
        )

        if st.button("🔍 Analisis Sentimen", use_container_width=True):
            if not user_input.strip():
                st.warning("Masukkan teks terlebih dahulu!")
            else:
                with st.spinner("Sedang menganalisis..."):
                    try:
                        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
                        with torch.no_grad():
                            outputs = model(**inputs)
                            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                            pred = torch.argmax(probs, dim=1).item()

                        label_map = {
                            0: ("Negatif 😠", "result-negative"),
                            1: ("Netral 😐", "result-neutral"),
                            2: ("Positif 😊", "result-positive")
                        }
                        sentiment, css_class = label_map.get(pred, ("Tidak diketahui", "result-neutral"))

                        # 🔹 Tampilkan hasil
                        st.markdown(f"""
                            <div class='result-box {css_class}'>
                                <p style="font-size:15px; margin:4px 0;">
                                    <b>Teks:</b> {user_input}
                                </p>
                                <p style="font-size:15px; font-weight:600; color:rgb(0,0,100); margin-top:6px;">
                                    <b>Prediksi Sentimen:</b> {sentiment}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat analisis: {e}")

    # =============================
    # 📂 ANALISIS MASSAL CSV
    # =============================
    with col2:
        st.markdown("""
        <h4 style="
            color: rgb(0,0,205); 
            font-size: 17px; 
            font-weight: 600; 
            margin-top: -6px; 
            margin-bottom: 8px;
        ">
        📂 Analisis Sentimen dari File CSV
        </h4>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Unggah file CSV yang berisi kolom teks", type="csv")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("📄 Pratinjau data:")
                st.markdown(df.head().to_html(index=False, classes='dataframe-table'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Gagal membaca file CSV: {e}")
                st.stop()

            text_column = st.selectbox("Pilih kolom teks untuk dianalisis:", options=df.columns)

            if st.button("🚀 Jalankan Analisis", use_container_width=True):
                with st.spinner("Sedang menganalisis seluruh teks..."):
                    def analyze_text(text):
                        try:
                            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                                pred = torch.argmax(probs, dim=1).item()
                            return {0: "Negatif 😠", 1: "Netral 😐", 2: "Positif 😊"}.get(pred, "Tidak diketahui")
                        except Exception:
                            return "Error"

                    df["Sentimen"] = df[text_column].apply(analyze_text)
                    st.success("✅ Analisis selesai!")

                    # 🔹 Tampilkan hasil dengan tabel biru
                    st.markdown(df.head(10).to_html(index=False, classes='dataframe-table'), unsafe_allow_html=True)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Unduh Hasil Analisis",
                        csv,
                        "hasil_sentimen.csv",
                        "text/csv",
                        use_container_width=True
                    )
