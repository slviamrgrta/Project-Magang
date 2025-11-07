import streamlit as st
import pandas as pd
import plotly.express as px
from src.prediction import train_and_predict


def show(df_harian):
    # =============================
    # 🎨 STYLE HALAMAN & TABEL KECIL
    # =============================
    st.markdown("""
    <style>
    .custom-table {
        border-collapse: collapse;
        width: 70%;
        font-size: 12px;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(128,128,128,0.25);
        table-layout: fixed;
        word-wrap: break-word;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .custom-table th {
        background-color: rgb(0,0,205);
        color: rgb(255,255,255);
        text-align: center;
        padding: 5px;
        font-size: 11.5px;
        white-space: nowrap;
    }
    .custom-table td {
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        padding: 4px;
        color: rgb(50,50,50);
        font-size: 11.5px;
    }
    .custom-table tr:nth-child(even) {background-color: rgb(255,255,255);}
    .custom-table tr:nth-child(odd) {background-color: rgba(0,0,205,0.05);}
    .custom-table tr:hover {
        background-color: rgba(34,139,34,0.1);
        transition: background-color 0.2s ease;
    }
    .custom-table {border: 1px solid rgba(128,128,128,0.25);}
    </style>
    """, unsafe_allow_html=True)

    # =============================
    # 🏷️ JUDUL HALAMAN
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
    # 🗓️ Ubah Data Harian ke Mingguan
    # =============================
    df_mingguan = (
        df_harian.groupby(df_harian["tanggal"].dt.to_period("W-SUN"))
        .agg(jumlah_permohonan=("jumlah_permohonan", "sum"))
        .reset_index()
    )
    df_mingguan["tanggal"] = df_mingguan["tanggal"].dt.start_time

    # =============================
    # ⚙️ Input Prediksi
    # =============================
    col_input, col_table = st.columns([1.3, 2], gap="large")

    with col_input:
        n_forecast = st.slider("Berapa minggu ke depan yang ingin diprediksi?", 1, 12, 4)
        predict_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True)

    with col_table:
        if predict_btn:
            with st.spinner(f"🔮 Sedang memprediksi {n_forecast} minggu ke depan..."):
                _, _, df_pred = train_and_predict(df_mingguan, periods=n_forecast, freq="W-MON")

            if df_pred is not None and not df_pred.empty:
                # ✨ Siapkan data tampil
                df_tampil = df_pred[["ds", "yhat"]].rename(columns={
                    "ds": "Minggu",
                    "yhat": "Prediksi Jumlah Permohonan"
                })

                # =============================
                # 📊 TABEL MINI BERWARNA
                # =============================
                st.markdown("##### Hasil Prediksi:")
                html_table = df_tampil.to_html(index=False, classes="custom-table")
                st.markdown(html_table, unsafe_allow_html=True)

                # Simpan hasil ke session
                st.session_state["df_pred"] = df_tampil
                st.session_state["n_forecast"] = n_forecast
            else:
                st.warning("❗ Data hasil prediksi kosong atau tidak valid.")
        else:
            st.info("Pilih jumlah minggu dan tekan **🚀 Jalankan Prediksi** untuk melihat hasil.")

    # =============================
    # 📈 GRAFIK INTERAKTIF
    # =============================
    if "df_pred" in st.session_state and st.session_state["df_pred"] is not None:
        df_pred = st.session_state["df_pred"]
        n_forecast = st.session_state["n_forecast"]

        df_recent = df_mingguan.tail(12)

        df_plot = pd.concat([
            df_recent.rename(columns={'tanggal': 'Minggu', 'jumlah_permohonan': 'Jumlah'}),
            df_pred.rename(columns={'Minggu': 'Minggu', 'Prediksi Jumlah Permohonan': 'Jumlah'})
        ])
        df_plot['Tipe'] = ['Aktual'] * len(df_recent) + ['Prediksi'] * len(df_pred)

        fig = px.line(
            df_plot,
            x='Minggu',
            y='Jumlah',
            color='Tipe',
            markers=True,
            text='Jumlah',
            labels={'Minggu': 'Minggu', 'Jumlah': 'Jumlah Permohonan'},
            hover_data={'Minggu': True, 'Jumlah': True, 'Tipe': True}
        )

        fig.update_traces(texttemplate='%{text:.0f}', textposition='top center')
        fig.update_layout(
            xaxis=dict(tickformat='%d %b %Y', tickangle=-60),
            yaxis=dict(title='Jumlah Permohonan'),
            legend=dict(title='Tipe Data'),
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
