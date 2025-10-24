import streamlit as st
import pandas as pd
import io
import base64
import plotly.express as px
from src.prediction import predict_future

def show(df_harian):
    # =============================
    # 🎨 STYLE HALAMAN
    # =============================
    st.markdown("""
    <style>
    h1, h2, h3, h4 {
        color: rgb(0,0,205);
        font-family: "Segoe UI", sans-serif;
        font-weight: 600;
    }
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
        margin-top: 10px;
    }
    .stButton button:hover {
        background: rgb(25,25,180);
        transform: scale(1.02);
    }
    .table-wrapper {
        position: relative;
        display: inline-block;
        margin-top: 6px;
    }
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        max-width: 600px;
        font-size: 14px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(128,128,128,0.3);
        table-layout: fixed;
        word-wrap: break-word;
    }
    .custom-table th {
        background-color: rgb(0,0,205);
        color: white;
        text-align: center;
        padding: 6px;
        font-size: 13px;
    }
    .custom-table td {
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        padding: 6px;
        color: rgb(40,40,40);
        font-size: 13px;
    }
    .custom-table tr:nth-child(even) {background-color: rgb(255,255,255);}
    .custom-table tr:nth-child(odd) {background-color: rgba(0,0,205,0.05);}
    .custom-table tr:hover {background-color: rgba(0,0,205,0.12); transition: background-color 0.2s ease;}
    .download-icon {
        position: absolute;
        top: -10px;
        right: -25px;
        font-size: 18px;
        text-decoration: none;
        color: rgb(0,0,205);
        transition: 0.2s;
    }
    .download-icon:hover {
        color: rgb(25,25,180);
        transform: scale(1.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # =============================
    # ⚙️ HALAMAN PREDIKSI
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
    # 🔹 2 KOLOM UTAMA
    # =============================
    col_input, col_table = st.columns([1.3, 2], gap="large")

    with col_input:
        st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
        n_forecast = st.slider("Berapa hari ke depan yang ingin diprediksi?", 1, 7, 7)
        predict_btn = st.button("🚀 Jalankan Prediksi", use_container_width=True)

    with col_table:
        st.markdown("<div style='margin-top:-5px'></div>", unsafe_allow_html=True)
        if predict_btn:
            with st.spinner(f"🔮 Sedang memprediksi {n_forecast} hari ke depan..."):
                df_pred = predict_future(df_harian, n_forecast=n_forecast)

            if df_pred is not None and not df_pred.empty:
                df_pred["tanggal"] = pd.to_datetime(df_pred["tanggal"])
                df_tampil = df_pred.copy()
                df_tampil["tanggal"] = df_tampil["tanggal"].dt.strftime("%d %b %Y")
                df_tampil = df_tampil.rename(columns={
                    "tanggal": "Tanggal",
                    "jumlah_permohonan_prediksi": "Jumlah Prediksi"
                })

                # Tombol download
                csv_buffer = io.StringIO()
                df_tampil.to_csv(csv_buffer, index=False)
                b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
                table_html = df_tampil.to_html(index=False, classes="custom-table", justify="center", border=0)
                download_link = f'<a href="data:text/csv;base64,{b64}" download="prediksi_{n_forecast}_hari.csv" class="download-icon" title="Download CSV">⬇️</a>'
                st.markdown(f"<div class='table-wrapper'>{table_html}{download_link}</div>", unsafe_allow_html=True)

                st.session_state["df_pred"] = df_pred
                st.session_state["n_forecast"] = n_forecast
            else:
                st.warning("❗ Data hasil prediksi kosong atau tidak valid.")
        else:
            st.info("Pilih jumlah hari dan tekan **🚀 Jalankan Prediksi** untuk melihat hasil.")

    # =============================
    # 📈 GRAFIK INTERAKTIF
    # =============================
    if "df_pred" in st.session_state and st.session_state["df_pred"] is not None:
        df_pred = st.session_state["df_pred"]
        n_forecast = st.session_state["n_forecast"]

        st.markdown("<div style='margin-top:40px'></div>", unsafe_allow_html=True)
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # 🔹 Judul grafik
        st.markdown(f"""
        <h3 style="
            text-align: left;
            color: rgb(0,0,205);
            font-weight: 700;
            font-family: 'Poppins', 'Segoe UI', sans-serif;
            font-size: 18px;
            margin-bottom: 10px;
        ">
        Grafik Prediksi Interaktif — {n_forecast} Hari ke Depan
        </h3>
        """, unsafe_allow_html=True)

        # --- Data untuk grafik ---
        df_harian["tanggal"] = pd.to_datetime(df_harian["tanggal"])
        df_2025 = df_harian[df_harian['tanggal'].dt.year == 2025].copy()

        if not df_2025.empty:
            df_recent = df_2025[df_2025['tanggal'] >= (df_2025['tanggal'].max() - pd.Timedelta(days=30))]
        else:
            df_recent = df_harian.tail(30)

        df_plot = pd.concat([
            df_recent[['tanggal', 'jumlah_permohonan']].rename(columns={'jumlah_permohonan': 'Jumlah'}),
            df_pred[['tanggal', 'jumlah_permohonan_prediksi']].rename(columns={'jumlah_permohonan_prediksi': 'Jumlah'})
        ], axis=0)
        df_plot['Tipe'] = ['Aktual'] * len(df_recent) + ['Prediksi'] * len(df_pred)

        # --- Grafik Plotly ---
        fig = px.line(
            df_plot,
            x='tanggal',
            y='Jumlah',
            color='Tipe',
            markers=True,
            text='Jumlah',
            labels={'tanggal': 'Tanggal', 'Jumlah': 'Jumlah Permohonan'},
            hover_data={'tanggal': True, 'Jumlah': True, 'Tipe': True}
        )

        # --- Tampilkan angka & pertebal garis ---
        fig.update_traces(
            texttemplate='%{text:.0f}',
            textposition='top center',
            textfont=dict(size=11, color='black'),
            line=dict(width=3)  # 🔹 Garis lebih tebal
        )

        # --- Layout dan Border Dalam ---
        fig.update_layout(
            xaxis=dict(
                tickformat='%d %b %Y',
                tickangle=-60,
                showgrid=False,
                tickmode='array',
                tickvals=df_plot['tanggal'],
                ticktext=[d.strftime('%d %b') for d in df_plot['tanggal']]
            ),
            yaxis=dict(
                title='Jumlah Permohonan',
                showgrid=False,
                range=[0, df_plot['Jumlah'].max() * 1.15]  # 🔹 Range sumbu Y diperbaiki di sini
            ),
            legend=dict(title='Tipe Data'),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            template='plotly_white',
            margin=dict(l=40, r=30, t=30, b=40),
            shapes=[
                dict(
                    type='rect',
                    xref='paper', yref='paper',
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(color='rgb(0,0,205)', width=2)
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)
