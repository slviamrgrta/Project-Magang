# pages/analisis.py
import streamlit as st
import pandas as pd
import plotly.express as px

def show(df_harian):
    """
    Menampilkan halaman Analisis Data Historis.
    df_harian: DataFrame hasil load dan preprocessing data historis
    """

    # --- CSS Tabel ---
    st.markdown("""
    <style>
    .custom-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 15px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0px 2px 6px rgba(128,128,128,0.3);
        table-layout: fixed;
        word-wrap: break-word;
    }
    .custom-table th {
        background-color: rgb(0,0,205);
        color: rgb(255,255,255);
        text-align: center;
        padding: 6px;
        font-size: 12px;
        white-space: nowrap;
    }
    .custom-table td {
        border: 1px solid rgba(128,128,128,0.3);
        text-align: center;
        padding: 5px;
        color: rgb(50,50,50);
        font-size: 12px;
    }
    .custom-table tr:nth-child(even) {background-color: rgb(255,255,255);}
    .custom-table tr:nth-child(odd) {background-color: rgba(0,0,205,0.05);}
    .custom-table tr:hover {background-color: rgba(34,139,34,0.1); transition: background-color 0.2s ease;}
    .custom-table {border: 1px solid rgba(128,128,128,0.3);}
    </style>
    """, unsafe_allow_html=True)

    # --- Format Tabel untuk Tampilan ---
    df_tampil = df_harian.copy()
    df_tampil["tahun"] = df_tampil["tahun"].astype(str)
    df_tampil["tanggal"] = pd.to_datetime(df_tampil["tanggal"]).dt.strftime("%d %b %Y")
    df_tampil = df_tampil.rename(columns={
        "tanggal":"Tanggal","jumlah_permohonan":"Jumlah","hari":"Hari","bulan":"Bulan","tahun":"Tahun",
        "is_weekend":"Weekend?","is_holiday":"Holiday?","dayofweek":"Hari ke","quarter":"Quarter",
        "jumlah_permohonan_lag10":"Lag10","jumlah_permohonan_lag20":"Lag20","jumlah_permohonan_lag30":"Lag30",
        "permohonan_mean10":"Mean10","permohonan_std10":"Std10","permohonan_mean20":"Mean20","permohonan_std20":"Std20",
        "permohonan_mean30":"Mean30","permohonan_std30":"Std30"
    })

    # --- Judul Data Historis (lebih rapat ke atas) ---
    st.markdown("""
    <h2 style="
        text-align: left; 
        color: rgb(0,0,205);
        font-weight: 700;                     
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        font-size: 21px;
        margin-top: -25px;   /* dinaikkan lebih rapat */
        margin-bottom: 0px;  
    ">
    Data Historis
    </h2>
    """, unsafe_allow_html=True)

    # --- Tabel Data Historis ---
    st.markdown(df_tampil.tail(10).to_html(classes="custom-table", index=False), unsafe_allow_html=True)

    # --- Analisis Jumlah Permohonan per Tahun ---
    df_harian["tanggal"] = pd.to_datetime(df_harian["tanggal"])
    df_harian["tahun"] = df_harian["tanggal"].dt.year
    df_harian["bulan"] = df_harian["tanggal"].dt.month
    df_pertahun_bulan = df_harian.groupby(["tahun","bulan"])["jumlah_permohonan"].sum().reset_index()
    df_pertahun_bulan["bulan_nama"] = df_pertahun_bulan["bulan"].apply(
        lambda x: pd.to_datetime(str(x), format="%m").strftime("%b")
    )

    st.markdown("""
    <h2 style="
        text-align: left; 
        color: rgb(0,0,205);
        font-weight: 700;                     
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        font-size: 21px;
        margin-top: 8px;
        margin-bottom: 2px;
    ">
    Analisis Jumlah Permohonan
    </h2>
    """, unsafe_allow_html=True)

    # --- Visualisasi ---
    col1, col2 = st.columns([3,1])

    with col2:
        tahun_list = sorted(df_pertahun_bulan["tahun"].unique())
        selected_year = st.selectbox("Pilih Tahun:", tahun_list)

    with col1:
        df_tahun = df_pertahun_bulan[df_pertahun_bulan["tahun"] == selected_year]

        fig = px.bar(
            df_tahun,
            x="bulan_nama",
            y="jumlah_permohonan",
            text="jumlah_permohonan",
            color="jumlah_permohonan",
            color_continuous_scale=[
                "rgb(220,20,60)", "rgb(135,206,250)", "rgb(0,0,205)"
            ],
            labels={"jumlah_permohonan": "Jumlah Permohonan", "bulan_nama": "Bulan"},
            title=f"Jumlah Permohonan per Bulan Tahun {selected_year}"
        )

        # --- Atur tampilan teks batang ---
        fig.update_traces(
            textposition="outside",
            texttemplate="%{text}",
            textfont=dict(color="rgb(50,50,50)", size=12)
        )

        # --- Sumbu Y diperpanjang ---
        y_max = df_tahun["jumlah_permohonan"].max() * 1.4  # diperbesar dari 1.15 ke 1.4

        # --- Layout ---
        fig.update_layout(
            height=420,
            margin=dict(l=80, r=60, t=70, b=40),
            plot_bgcolor="rgb(255,255,255)",
            paper_bgcolor="rgb(255,255,255)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False, range=[0, y_max]),
            font=dict(color="rgb(34,34,34)", size=13),
            bargap=0.3,
            bargroupgap=0.05,
            title=dict(
                x=0.5, y=0.92,
                xanchor="center", yanchor="top",
                font=dict(size=16, color="rgb(25,25,180)")
            ),
            shapes=[dict(
                type="rect",
                xref="paper", yref="paper",
                x0=-0.005, y0=-0.02, x1=1.015, y1=1.02,
                line=dict(color="rgb(0,0,205)", width=1.5),
                layer="below"
            )]
        )

        st.plotly_chart(fig, use_container_width=True)
