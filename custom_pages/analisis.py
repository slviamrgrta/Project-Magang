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

    # --- Judul Data Historis ---
    st.markdown("""
    <h2 style="
        text-align: left; 
        color: rgb(0,0,205);
        font-weight: 700;                     
        font-family: 'Poppins', 'Segoe UI', sans-serif;
        font-size: 21px;
        margin-top: -25px;   
        margin-bottom: 0px;  
    ">
    Data Historis Mingguan 
    </h2>
    """, unsafe_allow_html=True)

    # --- Baca file CSV mentah dan ubah menjadi agregasi mingguan ---
    try:
        df_raw = pd.read_csv("data/tbl_permohonan_202507221101.csv")

        # 🧹 Hapus 2 baris terakhir
        df_raw = df_raw.iloc[:-2]

        # Pastikan nama kolom seragam
        df_raw = df_raw.rename(columns={
            "tanggal_permohonan": "tanggal",
            "id_jenis_layanan": "jumlah_permohonan"
        })

        # Ubah kolom tanggal jadi datetime
        df_raw["tanggal"] = pd.to_datetime(df_raw["tanggal"], errors="coerce")

        # ===== 🔹 Agregasi Mingguan =====
        df_mingguan = (
            df_raw.groupby(df_raw["tanggal"].dt.to_period("W-SUN"))
            .agg(
                jumlah_permohonan=("jumlah_permohonan", "sum"),
                total_harga=("total_harga", "sum")
            )
            .reset_index()
        )

        # Ambil awal minggu sebagai tanggal representatif
        df_mingguan["tanggal"] = df_mingguan["tanggal"].dt.start_time
        df_mingguan = df_mingguan.rename(columns={"tanggal": "Tanggal", "jumlah_permohonan": "Jumlah Permohonan", "total_harga": "Total Harga"})

        # Format tanggal
        df_mingguan["Tanggal"] = df_mingguan["Tanggal"].dt.strftime("%d %b %Y")

        # Tampilkan 10 minggu terakhir
        st.markdown(df_mingguan.tail(10).to_html(classes="custom-table", index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Gagal memuat data mentah: {e}")

    # --- Analisis Jumlah Permohonan per Tahun (pakai df_harian dari parameter) ---
    df_harian["tanggal"] = pd.to_datetime(df_harian["tanggal"])
    df_harian["tahun"] = df_harian["tanggal"].dt.year
    df_harian["bulan"] = df_harian["tanggal"].dt.month

    df_pertahun_bulan = (
        df_harian.groupby(["tahun", "bulan"])["jumlah_permohonan"].sum().reset_index()
    )

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

    # --- Visualisasi tetap sama ---
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

        fig.update_traces(
            textposition="outside",
            texttemplate="%{text}",
            textfont=dict(color="rgb(50,50,50)", size=12)
        )

        y_max = df_tahun["jumlah_permohonan"].max() * 1.4

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
