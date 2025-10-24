import plotly.graph_objects as go
import pandas as pd

def plot_interaktif(df_harian, df_prediksi):
    """Visualisasi data aktual, tren, dan prediksi 7 hari ke depan."""
    fig = go.Figure()

    # Data Aktual
    fig.add_trace(go.Scatter(
        x=df_harian["tanggal"],
        y=df_harian["jumlah_permohonan"],
        mode="lines+markers",
        name="Data Aktual",
        line=dict(color="blue")
    ))

    # Moving Average 7 hari
    df_harian_sorted = df_harian.sort_values("tanggal")
    df_harian_sorted["MA7"] = df_harian_sorted["jumlah_permohonan"].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df_harian_sorted["tanggal"],
        y=df_harian_sorted["MA7"],
        mode="lines",
        name="MA 7 Hari",
        line=dict(color="green", dash="dot")
    ))

    # Prediksi
    fig.add_trace(go.Scatter(
        x=df_prediksi["tanggal"],
        y=df_prediksi["jumlah_permohonan_prediksi"],
        mode="lines+markers",
        name="Prediksi",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title="Prediksi Jumlah Permohonan & Tren",
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Permohonan",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def analisis_deskriptif(df_harian):
    """
    Menghasilkan statistik deskriptif & tren sederhana dari data historis.
    """
    stats = df_harian["jumlah_permohonan"].describe().to_frame().T
    stats["range"] = stats["max"] - stats["min"]

    # Tren sederhana: selisih rata-rata harian
    df_sorted = df_harian.sort_values("tanggal")
    df_sorted["diff"] = df_sorted["jumlah_permohonan"].diff()
    stats["rata2_pertambahan_harian"] = df_sorted["diff"].mean()

    # Konversi kolom yang panjang jadi lebih rapi
    stats = stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "range", "rata2_pertambahan_harian"]]
    
    return stats
