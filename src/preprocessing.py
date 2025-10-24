import pandas as pd
import numpy as np
import holidays

def load_and_prepare_data(file_path: str):
    """Baca CSV dan buat semua fitur yang dibutuhkan untuk prediksi harian jumlah permohonan."""
    
    # 1. Baca data
    df = pd.read_csv(file_path)

    # 2. Hapus kolom tidak relevan
    if "status" in df.columns:
        df = df.drop(columns=["status"])

    # 3. Ganti nama kolom jika perlu
    if "id_jenis_layanan" in df.columns:
        df = df.rename(columns={"id_jenis_layanan": "jumlah_permohonan"})

    # 4. Pastikan tanggal dalam datetime
    df["tanggal_permohonan"] = pd.to_datetime(df["tanggal_permohonan"])

    # 5. Agregasi per hari
    df_harian = (
        df.groupby(df["tanggal_permohonan"].dt.date)
          .agg(jumlah_permohonan=("jumlah_permohonan", "sum"),
               total_harga=("total_harga", "sum"))
          .reset_index()
          .rename(columns={"tanggal_permohonan": "tanggal"})
    )
    df_harian["tanggal"] = pd.to_datetime(df_harian["tanggal"])

    # 6. Fitur waktu
    df_harian["hari"] = df_harian["tanggal"].dt.day
    df_harian["bulan"] = df_harian["tanggal"].dt.month
    df_harian["tahun"] = df_harian["tanggal"].dt.year
    df_harian["dayofweek"] = df_harian["tanggal"].dt.dayofweek
    df_harian["quarter"] = df_harian["tanggal"].dt.quarter

    # 7. Weekend & Holiday
    indo_holidays = holidays.Indonesia(years=df_harian["tahun"].unique())
    df_harian["is_weekend"] = df_harian["dayofweek"].isin([5, 6]).astype(int)
    df_harian["is_holiday"] = df_harian["tanggal"].isin(indo_holidays).astype(int)

    # 8. Fitur lag dan rolling
    for lag in [10, 20, 30]:
        df_harian[f"jumlah_permohonan_lag{lag}"] = df_harian["jumlah_permohonan"].shift(lag)

    for window in [10, 20, 30]:
        df_harian[f"permohonan_mean{window}"] = df_harian["jumlah_permohonan"].shift(1).rolling(window).mean()
        df_harian[f"permohonan_std{window}"]  = df_harian["jumlah_permohonan"].shift(1).rolling(window).std()

    # 9. Drop baris kosong (karena lag)
    df_harian = df_harian.dropna().reset_index(drop=True)

    return df_harian
