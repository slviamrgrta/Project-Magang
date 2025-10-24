import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ============================================================
# 1. FUNGSI PEMBUATAN FITUR
# ============================================================
def buat_fitur(df):
    """Membuat fitur lag, rolling, dan kalender (sesuai fitur training)."""
    for lag in [10, 20, 30]:
        df[f"jumlah_permohonan_lag{lag}"] = df["jumlah_permohonan"].shift(lag)

    for window in [10, 20, 30]:
        df[f"permohonan_mean{window}"] = df["jumlah_permohonan"].rolling(window=window).mean()
        df[f"permohonan_std{window}"] = df["jumlah_permohonan"].rolling(window=window).std()

    df["hari"] = df["tanggal"].dt.day
    df["bulan"] = df["tanggal"].dt.month
    df["tahun"] = df["tanggal"].dt.year
    df["dayofweek"] = df["tanggal"].dt.dayofweek
    df["quarter"] = df["tanggal"].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Hari libur nasional (contoh)
    hari_libur = [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-05-01"),
        pd.Timestamp("2025-08-17"),
        pd.Timestamp("2025-12-25")
    ]
    df["is_holiday"] = df["tanggal"].isin(hari_libur).astype(int)
    return df


# ============================================================
# 2. FUNGSI PREDIKSI KE DEPAN
# ============================================================
def predict_future(df_harian, n_forecast=7, output_path="output/prediksi_7hari.csv"):
    """
    Melakukan prediksi jumlah permohonan beberapa hari ke depan menggunakan model SVR.
    """
    # --- Load model & scaler ---
    model_path = Path("models")
    best_model = joblib.load(model_path / "svr_model.pkl")
    scaler_x = joblib.load(model_path / "scaler_x.pkl")
    scaler_y = joblib.load(model_path / "scaler_y.pkl")

    # --- Fitur yang digunakan (harus sesuai training) ---
    features = [
        "jumlah_permohonan_lag10", "jumlah_permohonan_lag20", "jumlah_permohonan_lag30",
        "permohonan_mean10", "permohonan_std10",
        "permohonan_mean20", "permohonan_std20",
        "permohonan_mean30", "permohonan_std30",
        "hari", "bulan", "tahun", "dayofweek", "quarter",
        "is_holiday", "is_weekend"
    ]

    # --- Urutkan data berdasarkan tanggal ---
    df_harian = df_harian.sort_values("tanggal").reset_index(drop=True)

    # --- Simulasi prediksi step-by-step ---
    hasil_prediksi = []
    df_pred = df_harian.copy()

    for i in range(1, n_forecast + 1):
        tanggal_pred = df_pred["tanggal"].max() + pd.Timedelta(days=1)

        # Tambahkan baris kosong untuk tanggal baru
        df_pred = pd.concat([
            df_pred,
            pd.DataFrame({"tanggal": [tanggal_pred], "jumlah_permohonan": [np.nan]})
        ], ignore_index=True)

        # Siapkan data sementara untuk feature generation
        df_temp = df_pred.copy()
        df_temp["jumlah_permohonan"] = df_temp["jumlah_permohonan"].ffill()
        df_temp = buat_fitur(df_temp)

        # Ambil baris fitur tanggal prediksi
        X_new = df_temp.loc[df_temp["tanggal"] == tanggal_pred, features]

        if X_new.isna().sum().sum() > 0:
            print(f"❌ Data belum cukup untuk prediksi {tanggal_pred.date()}")
            print(X_new.isna().sum())
            break

        # Scaling dan prediksi
        X_new_scaled = scaler_x.transform(X_new)
        y_pred_scaled = best_model.predict(X_new_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]

        # Simpan hasil
        hasil_prediksi.append((tanggal_pred, y_pred))

        # Masukkan hasil prediksi ke df_pred untuk prediksi hari berikutnya
        df_pred.loc[df_pred["tanggal"] == tanggal_pred, "jumlah_permohonan"] = y_pred

    # --- Hasil akhir ---
    df_hasil = pd.DataFrame(hasil_prediksi, columns=["tanggal", "jumlah_permohonan_prediksi"])
    print("📅 Hasil prediksi 7 hari ke depan:")
    print(df_hasil)

    # Simpan ke file CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_hasil.to_csv(output_path, index=False)
    print(f"✅ Hasil disimpan di: {output_path}")

    return df_hasil


# ============================================================
# 3. VISUALISASI
# ============================================================
def plot_prediction(df_harian, df_hasil):
    """Menampilkan grafik 30 hari terakhir + prediksi 7 hari ke depan."""
    df_harian = df_harian.sort_values("tanggal").reset_index(drop=True)
    df_recent = df_harian[df_harian["tanggal"] >= (df_harian["tanggal"].max() - pd.Timedelta(days=30))]

    plt.figure(figsize=(12, 6))
    plt.plot(df_recent["tanggal"], df_recent["jumlah_permohonan"],
             label="Data Aktual (30 Hari Terakhir)", color="blue")
    plt.plot(df_hasil["tanggal"], df_hasil["jumlah_permohonan_prediksi"],
             label="Prediksi 7 Hari ke Depan", color="red", linestyle="--", marker="o")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)

    for _, row in df_hasil.iterrows():
        plt.text(row["tanggal"], row["jumlah_permohonan_prediksi"] + 0.3,
                 f"{row['jumlah_permohonan_prediksi']:.1f}", ha="center", color="red", fontsize=9)

    plt.title("Prediksi Jumlah Permohonan 7 Hari ke Depan")
    plt.xlabel("Tanggal")
    plt.ylabel("Jumlah Permohonan")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. MAIN UNTUK DIJALANKAN LANGSUNG
# ============================================================
if __name__ == "__main__":
    # Load data historis
    df_harian = pd.read_csv("data/df_harian.csv", parse_dates=["tanggal"])

    # Jalankan prediksi
    df_hasil = predict_future(df_harian, n_forecast=7, output_path="output/prediksi_7hari.csv")

    # Visualisasi hasil
    plot_prediction(df_harian, df_hasil)
