import pandas as pd
import numpy as np
import joblib
from prophet import Prophet

# =========================================================
# 1️⃣ Load model Prophet yang sudah disimpan
# =========================================================
def load_model():
    model_path = "model/prophet_model.pkl"
    model = joblib.load(model_path)
    return model

# =========================================================
# 2️⃣ Fungsi Prediksi (future only)
# =========================================================
def train_and_predict(df_mingguan, periods=4, freq="W-MON"):
    """
    Melakukan prediksi masa depan murni menggunakan model Prophet yang sudah disimpan.
    Tidak memprediksi ulang data test, hanya periode setelah data terakhir.
    """
    # Pastikan format kolom sesuai
    df_prophet = df_mingguan.rename(columns={
        "tanggal": "ds",
        "jumlah_permohonan": "y"
    }).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    # 📅 Cek tanggal terakhir
    last_date = df_prophet["ds"].max()
    print(f"📅 Tanggal terakhir di dataset: {last_date}")

    # Load model
    model = load_model()

    # 🔮 Prediksi masa depan setelah data terakhir
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=periods,
        freq=freq
    )
    print(f"🔮 Prediksi dimulai dari: {future_dates[0].date()}")

    future_df = pd.DataFrame({"ds": future_dates})

    # Prediksi
    forecast_future = model.predict(future_df)

    # Jika model kamu dilatih dengan log transform (log1p), kembalikan ke skala asli
    if "yhat" in forecast_future.columns:
        forecast_future[["yhat", "yhat_lower", "yhat_upper"]] = np.expm1(
            forecast_future[["yhat", "yhat_lower", "yhat_upper"]]
        )

    # Hanya kembalikan data masa depan
    forecast_future = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    print(f"\n✅ Prediksi {periods} minggu ke depan dimulai dari {future_dates[0].date()}")

    return model, None, forecast_future
