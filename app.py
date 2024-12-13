import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file, flash
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Flask uygulamasını başlat
app = Flask(__name__)
app.secret_key = 'secret_key'  # Flash mesajları için gerekli

# Model dosyasını yükle
MODEL_PATH = 'model.keras'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    model = None

# Static klasörünü oluştur (varsa hata yapmaz)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Ana sayfa rotası
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            flash('Lütfen bir hisse senedi kodu girin.', 'warning')
            return render_template('index.html')

        if not model:
            flash('Model dosyası bulunamadı. Lütfen "model.keras" dosyasını kontrol edin.', 'danger')
            return render_template('index.html')

        try:
            # Veri indir
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 12, 1)
            df = yf.download(stock, start=start, end=end)
            data_desc = df.describe()

            if df.empty:
                flash('Veri indirilemedi. Lütfen geçerli bir hisse senedi kodu girin.', 'danger')
                return render_template('index.html')

            # Hareketli ortalamalar
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()

            # Eğitim ve test verisi
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

            # Veriyi ölçeklendir
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Veri hazırlığı
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # Tahminler
            y_predicted = model.predict(x_test)
            scale_factor = 1 / scaler.scale_[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Grafikler oluştur
            ema_chart_path = os.path.join(STATIC_DIR, "ema_20_50.png")
            plt.figure(figsize=(12, 6))
            plt.plot(df.Close, 'y', label='Kapanış Fiyatı')
            plt.plot(ema20, 'g', label='EMA 20')
            plt.plot(ema50, 'r', label='EMA 50')
            plt.title("Kapanış Fiyatı (20 ve 50 Gün EMA)")
            plt.legend()
            plt.savefig(ema_chart_path)
            plt.close()

            ema_chart_path_100_200 = os.path.join(STATIC_DIR, "ema_100_200.png")
            plt.figure(figsize=(12, 6))
            plt.plot(df.Close, 'y', label='Kapanış Fiyatı')
            plt.plot(ema100, 'g', label='EMA 100')
            plt.plot(ema200, 'r', label='EMA 200')
            plt.title("Kapanış Fiyatı (100 ve 200 Gün EMA)")
            plt.legend()
            plt.savefig(ema_chart_path_100_200)
            plt.close()

            prediction_chart_path = os.path.join(STATIC_DIR, "stock_prediction.png")
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, 'g', label='Gerçek Fiyat')
            plt.plot(y_predicted, 'r', label='Tahmin Edilen Fiyat')
            plt.title("Tahmin vs Gerçek Eğilim")
            plt.legend()
            plt.savefig(prediction_chart_path)
            plt.close()

            # CSV kaydet
            csv_file_path = os.path.join(STATIC_DIR, f"{stock}_dataset.csv")
            df.to_csv(csv_file_path)

            return render_template('index.html',
                                   plot_path_ema_20_50=ema_chart_path,
                                   plot_path_ema_100_200=ema_chart_path_100_200,
                                   plot_path_prediction=prediction_chart_path,
                                   data_desc=data_desc.to_html(classes='table table-bordered'),
                                   dataset_link=os.path.basename(csv_file_path))

        except Exception as e:
            flash(f'Hata oluştu: {str(e)}', 'danger')

    return render_template('index.html')

# Dosya indirme rotası
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    flash('Dosya bulunamadı.', 'danger')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
