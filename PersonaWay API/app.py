from flask import Flask, request, jsonify
import mysql.connector  
import tensorflow as tf
import numpy as np
import os
import h5py
import sys
import logging
import pickle
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, GCS_BUCKET_NAME
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure database connection using mysql.connector
db = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

# Fungsi untuk memeriksa koneksi database
def check_db_connection():
    try:
        # Cek apakah koneksi berhasil
        cursor = db.cursor()
        cursor.execute("SELECT 1")  # Query sederhana untuk tes koneksi
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            print("Database connection successful!")
        else:
            print("Database connection failed!")
            raise Exception("Database connection check failed.")

    except mysql.connector.Error as e:  # Menggunakan mysql.connector.Error
        print(f"Error while connecting to database: {e}")
        raise e  # Raise exception jika terjadi error

# Cek koneksi database
check_db_connection()

def check_model():
    model_path = "model.h5"  # Sesuaikan path jika diperlukan
    try:
        print(f"Checking model file at: {model_path}")
        
        # Cek apakah file dapat dibuka sebagai file HDF5
        with h5py.File(model_path, "r") as f:
            print("File opened successfully. HDF5 contents:")
            print(f.keys())
        
        # Cek apakah file dapat dimuat sebagai model Keras
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    
    except Exception as e:
        print(f"Error while checking model: {e}")


# Load ML model directly from local directory
def load_model():
    """
    Memuat model TensorFlow/Keras dari file lokal.
    Menyertakan penanganan file tidak ditemukan dan objek kustom.
    """
    model_path = "model.h5"  # Sesuaikan path jika diperlukan
    
    try:
        # Periksa apakah file model ada
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")

        # Memuat model, sertakan custom_objects jika model menggunakan fungsi atau lapisan khusus
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()}  # Tambahkan objek lain jika diperlukan
        )
        
        print("Model loaded successfully!")
        return model

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        raise fnf_error  # Berhenti jika file tidak ditemukan
    
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        raise e  # Berhenti jika ada kesalahan lain


# Load ML model
model = load_model()


# 1. Tambahkan error handling yang lebih spesifik pada route predict
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validasi request JSON
        if not request.is_json:
            return jsonify({"error": "Content-Type harus application/json"}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Request body harus mengandung field 'text'"}), 400

        user_text = data['text']
        if not isinstance(user_text, str) or not user_text.strip():
            return jsonify({"error": "Field 'text' tidak boleh kosong"}), 400

        # Cek apakah model sudah dimuat
        if model is None:
            return jsonify({"error": "Model belum dimuat"}), 500

        # Proses tokenisasi
        try:
            with open('tokenizer.pkl', 'rb') as file:
             tokenizer = pickle.load(file)
            # tokenizer = Tokenizer(num_words=10000)
            #tokenizer.fit_on_texts([user_text])
             sequences = tokenizer.texts_to_sequences([user_text])
             padded_sequences = pad_sequences(sequences, maxlen=100)
        except Exception as e:
            return jsonify({"error": f"Error saat memproses teks: {str(e)}"}), 500

        # Proses prediksi
        try:
            prediction = model.predict(padded_sequences)
            predicted_label = np.argmax(prediction, axis=1)[0]
        except Exception as e:
            return jsonify({"error": f"Error saat melakukan prediksi: {str(e)}"}), 500

        # Query database
        try:
            cursor = db.cursor(dictionary=True)  # Gunakan dictionary=True agar hasil dalam bentuk dict
            cursor.execute("SELECT * FROM disc WHERE predik=%s", (int(predicted_label),))
            sql_result = cursor.fetchone()
            cursor.close()
            
            if sql_result is None:
                return jsonify({"error": f"Data untuk prediksi {predicted_label} tidak ditemukan"}), 404
        except Exception as e:
            return jsonify({"error": f"Error saat mengakses database: {str(e)}"}), 500

        # Siapkan response
        image_blob = f"gambar-disc/{predicted_label}.png"
        image_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{image_blob}"

        return jsonify({
            "status": "success",
            "prediction": int(predicted_label),
            "data": sql_result,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    try:
        model_status = "Loaded" if model is not None else "Not Loaded"
        db_status = "Connected" if db.is_connected() else "Disconnected"  # Menggunakan db.is_connected() untuk memeriksa status koneksi
        
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_path": os.path.abspath("model.h5") if os.path.exists("model.h5") else "Model file not found",
            "database_status": db_status,
            "supported_classes": [],  # Implement if class mapping is available
            "tensorflow_version": tf.__version__,
            "python_version": sys.version
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)