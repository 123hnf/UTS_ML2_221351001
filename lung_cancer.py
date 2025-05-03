import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="lung_cancer.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Risiko Kanker Paru-Paru")
st.write("Masukkan informasi kesehatan untuk memprediksi apakah Anda berisiko terkena kanker paru-paru.")

# Form input pengguna
GENDER = st.selectbox("Jenis Kelamin?", ["Perempuan", "Laki-Laki"])
AGE = st.number_input("Umur", min_value=1, max_value=100, value=30)
SMOKING = st.selectbox("Apakah Anda merokok?", ["Tidak", "Ya"])
YELLOW_FINGERS = st.selectbox("Apakah jari Anda menguning?", ["Tidak", "Ya"])
ANXIETY = st.selectbox("Apakah Anda mengalami kecemasan?", ["Tidak", "Ya"])
PEER_PRESSURE = st.selectbox("Apakah Anda terpengaruh tekanan teman sebaya untuk merokok?", ["Tidak", "Ya"])
CHRONIC_DISEASE = st.selectbox("Apakah Anda memiliki penyakit kronis?", ["Tidak", "Ya"])
FATIGUE = st.selectbox("Apakah Anda sering merasa lelah?", ["Tidak", "Ya"])
ALLERGY = st.selectbox("Apakah Anda memiliki alergi?", ["Tidak", "Ya"])
WHEEZING = st.selectbox("Apakah Anda mengalami mengi?", ["Tidak", "Ya"])
ALCOHOL_CONSUMING = st.selectbox("Apakah Anda mengonsumsi alkohol?", ["Tidak", "Ya"])
COUGHING = st.selectbox("Apakah Anda sering batuk?", ["Tidak", "Ya"])
SHORTNESS_OF_BREATH = st.selectbox("Apakah Anda mengalami sesak napas?", ["Tidak", "Ya"])
SWALLOWING_DIFFICULTY = st.selectbox("Apakah Anda mengalami kesulitan menelan?", ["Tidak", "Ya"])
CHEST_PAIN = st.selectbox("Apakah Anda sering merasakan sakit dada?", ["Tidak", "Ya"])

# Jika tombol diklik
if st.button("Prediksi Risiko"):
    # Mapping input Yes/No ke 1/0
    def map_input(val):
        return 1 if val == "Ya" else 0

    def map_gender(val):
        return 1 if val == "Laki-Laki" else 0

    input_data = np.array([[ 
        AGE,
        map_gender(GENDER),
        map_input(SMOKING),
        map_input(YELLOW_FINGERS),
        map_input(ANXIETY),
        map_input(PEER_PRESSURE),
        map_input(CHRONIC_DISEASE),
        map_input(FATIGUE),
        map_input(ALLERGY),
        map_input(WHEEZING),
        map_input(ALCOHOL_CONSUMING),
        map_input(COUGHING),
        map_input(SHORTNESS_OF_BREATH),
        map_input(SWALLOWING_DIFFICULTY),
        map_input(CHEST_PAIN),
    ]])

    # Scaling
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Prediksi
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Karena model binary sigmoid → outputnya probabilitas antara 0-1
    prob = prediction[0][0]  # Ambil nilai prediksi tunggal
    predicted_label = 1 if prob > 0.5 else 0  # Threshold 0.5

    # Konversi ke label string
    result = label_encoder.inverse_transform([predicted_label])[0] if hasattr(label_encoder, 'inverse_transform') else predicted_label

    st.success(f"**Prediksi: {str(result).upper()}**")
    st.info(f"Probabilitas risiko: {prob:.2f}")
