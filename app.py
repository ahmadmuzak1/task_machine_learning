from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat pipeline Random Forest dan fitur yang digunakan saat training
bundle = joblib.load("random_forest_bundle.joblib")
model = bundle["model"]
feature_columns = bundle["feature_columns"]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Random Forest Classifier"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or "data" not in payload:
        return jsonify({
            "error": "Kirim JSON dengan format {'data': {...}} atau {'data': [{...}]}"
        }), 400

    data = payload["data"]

    # Satu baris boleh dikirim sebagai object; beberapa baris sebagai list of objects
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or len(data) == 0 or not all(isinstance(row, dict) for row in data):
        return jsonify({"error": "'data' harus berupa object atau list berisi object"}), 400

    df = pd.DataFrame(data)

    # Periksa fitur yang wajib ada agar input cocok dengan model training
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        return jsonify({
            "error": "Ada fitur wajib yang belum dikirim",
            "missing_features": missing,
            "required_features": feature_columns
        }), 400

    # Susun kolom sesuai urutan saat training; kolom tambahan diabaikan
    df = df[feature_columns]

    try:
        predictions = model.predict(df)
        return jsonify({"prediction": [str(value) for value in predictions]})
    except Exception as exc:
        return jsonify({"error": "Prediksi gagal", "detail": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
