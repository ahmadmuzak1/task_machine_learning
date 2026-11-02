# Dampak Media Sosial pada Kehidupan Mahasiswa (Laporan Lab Machine Learning)

Ini project laporan lab Machine Learning yang aku kerjain buat tugas kuliah. Isinya satu notebook yang jalan dari awal sampai akhir: ambil data, bersihin, bagi data, latih model, nilai hasilnya, terus dipasang jadi API sederhana pakai Flask.

Datanya soal kebiasaan pakai media sosial mahasiswa (jam pakai, jam tidur, tingkat stres, dll) dan dampaknya ke kehidupan mereka.

## Isi repo

```
.
├── laporan_lab_1_regresi.ipynb         # notebook utama, semua tahapan ada di sini
├── Social_media_impact_on_life.csv     # dataset
└── README.md
```

File lain (`app.py`, `random_forest_bundle.joblib`, hasil prediksi, dll) bakal kebentuk sendiri waktu notebook dijalankan.

## Dataset

`Social_media_impact_on_life.csv` isinya 4.500 baris dan 16 kolom. Kolomnya:

| Kolom | Keterangan |
|---|---|
| Student_ID | ID mahasiswa (nggak dipakai buat modelnya) |
| Age | Umur |
| Gender | Jenis kelamin |
| Academic_Level | Jenjang pendidikan (High School, Undergraduate, dst) |
| Primary_Platform | Platform yang paling sering dipakai |
| Daily_Usage_Hours | Jam pakai media sosial per hari |
| Weekend_Extra_Hours | Tambahan jam pakai di akhir pekan |
| Device_Type | Jenis perangkat |
| Sleep_Duration_Hours | Lama tidur (jam) |
| Sleep_Quality_Score | Skor kualitas tidur |
| Late_Night_Usage | Pakai media sosial larut malam atau nggak (True/False) |
| Social_Comparison_Frequency | Seberapa sering membandingkan diri dengan orang lain |
| Perceived_Stress_Score | Skor stres yang dirasakan |
| Mental_Health_Index | Indeks kesehatan mental |
| Academic_Performance_GPA | IPK |
| Overall_Impact | Dampak keseluruhan: Beneficial, Neutral, atau Negative |

Di data ini ada nilai kosong, salah satunya di kolom IPK (sekitar 85 baris), jadi sempat perlu ditangani dulu.

## Alur di notebook

1. **Pengumpulan data.** Baca CSV pakai pandas, cek ukuran, kolom, tipe data, dan statistik dasar.
2. **Pra-pemrosesan.** Isi nilai kosong (median buat angka, modus buat kategori), tangani outlier pakai IQR, hapus data duplikat, ubah kategori jadi angka (one-hot dan label encoding), dan standardisasi.
3. **EDA.** Histogram, boxplot, dan heatmap korelasi buat ngelihat sebaran dan hubungan antar kolom.
4. **Pembagian data.** 80% training, 20% testing (`random_state=42`). Buat klasifikasi pakai `stratify` supaya proporsi kelasnya tetap sama.
5. **Pemilihan algoritma.** Random Forest, dua versi: Classifier (nebak kategori) dan Regressor (nebak angka).
6. **Pelatihan model.** Preprocessing dan model digabung dalam satu `Pipeline` scikit-learn.
7. **Evaluasi model.** Accuracy, F1, confusion matrix, feature importance buat klasifikasi. MAE, MSE, RMSE, dan R² buat regresi.
8. **Deployment.** Model disimpan pakai joblib, lalu dibungkus jadi API Flask.

## Hasil

### Klasifikasi: nebak `Overall_Impact`

Pakai `RandomForestClassifier` (200 pohon, `class_weight="balanced"`).

| Metrik | Nilai |
|---|---|
| Accuracy | 0.9589 |
| Macro F1 | 0.9133 |

| Kelas | Precision | Recall | F1 | Jumlah data uji |
|---|---|---|---|---|
| Beneficial | 0.99 | 0.97 | 0.98 | 736 |
| Negative | 0.91 | 0.88 | 0.89 | 33 |
| Neutral | 0.81 | 0.93 | 0.87 | 131 |

Datanya nggak seimbang (kelas Negative cuma 33 dari 900 data uji), jadi aku lebih percaya Macro F1 daripada Accuracy.

### Regresi: nebak IPK (`Academic_Performance_GPA`)

Ini tambahan supaya ada metrik MAE, MSE, dan R². Pakai `RandomForestRegressor`, dibandingin sama baseline yang selalu nebak rata-rata IPK. Kolom `Overall_Impact` dibuang dari fitur karena itu variabel hasil, bukan penyebab.

| Metrik | Random Forest | Baseline |
|---|---|---|
| MAE | sekitar 0.22 | sekitar 0.33 |
| MSE | sekitar 0.076 | sekitar 0.168 |
| RMSE | sekitar 0.276 | |
| R² | sekitar 0.55 | sekitar 0 |

Artinya rata-rata tebakan IPK meleset sekitar 0.22 poin, dan model bisa menjelaskan kira-kira 55% variasi IPK. Lumayan, tapi jelas belum bagus-bagus amat. Wajar sih, IPK dipengaruhi banyak hal yang nggak ada di dataset ini.

Angka bisa beda tipis kalau dijalankan di komputer lain (beda versi library).

## Cara menjalankan

Butuh Python 3 dan library berikut:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib flask requests jupyter
```

Terus:

1. Clone repo ini dan taruh CSV di folder yang sama dengan notebook.
2. Buka `laporan_lab_1_regresi.ipynb` di Jupyter atau Google Colab.
3. Jalankan semua cell dari atas ke bawah.

## Coba API-nya

Cell `%%writefile app.py` di notebook bakal bikin file `app.py`. Jalankan di terminal:

```bash
python app.py
```

Server jalan di `http://127.0.0.1:5000`.

Cek servernya hidup:

```bash
curl http://127.0.0.1:5000/health
```

Minta prediksi (semua kolom fitur harus dikirim, kecuali `Student_ID` dan `Overall_Impact`):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {
    "Age": 19,
    "Gender": "Female",
    "Academic_Level": "Undergraduate",
    "Primary_Platform": "Snapchat",
    "Daily_Usage_Hours": 7.5,
    "Weekend_Extra_Hours": 1.5,
    "Device_Type": "Smartphone",
    "Sleep_Duration_Hours": 5.9,
    "Sleep_Quality_Score": 2,
    "Late_Night_Usage": true,
    "Social_Comparison_Frequency": "Always",
    "Perceived_Stress_Score": 20.0,
    "Mental_Health_Index": 73,
    "Academic_Performance_GPA": 2.73
  }}'
```

Balasannya bentuknya kayak `{"prediction": ["..."]}`. Kalau ada fitur yang kurang, API bakal ngasih tahu fitur mana yang belum dikirim.

API ini memakai model **klasifikasi**. Model regresi cuma buat evaluasi di notebook, belum dipasang di API.

## Catatan dan kekurangan

- Dataset sangat nggak seimbang, jadi kelas Negative paling susah ditebak.
- Skor klasifikasinya tinggi banget. Aku belum ngecek apakah `Overall_Impact` itu sebenarnya dibentuk dari kolom lain di dataset (misalnya Mental_Health_Index atau IPK). Kalau iya, skornya bisa jadi kelihatan lebih bagus dari kenyataannya.
- Cuma pakai satu algoritma (Random Forest) dan belum ada tuning hyperparameter.
- Beberapa cell di bagian awal notebook agak berantakan karena hasil coba-coba, ada yang ngebaca CSV berulang kali.
- API-nya masih versi belajar (jalan di lokal, tanpa autentikasi), jangan dipakai buat produksi.

## Pembuat

Ahmad Muzaki Rahmadiansyah
Teknik Informatika, Fakultas Ilmu Komputer, Universitas Pamulang (kelas 05TPLP019)
