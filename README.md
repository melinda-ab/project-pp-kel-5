# Evaluasi Performa Wiener Filter dalam Meningkatkan Akurasi Estimasi Pitch pada Sinyal Audio Noise

## Project Overview
Repositori ini berisi Final Project Kelompok 5 untuk mata kuliah Pengenalan Pola.

Kami mengevaluasi ketahanan algoritma pitch detection pYIN (Probabilistic YIN) melalui sistem yang diusulkan menggunakan Wiener Filter di domain frekuensi. Dengan kriteria minimalisasi kesalahan berbasis MSE, sistem ini dirancang untuk mencari titik keseimbangan optimal antara retensi integritas sinyal dan penekanan derau (noise).

Video presentasi dan paper dapat diakses melalui tautan berikut:
https://drive.google.com/drive/folders/1xx7c5Rt5r4zOVa6U5WX7h6X47955n_7k?hl=ID

---

## Contributors
Kelompok 5:
- Gilbert Nathaniel
- Dhimas Putra Sulistio
- Bobby Rahman Hartanto
- Melinda Annastasia B.
- Arnoldus Dharma W. M. 

---

## Architecture & Phases

Logika utama dijalankan melalui 'pipeline.py' dengan tahapan berikut:

1. **Preprocessing**: Load `.wav`, trim silence (30dB), dan normalisasi amplitudo ke 0.89.
2. **Noise Injection**: Penambahan AWGN (Additive White Gaussian Noise) sesuai target SNR.
3. **Denoising**: Penerapan **Optimal Linear Wiener Filter** untuk menekan derau:
   $$\text{Gain} = \frac{|S|^2 - \alpha P_n}{|S|^2}$$
4. **Pitch Detection**: Estimasi $f_0$ menggunakan `librosa.pyin` pada sinyal asli & denoised.
5. **Evaluation**: Komparasi hasil estimasi terhadap *ground truth* dari filename.

---

## Evaluation Metrics

| Metrik | Deskripsi | Formula |
| :--- | :--- | :--- |
| **RPA** | Raw Pitch Accuracy | % frame dengan $|error| < 50$ cents |
| **Mean Error** | Rata-rata deviasi | Mean absolute error pada voiced frames |
| **Cents** | Skala logaritmik | $1200 \times \log_2(f_{est} / f_{gt})$ |
