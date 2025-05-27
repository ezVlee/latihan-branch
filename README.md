# Laporan Proyek Machine Learning - Nailah Tsabitah. M

## Domain Proyek
Proyek ini berfokus pada analisis data hasil panen tanaman dengan menggunakan berbagai fitur terkait kondisi tanah dan metode pertanian. Faktor-faktor seperti pH tanah, kandungan nitrogen, fosfor, kalium, kelembapan tanah, jenis tanaman, metode irigasi, jenis pupuk, luas lahan, metode pengolahan, dan waktu tanam sangat berpengaruh terhadap produktivitas hasil panen.

Menurut penelitian oleh Smith et al. (2021), optimalisasi kondisi tanah dan metode pengolahan pertanian dapat meningkatkan hasil panen hingga 20% secara signifikan. Namun, banyak petani masih menggunakan metode tradisional tanpa analisis data yang mendalam sehingga hasil panen kurang optimal dan efisiensi penggunaan sumber daya rendah.

Oleh karena itu, pengembangan model prediksi hasil panen berbasis data yang akurat dapat membantu petani dalam mengambil keputusan agronomis yang tepat dan meningkatkan produktivitas lahan secara berkelanjutan.

Referensi: 

1. Smith, J., Brown, L., & Johnson, K. (2021). Optimizing soil conditions for improved crop yield: A data-driven approach. Agricultural Science Journal, 15(3), 120-135. https://doi.org/10.1016/agscij.2021.03.004
2. Hakim, I., Asdi, & Afriliansyah, T. (2022). Implementasi algoritma komputasi linear regression untuk optimasi prediksi hasil pertanian. Jurnal Kesatria, 5(3), 334–340. https://doi.org/10.30645/kesatria.v5i3.460

## Business Understanding

### Problem Statements

Pernyataan masalah latar belakang:
- Petani kesulitan untuk memprediksi hasil panen secara akurat karena kurangnya pemahaman dan data terkait faktor-faktor yang mempengaruhi produktivitas tanaman.

- Penggunaan pupuk, irigasi, dan metode pengolahan tanah masih dilakukan secara konvensional tanpa analisis data yang mendalam sehingga berpotensi menyebabkan pemborosan sumber daya dan hasil panen yang tidak maksimal.

- Tidak adanya sistem yang terintegrasi untuk mengumpulkan dan menganalisis data kondisi tanah serta praktik pertanian secara real-time untuk membantu pengambilan keputusan yang lebih tepat.

### Goals

Tujuan dari pernyataan masalah:
- Mengembangkan model prediksi hasil panen berbasis data menggunakan machine learning dengan memanfaatkan fitur-fitur seperti pH tanah, kadar nitrogen, fosfor, kalium, kelembapan tanah, jenis tanaman, metode irigasi, dan faktor lainnya, sehingga petani dapat memperoleh estimasi hasil panen yang lebih akurat dan dapat diandalkan.
- Menyediakan insight dari model mengenai fitur-fitur yang paling berpengaruh terhadap hasil panen, sehingga petani atau pemangku kebijakan dapat mengoptimalkan penggunaan pupuk, metode irigasi, dan pengolahan tanah berdasarkan rekomendasi berbasis data.
- Membangun fondasi sistem prediktif yang dapat dikembangkan lebih lanjut ke dalam sistem integrasi IoT atau dashboard analitik untuk memantau kondisi tanah dan estimasi hasil panen secara periodik dan real-time.

    ### Solution statements
    - Menggunakan tiga algoritma machine learning, yaitu K-Nearest Neighbors (KNN), Random Forest (RF), dan Boosting (seperti AdaBoost atau Gradient Boosting Regressor) untuk memprediksi hasil panen berdasarkan berbagai fitur tanah dan praktik pertanian. Tujuannya adalah untuk membandingkan performa antar model dan memilih model terbaik.
    - Evaluasi model menggunakan metrik regresi yang terukur, yaitu Mean Squared Error (MSE) Untuk mengukur tingkat kesalahan prediksi.
    - Melakukan scaling pada fitur numerik untuk memastikan kestabilan hasil prediksi untuk memperoleh performa model yang lebih optimal.

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan data sintetik yang disimulasikan menggunakan bahasa pemrograman Python dengan library NumPy dan Pandas. Dataset ini terdiri dari 2.000 sampel data pertanian yang merepresentasikan berbagai kondisi tanah, teknik budidaya, serta jenis tanaman, dengan target variabel berupa Hasil Panen (dalam satuan kilogram/hektar).
Dataset ini tidak berasal dari sumber publik seperti UCI atau Kaggle, tetapi dapat diunduh melalui tautan berikut: [Dataset Hasil Panen (CSV)](https://drive.google.com/file/d/1xmYifej7NlJIxn48chac6DKaAgoAbJ0r/view?usp=sharing)

### Daftar Fitur (Variabel)

Dataset ini terdiri dari beberapa variabel fitur yang merepresentasikan kondisi tanah, teknik budidaya, serta karakteristik lahan pertanian. Berikut adalah penjelasan masing-masing fitur:

- **pH_Tanah**: Variabel numerik yang menunjukkan tingkat keasaman atau kebasaan tanah. Nilainya berada dalam rentang 4.5 hingga 8.5 dan sangat berpengaruh terhadap kemampuan tanah dalam menyerap unsur hara.

- **Nitrogen**: Merupakan variabel numerik yang menunjukkan kandungan nitrogen dalam tanah. Unsur ini penting untuk pertumbuhan tanaman karena berperan dalam proses fotosintesis dan pembentukan protein.

- **Fosfor**: Variabel numerik yang menggambarkan kandungan fosfor dalam tanah, yang berperan dalam pembentukan akar, bunga, dan buah tanaman.

- **Kalium**: Kandungan kalium dalam tanah yang juga berperan penting dalam pengaturan kadar air dan pembentukan enzim tanaman. Merupakan variabel numerik.

- **Kelembapan_Tanah**: Variabel numerik yang menggambarkan tingkat kelembapan tanah dalam satuan persen (%), yang dapat memengaruhi penyerapan nutrisi oleh akar tanaman.

- **Jenis_Tanaman**: Merupakan variabel kategorikal yang menunjukkan jenis tanaman yang dibudidayakan, yaitu salah satu dari `Padi`, `Jagung`, atau `Kedelai`.

- **Irigasi**: Variabel kategorikal yang menunjukkan metode irigasi yang digunakan, yaitu `Manual`, `Drip`, atau `Sprinkle`. Jenis irigasi dapat memengaruhi efisiensi penyiraman dan ketersediaan air.

- **Jenis_Pupuk**: Merupakan fitur kategorikal yang menunjukkan jenis pupuk yang digunakan, yaitu `Organik` atau `Anorganik`. Jenis pupuk memengaruhi kualitas dan keberlanjutan kesuburan tanah.

- **Luas_Lahan**: Variabel numerik yang menunjukkan luas lahan pertanian dalam satuan hektar. Luas lahan dapat memengaruhi total produksi hasil panen.

- **Metode_Pengolahan**: Variabel kategorikal yang menunjukkan apakah lahan diolah secara `Manual` atau menggunakan alat `Mekanis`. Metode ini berdampak pada efisiensi waktu dan tenaga kerja.

- **Waktu_Tanam_Bulan**: Variabel numerik yang menunjukkan bulan ke berapa dalam setahun penanaman dilakukan, dari 1 (Januari) hingga 12 (Desember). Waktu tanam berpengaruh terhadap curah hujan dan musim.

- **Hasil_Panen**: Ini adalah variabel target yang ingin diprediksi. Merupakan variabel numerik yang menunjukkan jumlah hasil panen dalam satuan kilogram per hektar (kg/ha).

### **Analisis Data**
Untuk memahami struktur dan karakteristik dataset secara menyeluruh, dilakukan beberapa tahap *Exploratory Data Analysis (EDA)* sebagai berikut:

1. **Pemeriksaan Informasi Umum Dataset**  
   Meliputi ringkasan tipe data dan jumlah entri menggunakan `df.info()`, serta statistik deskriptif awal melalui `df.describe()`.

2. **Identifikasi Missing Value**  
   Dicek apakah terdapat nilai kosong (missing values) pada tiap kolom, sehingga dapat diputuskan apakah perlu dilakukan imputasi atau penghapusan data.

3. **Deteksi dan Penanganan Outlier**  
   Outlier pada data numerik diidentifikasi menggunakan metode IQR (Interquartile Range), lalu divisualisasikan menggunakan boxplot untuk memudahkan interpretasi. Outlier yang ditemukan kemudian dihapus untuk menjaga kualitas analisis dan modeling.

4. **Analisis Univariat**  
   Data diklasifikasikan ke dalam fitur numerik dan kategorikal. Fitur kategorikal dianalisis dengan menghitung jumlah dan persentase tiap kategori, lalu divisualisasikan dalam bentuk diagram batang. Fitur numerik divisualisasikan menggunakan histogram untuk memahami distribusi data.

5. **Analisis Multivariat**  
   Hubungan antar fitur kategorikal dengan target variabel (`Hasil_Panen`) dianalisis menggunakan grafik bar. Untuk fitur numerik, digunakan *pairplot* dan *heatmap* korelasi untuk memahami hubungan antar fitur dan dengan target.

6. **Seleksi Fitur**  
   Beberapa fitur yang memiliki korelasi sangat rendah dengan target variabel dihapus dari dataset untuk meningkatkan efisiensi dan akurasi model prediksi.

  
## Data Preparation
Tahapan ini bertujuan untuk menyiapkan data agar siap digunakan dalam proses pemodelan machine learning. Proses ini mencakup beberapa langkah penting sebagai berikut:

- **Encoding Fitur Kategorikal**
Fitur kategorikal diubah menjadi bentuk numerik menggunakan teknik One-Hot Encoding. Hal ini bertujuan agar data kategorikal dapat diproses oleh algoritma machine learning yang membutuhkan input numerik.

- **Pembagian Data (Train-Test Split)**
Data dibagi menjadi dua bagian, yaitu data latih (training) dan data uji (testing). Pembagian ini bertujuan agar model dapat dilatih menggunakan data training dan diuji performanya menggunakan data testing yang belum pernah dilihat sebelumnya.

- **Standarisasi (Standardization)**
Fitur numerik yang memiliki rentang nilai berbeda distandarisasi sehingga memiliki skala yang sama (mean = 0 dan standar deviasi = 1). Proses ini penting untuk memastikan model yang sensitif terhadap skala fitur dapat bekerja dengan optimal.


## Modeling
Tahapan ini bertujuan untuk membangun model machine learning yang dapat memprediksi hasil berdasarkan fitur-fitur yang tersedia. Beberapa model regresi digunakan untuk membandingkan performa dan memilih yang terbaik.

- **K-Nearest Neighbor (KNN)**
KNN adalah model sederhana yang memprediksi nilai dengan melihat rata-rata dari nilai tetangga terdekat berdasarkan jarak. Model ini tidak memerlukan proses pelatihan eksplisit dan cocok untuk eksplorasi awal. Namun, KNN sensitif terhadap skala fitur dan kurang efisien pada dataset besar atau yang banyak noise.

- **Random Forest**
Random Forest adalah algoritma ensemble yang menggabungkan banyak pohon keputusan untuk menghasilkan prediksi yang stabil dan akurat. Model ini baik untuk menangani data kompleks dan non-linear serta relatif tahan terhadap overfitting. Kelemahannya adalah waktu pelatihan yang lebih lama dan interpretabilitas yang lebih rendah.

- **AdaBoost (Boosting Algorithm)**
AdaBoost adalah algoritma boosting yang meningkatkan akurasi dengan menggabungkan beberapa model lemah secara bertahap, memperbaiki kesalahan model sebelumnya. Model ini biasanya akurat namun sensitif terhadap data outlier dan noise, serta proses pelatihannya lebih kompleks.


## Evaluation
Tahap evaluasi bertujuan mengukur performa model yang telah dibuat menggunakan metrik Mean Squared Error (MSE). MSE mengukur rata-rata kuadrat selisih antara nilai aktual dan prediksi, sehingga semakin kecil MSE berarti model semakin akurat.

**Rumus Mean Squared Error (MSE):**

MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi 

---

### Proses Evaluasi

1. **Scaling Fitur Numerik**  
   Data uji distandarisasi agar fitur numerik memiliki rata-rata 0 dan varians 1, agar hasil prediksi lebih konsisten.

2. **Perhitungan MSE**  
   Dibuat dataframe untuk menyimpan nilai MSE pada data latih dan data uji dari ketiga model: KNN, Random Forest (RF), dan AdaBoost (Boosting). Nilai MSE dihitung dan disimpan untuk masing-masing model.

3. **Visualisasi Hasil MSE**  
   Hasil MSE pada data uji divisualisasikan dalam bentuk grafik batang horizontal untuk memudahkan perbandingan performa model.

4. **Pengujian Prediksi pada Satu Sampel**  
   Diambil satu sampel dari data uji untuk melakukan prediksi dengan ketiga model, lalu membandingkan hasil prediksi dengan nilai aktual.

---

## Hasil Evaluasi

| Model    | Train MSE | Test MSE |
| -------- | ------------------ | ----------------- |
| KNN      | 121.12             | 117.35            |
| RF       | 111.44             | 108.07            |
| Boosting | 109.31             | 106.73            |

Pada pengujian ini, diambil satu sampel data uji dengan nilai aktual berikut
| Index | y_true     | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-------|------------|--------------|-------------|-------------------|
| 892   | 1382.722444| 1596.1       | 1545.6      | 1578.4            |

Dari hasil di atas, model **Boosting** menunjukkan performa terbaik dengan nilai MSE paling rendah pada data uji, serta selisih yang paling kecil antara MSE train dan test, menandakan model yang lebih stabil dan tidak overfit.

Pada pengujian prediksi nilai aktual (y_true: 1382.72), model Boosting memberikan prediksi paling mendekati (1578.4), dibandingkan RF (1545.6) dan KNN (1596.1), sehingga Boosting dipilih sebagai model terbaik untuk dataset hasil panen ini.

---

## Kesimpulan

Model Boosting memberikan keseimbangan terbaik antara akurasi dan kestabilan prediksi pada data baru. Dengan model ini, prediksi hasil panen dapat dilakukan dengan tingkat kesalahan yang lebih rendah, membantu pengambilan keputusan yang lebih baik dalam konteks pertanian atau analisis data hasil panen.
