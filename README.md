# Laporan Proyek Machine Learning - Muhammad Nanda Setiawan

## Domain Proyek

Seiring berjalannya waktu, peningkatan konsumsi daya listrik semakin meningkat. Disisi lain, perusahaan-perusahaan elektronik juga berlomba meningkatkan efisiensi produk mereka. Untuk itu, perusahaan listrik perlu upaya untuk mengontrol dan mengelola tenaga listrik mereka secara efektif. Mereka perlu mengetahui energi yang dihasilkan dan dikonsumsi untuk menyeimbangkan produksi dan konsumsi, menurunkan biaya produksi, dan mengendalikan perencanaan ke depan. Salah satu cara penyelesaian permasalahan ini dapat dilakukan dengan melakukan prediksi penggunaan daya listrik. Prediksi ini dapat dilakukan machine learning [1]. Pada project ini akan dilakukan prediksi konsumsi daya listrik dengan mengambil studi kasus pada salah salah satu wilayah di kota Tetouan, Maroko.

## Business Understanding

### Problem Statements  
- Bagaimana pola konsumsi daya listrik pada dataset studi kasus?
- Bagaimana membuat prediksi konsumsi daya listrik untuk beberapa jam ke depan?

### Goals  
- Mengetahui pola konsumsi daya listrik pada dataset studi kasus
- Melakukan prediksi konsumsi daya listrik untuk beberapa jam ke depan?

    ### Solution statements  
    
    Pada project ini, menggunakana model baseline yang digunakan sebagai pembanding:
    1. _Multi Step Baseline_:  Model yang memprediksi n-time point ke depan berdasarkan nilai terakhir.
    2. _Repeat Baseline_: Model yang mengasumsikan n-timep point ke depan sama dengan n-time point sebelumnya.
    
    Adapun model machine learning learning yang diusulkan dalam project ini:  
    1. RNN (_Recurrent Neural Network_) adalah .
    2. LSTM (_Long Short Term Memory_) adalah jenis jaringan saraf berulang yang mampu mempelajari ketergantungan urutan dalam masalah prediksi urutan.
    3. GRU (_Gated Recurrent Unit_) merupakan salah satu jenis Recurrent Neural Network (RNN) yang pada kasus tertentu memiliki kelebihan dibandingkan long short term memory (LSTM). GRU menggunakan lebih sedikit memori dan lebih cepat daripada LSTM.

    Kedua model tersebut akan dibandingkan dengan model baseline. Adapun metric yang digunakan, yaitu:
    1. RMSE (_Root Mean Squared Error_)
    2. MAE (_Mean Absolute Error_)


## Data Understanding  

Dataset yang dipakai didapat dari sebuah platform penyedia dataset untuk machine learning, yaitu UCI Dataset. Untuk proyek ini, dataset yang saya pakai yaitu [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Power+consumption+of+Tetouan+city). 
Berikut ini deskripsi dari variable/feature yang terdapat dalam dataset:

* Date Time: informasi waktu data terekam (per 10 menit)
* Temperature: temperatur / suhu yang tercatat pada sensor
* Humidity: kelembapan yang tercatat pada sensor
* Wind Speed of Tetouan City: kecepatan angin yang tercatat pada sensor
* General Diffuse Flows: aliran difus secara global
* Diffuse Flows : aliran difus lokal
* Power Consumption of zone 1 Tetouan City : Konsumsi daya listrik pada zona 1 Kota Tetouan
* Power Consumption of zone 2 Tetouan City : Konsumsi daya listrik pada zona 2 Kota Tetouan
* Power Consumption of zone 3 Tetouan City : Konsumsi daya listrik pada zona 3 Kota Tetouan

Dari fitur yang tersedia dilakukan _Feature Engineering_ untuk mengekstrak fitur `month` dan `hour` dari `Date Time`. Kemudian mencoba melakukan analisis univariate dan multivariate untuk lebih memahami pola pada data dan pola konsumsi daya listrik pada setiap zona. Tahapan analisis yang dilakukan diantaranya:
* Pengecekan historis konsumsi daya listrik pada tiap zona
* pengecekan distribusi fitur
* Pengecekan korelasi fitur dan konsumsi listrik pada tiap zona

Dari hasil analisis yang dilakukan didapat kesimpulan berupa:
* Pola konsumsi daya listrik pada ketiga zona di Kota Tetuoan hanya dari awal sampai pertengahan tahun. Ini akan sulit jika melakukan pemodelan timeseries dengan data testing yang berada di bulan-bulan mendekati akhir tahun.
* Hanya satu fitur yang memiliki distribusi mendekati normal. Disisi lain, fitur baru hasil _feature engineering_ memiliki distribusi yang cenderung akan seragam.
* Beberapa fitur memiliki tren yang mirip terhadap penggunaan daya listrik di setiap zona. Namun, secara korelasi linear nilainya tidak begitu identik. 

Dari hasil Analisis tersebut, disimpulkan data memiliki distribusi yang beragam, sehingga perlu menggunakan model yang robust/advance untuk melakukan forecasting. Adanya perbedaan distribusi konsumsi daya listrik tiap zona mendekati akhir tahun, mungkin menyebabkan model yang bekerja baik pada suatu zona tidak begitu baik pada zona lainnya. Untuk itu, pada project ini hanya menggunakan salah satu zona sebagai studi kasus.

## Data Preparation  

Tahapan persiapan data: 
1. _Feature Selection_:   
    * Hanya menggunakan `temperature`, `humidity`, `wind_speed`, `month`, dan `hour` sebagai prediktor.  
    * Dari tiga zona Kota Tetouan, dipilih salah satu (Zona 1) sebagai label untuk studi kasus pada project ini. 
2. Pembuatan _Window_:  
    * Dari data awal tabular, dibentuk ke dalam _window_ dengan panjang 8 jam / 48 titik waktu (karena sumber data per 10 menit)
3. Splitting Data dibagi menjadi:  
    * data train: data untuk melatih model  
    * data validasi: data yang digunakan sebagai validasi dalam proses pembautan model  
    * data tes: data yang digunakan  
4. _Data Scaling_:
    * Scaling data dengan teknik standardisasi data: 
        * $$z_i = \frac{x_i - \mu_{dt\_train}}{\sigma_{dt\_train}},  i \in \{dt\_train, dt\_validasi, dt\_tes \}$$  
        * $$\mu$$ dan $$\sigma$$ hanya menggunakan data train untuk mencegah _data leakage_


## Modeling

Pada tahap ini, random seed diatur pada angka tertentu untuk lebih memudahkan _tracking_ dan _tuning model_. Adapun _hyperparameter_ yang diatur, yaitu nilai _epoch_, jumlah _neuron_, fungsi aktivasi, initialisi kernel dan arsitektur dasar model. Arsitektur dasar yang digunakan, terdiri dari layer custom (rnn/lstm/gru) diikuti dengan layer `BatchNormalization`, `GlobalAveragePooling1D`, `Dense`, dan `Reshape` untuk _forecasting_ konsumsi daya listrik n-jam ke depan. Pada setiap model dilakukan _plot learning precess_ -nya untuk melihat model yang mampu belajar dengan baik. Berikut ini hasil _learning process_ tiap model:
* [![RNN Learning Process](https://drive.google.com/file/d/1uwQCosRVPYRRZIObFyi7hiDemLgqYha8/view?usp=sharing "RNN Learning Process")](https://drive.google.com/file/d/1uwQCosRVPYRRZIObFyi7hiDemLgqYha8/view?usp=sharing)  
* [![LSTM Learning Process](https://drive.google.com/file/d/1uw8sXDH0P994zqql7TT5LH3q5krqRZAH/view?usp=sharing "LSTM Learning Process")](https://drive.google.com/file/d/1uw8sXDH0P994zqql7TT5LH3q5krqRZAH/view?usp=sharing)  
* [![GRU Learning Process](https://drive.google.com/file/d/1uy8yxNz_NQTlpCKDBRmh_jspAwrShLm6/view?usp=sharing "GRU Learning Process")](https://drive.google.com/file/d/1uy8yxNz_NQTlpCKDBRmh_jspAwrShLm6/view?usp=sharing)


## Evaluation

Dari beberapa model yang digunakan, berikut ini perbandingan hasil metric MAE dan RMSE pada data testing.  
[![Hasil Metrik](https://drive.google.com/file/d/1v9YCkYrSVc1uoB3hpi4JZPOkVZ7CVfvr/view?usp=sharing "Hasil Metrik")](https://drive.google.com/file/d/1v9YCkYrSVc1uoB3hpi4JZPOkVZ7CVfvr/view?usp=sharing)  
Terlihat model GRU berhasil mengungguli model lainnya dalam melakukan _forecasting_, dengan skor MAE: 0.37 dan RMSE: 0.52. 

Berikut ini hasil sampel _forecasting_ pada data tes dengan model GRU  
[![GRU Prediction](https://drive.google.com/file/d/1v7AI3XKbo0w_dU19P7qTTni8rP0IehJZ/view?usp=sharing "Gru Prediction")](https://drive.google.com/file/d/1v7AI3XKbo0w_dU19P7qTTni8rP0IehJZ/view?usp=sharing)


## References  

[1] Salam, A., & El Hibaoui, A. (2018, December). Comparison of Machine Learning Algorithms for the Power Consumption Prediction:-Case Study of Tetouan cityâ€“. In 2018 6th International Renewable and Sustainable Energy Conference (IRSEC) (pp. 1-5). IEEE.
