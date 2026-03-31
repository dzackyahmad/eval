# Cara Penggunaan

1. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

2. (Opsional) Upload Ground Truth jika ingin mengganti default.

3. Upload file prediksi (CSV) dengan format:

   ```
   id,label
   test_001,fake_screen
   ```

4. Hasil akan otomatis muncul:

   * Tabel perbandingan (SAMA / BEDA)
   * Highlight merah untuk error
   * Akurasi (%)
   * Detail gambar

5. Download hasil evaluasi jika diperlukan.
