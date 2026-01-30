# Life Media Face Recognition Presence System (LIFACE)
LIFACE adalah sistem presensi karyawan berbasis pengenalan wajah yang dikembangkan untuk PT SaranaInsan MudaSelaras. 
Sistem ini mencatat kehadiran secara otomatis dan real-time menggunakan kamera dan menyediakan dashboard monitoring berbasis web.

# Fitur Utama
- Deteksi dan pengenalan wajah secara real-time
- Presensi otomatis (jam masuk & jam pulang)
- Dashboard monitoring kehadiran karyawan
- Manajemen data karyawan (tambah, ubah, hapus)
- Rekap dan export data presensi
- Hak akses admin

# Teknologi yang digunakan
- Python
- Flask
- OpenCV & face_recognition
- MySQL
- HTML, CSS, JavaScript
- Jinja2
- Werkzeug

# Arsitektur Sistem
- Frontend: Web (HTML, CSS, JS, Jinja2)
- Backend: Flask (Python)
- Face Recognition: OpenCV & face_recognition
- Database: MySQL
- Perangkat: Webcam & Jetson

# Kebutuhan Sistem
## Software
- Python 3.8+
- MySQL Server
- Browser (Chrome / Firefox)

## Hardware
- Webcam minimal 720p
- Perangkat Jetson / PC Server
- RAM minimal 4 GB

# Instalasi
1. Clone repository
- https://github.com/JOULE2251/Face-Recognition.git
- cd liface

2. Install dependency
- python -m venv fcenv
- source fcenv/Scripts/activate
- pip install -r requirements.txt

3. Konfigurasi database
- Buat database MySQL
- Atur koneksi di file config / app.py

4. Download file trainer.yml pada link google drive berikut ini
- https://drive.google.com/drive/folders/1PtvG3n0ZhU7Qpdn9Oq7raKfpjqsZh6_S?usp=sharing

5. Jalankan aplikasi
python app.py

# Cara Penggunaan
1. Admin login ke dashboard
2. Admin menambahkan data karyawan dan foto wajah
3. Karyawan melakukan presensi dengan menghadapkan wajah ke kamera
4. Sistem mencatat jam masuk/pulang otomatis
5. Admin memonitor dan merekap data presensi melalui dashboard

# Hak Akses
- Admin: Login, manajemen karyawan, monitoring & rekap presensi
- User (Karyawan): Melakukan presensi dan melihat data presensi

# Batasan Sistem
- Tidak terintegrasi dengan sistem payroll
- Tidak mendukung presensi GPS
- Tidak tersedia aplikasi mobile (Android/iOS)
- Fokus pada face recognition saja

# Tim Pengembang
- Nisa Afriyani
- Julian Duwi Prasetya
- Wahyu Saofi
- Ahmad Widad Saksana

Program Studi Teknologi Informasi  
Universitas 'Aisyiyah Yogyakarta  
Tahun 2025
