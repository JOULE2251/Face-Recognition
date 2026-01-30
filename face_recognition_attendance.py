import cv2
import os
import numpy as np
from datetime import datetime
import csv
import json
import mysql.connector
import threading
import time 


# -------------------- Fungsi Tentukan Shift --------------------
def tentukan_shift(jam_str):
    jam = datetime.strptime(jam_str, "%H:%M:%S").time()

    if jam >= datetime.strptime("05:00:00", "%H:%M:%S").time() and jam <= datetime.strptime("08:00:00", "%H:%M:%S").time():
        return "Shift 1", "H"  # Hadir Tepat Waktu
    elif jam > datetime.strptime("08:00:00", "%H:%M:%S").time() and jam <= datetime.strptime("14:00:00", "%H:%M:%S").time():
        return "Shift 1", "T"  # Terlambat
    elif jam > datetime.strptime("14:00:00", "%H:%M:%S").time() and jam <= datetime.strptime("16:00:00", "%H:%M:%S").time():
        return "Shift 2", "H"
    elif jam > datetime.strptime("16:00:00", "%H:%M:%S").time() and jam <= datetime.strptime("21:00:00", "%H:%M:%S").time():
        return "Shift 2", "T"
    else:
        return "Tidak Dikenal", "A"  # Absen


# -------------------- Fungsi Simpan ke Database --------------------
def simpan_ke_database(nama, tanggal, shift, status, jam_masuk="", jam_pulang=""):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="liface_db"
        )
        cursor = conn.cursor()

        # Cek apakah sudah ada data presensi untuk nama dan tanggal ini
        cursor.execute("SELECT * FROM presensi WHERE nama=%s AND tanggal=%s", (nama, tanggal))
        result = cursor.fetchone()

        if result:
            # Sudah presensi, update jam pulang
            cursor.execute("""
                UPDATE presensi SET jam_pulang=%s WHERE nama=%s AND tanggal=%s
            """, (jam_pulang, nama, tanggal))
        else:
            # Belum ada → buat data presensi baru
            cursor.execute("""
                INSERT INTO presensi (nama, tanggal, shift, status, jam_masuk, jam_pulang)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (nama, tanggal, shift, status, jam_masuk, jam_pulang))

        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Data presensi disimpan ke database!")

    except Exception as e:
        print("[ERROR] Gagal simpan ke database:", e)


# -------------------- Fungsi Tandai Absen Harian --------------------
def tandai_absen_harian():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="liface_db"
        )
        cursor = conn.cursor()

        # Ambil daftar semua karyawan
        cursor.execute("SELECT nama_lengkap FROM karyawan")
        semua_nama = [row[0] for row in cursor.fetchall()]

        # Cek siapa yang belum presensi hari ini
        tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")
        cursor.execute("SELECT nama FROM presensi WHERE tanggal=%s", (tanggal_hari_ini,))
        sudah_presensi = [row[0] for row in cursor.fetchall()]

        belum_presensi = set(semua_nama) - set(sudah_presensi)

        for nama in belum_presensi:
            cursor.execute("""
                INSERT INTO presensi (nama, tanggal, shift, status, jam_masuk, jam_pulang)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (nama, tanggal_hari_ini, "-", "A", "", ""))
            print(f"[INFO] {nama} ditandai sebagai ABSEN (A) untuk tanggal {tanggal_hari_ini}")

        conn.commit()
        cursor.close()
        conn.close()

        update_json()

    except Exception as e:
        print("[ERROR] Gagal tandai absen:", e)


# -------------------- Fungsi Simpan Presensi (dengan logika per hari) --------------------
def simpan_presensi(nama):
    filename = "presensi.csv"
    now = datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    jam = now.strftime("%H:%M:%S")

    shift, status = tentukan_shift(jam)
    file_exists = os.path.isfile(filename)
    data_lama = []

    sudah_presensi_hari_ini = False

    if file_exists:
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_lama.append(row)
                if row["Nama"] == nama and row["Tanggal"] == tanggal:
                    sudah_presensi_hari_ini = True

    if sudah_presensi_hari_ini:
        # Update jam pulang
        for row in data_lama:
            if row["Nama"] == nama and row["Tanggal"] == tanggal:
                row["Jam Pulang"] = jam
        print(f"[INFO] {nama} presensi pulang jam {jam}")
        simpan_ke_database(nama, tanggal, shift, status, jam_pulang=jam)
    else:
        # Tambah data baru (presensi masuk)
        new_row = {
            "Nama": nama,
            "Tanggal": tanggal,
            "Shift": shift,
            "Status": status,
            "Jam Masuk": jam,
            "Jam Pulang": ""
        }
        data_lama.append(new_row)
        print(f"[INFO] {nama} presensi masuk jam {jam} ({shift}, {status})")
        simpan_ke_database(nama, tanggal, shift, status, jam_masuk=jam)

    # Simpan ulang CSV
    with open(filename, "w", newline="") as f:
        fieldnames = ["Nama", "Tanggal", "Shift", "Status", "Jam Masuk", "Jam Pulang"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data_lama)

    update_json()

# -------------------- Fungsi Update JSON (ambil langsung dari database) --------------------
def update_json():
    json_file = "presensi.json"
    data = []

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="liface_db"
        )
        cursor = conn.cursor(dictionary=True)

        # Ambil semua data presensi bergabung dengan data karyawan
        cursor.execute("""
            SELECT 
                p.id,
                k.nama_lengkap AS Nama,
                p.tanggal AS Tanggal,
                p.shift AS Shift,
                p.status AS Status,
                p.jam_masuk AS Jam_Masuk,
                p.jam_pulang AS Jam_Pulang
            FROM presensi p
            JOIN karyawan k ON p.nama = k.nama_lengkap
            ORDER BY p.tanggal DESC, k.nama_lengkap ASC
        """)

        data = cursor.fetchall()

        cursor.close()
        conn.close()

        # Simpan hasil query ke file JSON
        with open(json_file, "w") as jf:
            json.dump(data, jf, indent=4, default=str)

        print("[INFO] presensi.json diperbarui dari database!")

    except Exception as e:
        print("[ERROR] Gagal memperbarui JSON dari database:", e)

# -------------------- Training Data --------------------
dataset_path = "dataset"
faces = []
labels = []
label_dict = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(current_id)

    current_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# -------------------- Real-Time Recognition --------------------
def auto_absen_midnight():
    """Thread background untuk auto-absen jam 00:00 setiap hari."""
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == "00:00":
            print("[AUTO] Mengecek absensi harian...")
            tandai_absen_harian()
            print("[AUTO] Proses tandai absen selesai.")
            # Tunggu 60 detik supaya gak ke-trigger berulang kali di menit yang sama
            time.sleep(60)
        time.sleep(30)  # Cek setiap 30 detik

# Jalankan thread background auto absen
threading.Thread(target=auto_absen_midnight, daemon=True).start()

# -------------------- Real-Time Recognition --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = "Face Recognition Attendance"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)
cv2.moveWindow(window_name, 200, 100)

print("[INFO] Mulai deteksi wajah... Tekan ESC untuk keluar.")

presensi_dicatat = set()
deteksi_waktu = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)

        if confidence < 70:
            nama = label_dict[label]
            cv2.putText(frame, nama, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if nama not in deteksi_waktu:
                deteksi_waktu[nama] = time.time()

            durasi = time.time() - deteksi_waktu[nama]
            progress = min(int((durasi / 5) * 100), 100)

            bar_x1, bar_y1 = x, y + h + 20
            bar_x2 = x + int(w * (progress / 100))
            cv2.rectangle(frame, (x, bar_y1), (x + w, bar_y1 + 10), (100, 100, 100), 2)
            cv2.rectangle(frame, (x, bar_y1), (bar_x2, bar_y1 + 10), (0, 255, 255), -1)
            cv2.putText(frame, f"{progress}%", (x + w + 10, bar_y1 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Catat presensi setelah progress penuh
            if progress >= 100 and nama not in presensi_dicatat:
                simpan_presensi(nama)
                presensi_dicatat.add(nama)

        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow(window_name, frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # ESC untuk keluar
        break

# Saat program berhenti, perbarui JSON terakhir kali
update_json()

cap.release()
cv2.destroyAllWindows()