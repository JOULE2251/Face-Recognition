import cv2
import os
import time
import sys
import mysql.connector
import webbrowser

# ===========================
# Ambil data dari argumen
# ===========================
if len(sys.argv) < 4:
    print("[ERROR] Argumen tidak lengkap. Gunakan: python detect_face_video.py <NIP> <NAMA_LENGKAP> <DIVISI>")
    sys.exit(1)

nip = sys.argv[1]
person_name = sys.argv[2]
division = sys.argv[3]

# ===========================
# Simpan data ke database
# ===========================
def simpan_ke_db(nip, nama, divisi):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="liface_db"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO karyawan (nip, nama_lengkap, divisi)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE nama_lengkap=%s, divisi=%s
        """, (nip, nama, divisi, nama, divisi))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Data karyawan {nama} disimpan ke database.")
    except Exception as e:
        print("[ERROR] Gagal menyimpan data ke database:", e)

simpan_ke_db(nip, person_name, division)

# ===========================
# Persiapan folder dataset
# ===========================
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

person_path = os.path.join(dataset_path, person_name)
if not os.path.exists(person_path):
    os.makedirs(person_path)

# ===========================
# Mulai kamera
# ===========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak dapat dibuka!")
    sys.exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 1
max_images = 200
IMG_SIZE = (200, 200)

print(f"[INFO] Mulai mengambil dataset untuk {person_name}...")

while True:
    ret, img = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, IMG_SIZE)

        file_name = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(file_name, face_resized)
        print(f"[INFO] Gambar {count}/{max_images} disimpan: {file_name}")

        count += 1
        time.sleep(0.1)

        if count > max_images:
            print(f"[INFO] Dataset {person_name} selesai dibuat ({max_images} gambar).")
            cap.release()

            done_img = img.copy()
            cv2.putText(done_img, "TRAINING SELESAI!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('img', done_img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

            dashboard_url = "http://localhost:5000/dashboard-admin"
            print(f"[INFO] Membuka {dashboard_url}...")
            webbrowser.open(dashboard_url)
            sys.exit()

    cv2.putText(img, f"Nama: {person_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Foto ke: {count}/{max_images}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Tekan ESC untuk berhenti manual", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        print("[INFO] Pengambilan dataset dihentikan manual.")
        break

cap.release()
cv2.destroyAllWindows()
