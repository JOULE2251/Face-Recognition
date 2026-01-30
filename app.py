from flask import Flask, jsonify, render_template
from flask import Flask, jsonify, request, redirect, session, url_for, Response
from flask_cors import CORS
import cv2
import os
import numpy as np
import time
import subprocess
import threading
import mysql.connector
from werkzeug.security import check_password_hash
from datetime import datetime
import calendar
import json
import csv

app = Flask(__name__)
CORS(app)
app.secret_key = 'lifeMediaLoginSystem_2025'

# === Koneksi ke Database ===
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="liface_db"
    )

# === Fungsi Rekap Bulanan (JSON / Diringkas per Bulan) ===
def load_rekap_bulanan():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT DATE_FORMAT(tanggal, '%Y-%m') AS bulan, COUNT(*) AS total
        FROM presensi
        GROUP BY DATE_FORMAT(tanggal, '%Y-%m')
        ORDER BY bulan DESC
    """)

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    return data

# === Route Halaman Login (GET) ===
@app.route('/login', methods=['GET'])
def show_login_page():
    return render_template('Login.html')

# === Proses Login (POST) ===
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and password == user['password']:
        # ✅ Simpan status login ke session
        session['username'] = user['username']
        session['logged_in'] = True
        return redirect(url_for('dashboard_admin'))
    else:
        # Jika salah login
        return render_template('Login.html', error="Incorrect username or password.")

# === Route Logout ===
@app.route('/logout')
def logout():
    # ✅ Hapus session saat logout
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('show_login_page'))

# === Route Dashboard Admin ===
@app.route('/dashboard-admin')
def dashboard_admin():
    # ✅ Cek apakah sudah login
    if not session.get('logged_in'):
        return redirect(url_for('show_login_page'))  # Kalau belum login, arahkan ke login

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM presensi ORDER BY tanggal DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('DashboardAdmin.html', presensi=data)

# === Route Rekap Bulanan Admin ===
@app.route('/rekap')
def rekap_bulanan():
    if not session.get('logged_in'):
        return redirect(url_for('show_login_page'))

    data = load_rekap_bulanan()
    return render_template('DashboardAdmin.html', data=data)

@app.route('/addpersonel')
def add_personel():
    # 🔒 Cek apakah admin sudah login
    if not session.get('logged_in'):
        return redirect(url_for('login'))  # kalau belum login, paksa login dulu
    
    # kalau sudah login, tampilkan halaman tambah personel
    return render_template('addpersonel.html')

@app.route('/train_face', methods=['POST'])
def train_face():

    if not session.get('logged_in'):
        return jsonify({
            "error": "Unauthorized. Please log in first."
        }), 401
    global training_mode, training_target, training_count, training_folder

    nip = request.form.get('nip')
    nama_lengkap = request.form.get('nama_lengkap')
    divisi = request.form.get('divisi')

    print(f"[DEBUG] Data received: {nip}, {nama_lengkap}, {divisi}")

    # Simpan ke database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO karyawan (nip, nama_lengkap, divisi)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE nama_lengkap=%s, divisi=%s
        """, (nip, nama_lengkap, divisi, nama_lengkap, divisi))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Data {nama_lengkap} saved to the database.")
    except Exception as e:
        print("[ERROR] Failed to save data:", e)
        return jsonify({"error": str(e)}), 500

    # Siapkan folder dataset
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    training_folder = os.path.join(dataset_path, nama_lengkap)
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    # Aktifkan mode training
    training_mode = True
    training_target = nama_lengkap
    training_count = 0

    print(f"[INFO] Training mode is active for {nama_lengkap}.")
    return jsonify({"message": f"Start training for {nama_lengkap}..."}), 200


# === Halaman Dashboard User ===
@app.route('/')
def serve_dashboard_user():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # ambil data presensi awalnya (misal bulan Oktober, biar ada data default)
    cursor.execute("""
        SELECT * FROM presensi 
        WHERE MONTH(tanggal) = 10 
        ORDER BY tanggal DESC
    """)
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('DashboardUser.html', presensi=data)

@app.route('/api/presensi/<bulan>/<tahun>', methods=['GET'])
def get_presensi_by_bulan(bulan, tahun):
    # Mapping nama bulan ke angka
    bulan_mapping = {
        'january': 1, 'february': 2, 'march': 3,
        'april': 4, 'mey': 5, 'june': 6,
        'july': 7, 'agust': 8, 'september': 9,
        'october': 10, 'november': 11, 'december': 12
    }

    bulan_num = bulan_mapping.get(bulan.lower())
    if not bulan_num:
        return jsonify([])

    # Koneksi ke database MySQL
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='liface_db'
    )
    cursor = conn.cursor(dictionary=True)

    # Ambil data berdasarkan bulan dan tahun
    query = """
        SELECT nama, tanggal, status
        FROM presensi
        WHERE MONTH(tanggal) = %s
        AND YEAR(tanggal) = %s
        ORDER BY nama, tanggal
    """
    cursor.execute(query, (bulan_num, tahun))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    if not results:
        return jsonify([])

    data_terstruktur = {}
    for row in results:
        nama = row['nama']
        tanggal = row['tanggal'].day
        status = row['status']
        if nama not in data_terstruktur:
            data_terstruktur[nama] = {}
        data_terstruktur[nama][tanggal] = status

    hasil = []
    for nama, presensi in data_terstruktur.items():
        total_hadir = sum(1 for s in presensi.values() if s in ['H', 'T'])
        hasil.append({
            "nama": nama,
            "presensi": presensi,
            "total_hadir": total_hadir
        })

    return jsonify(hasil)

# === API untuk ambil data presensi (Dashboard User) ===
@app.route('/api/presensi-user')
def get_presensi_user():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT id, nama, jam_masuk, jam_pulang, tanggal
        FROM presensi
        ORDER BY tanggal DESC
    """)
    data = cursor.fetchall()

    # Konversi jam_masuk dan jam_pulang ke string agar bisa di-JSON-kan
    for row in data:
        if isinstance(row.get('jam_masuk'), (bytes, bytearray)):
            row['jam_masuk'] = row['jam_masuk'].decode()
        elif row.get('jam_masuk') is not None:
            row['jam_masuk'] = str(row['jam_masuk'])

        if isinstance(row.get('jam_pulang'), (bytes, bytearray)):
            row['jam_pulang'] = row['jam_pulang'].decode()
        elif row.get('jam_pulang') is not None:
            row['jam_pulang'] = str(row['jam_pulang'])

    cursor.close()
    conn.close()

    return jsonify(data)

@app.route('/user_presensi')
def user_presensi():
    # Ambil data presensi dari database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, nama, jam_masuk, jam_pulang
        FROM presensi
        ORDER BY tanggal DESC
    """)
    data_presensi = cursor.fetchall()
    cursor.close()
    conn.close()

    # Tampilkan halaman presensi
    return render_template('UserPresensi.html', data_presensi=data_presensi)

# ==========================
# === FACE RECOGNITION STREAM UNTUK WEB ===
# ==========================

# === Variabel global untuk mode training ===
training_mode = False
training_target = None
training_count = 0
training_max = 200
training_folder = None

import cv2
import os
import numpy as np
import time
from datetime import datetime

dataset_path = "dataset"
faces = []
labels = []
label_dict = {}
current_id = 0

# === Inisialisasi recognizer ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")
    print("[INFO] Face model successfully loaded from trainer.yml")
else:
    print("[WARNING] There is no trainer.yml model yet, using manual dataset...")

# === Jika belum ada file trainer.yml, latih dari dataset ===
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

if faces and not os.path.exists("trainer.yml"):
    recognizer.train(np.array(faces), np.array(labels))
    recognizer.save("trainer.yml")
    print("[INFO] The face model is trained from scratch and saved to trainer.yml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

# --- Fungsi bantu untuk simpan presensi ke database ---
def simpan_presensi_ke_db(nama):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        tanggal = datetime.now().strftime("%Y-%m-%d")
        jam = datetime.now().strftime("%H:%M:%S")

        cursor.execute("SELECT * FROM presensi WHERE nama=%s AND tanggal=%s", (nama, tanggal))
        result = cursor.fetchone()

        if result:
            cursor.execute("""
                UPDATE presensi SET jam_pulang=%s WHERE nama=%s AND tanggal=%s
            """, (jam, nama, tanggal))
        else:
            cursor.execute("""
                INSERT INTO presensi (nama, tanggal, shift, status, jam_masuk, jam_pulang)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (nama, tanggal, "-", "H", jam, ""))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Presence saved for {nama} ({jam})")
    except Exception as e:
        print("[ERROR] Failed to save attendance:", e)


# --- Fungsi utama streaming video ---
def generate_frames():
    global training_mode, training_target, training_count, training_max, training_folder, recognizer
    global label_dict, recognizer 

    presensi_dicatat = set()
    deteksi_waktu = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)

        # === MODE TRAINING (ambil dataset dari browser) ===
        if training_mode:
            for (x, y, w, h) in faces_detected:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                file_name = os.path.join(training_folder, f"{training_count+1}.jpg")
                cv2.imwrite(file_name, face_resized)
                training_count += 1

                cv2.putText(frame, f"Training: {training_target} ({training_count}/{training_max})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)

                if training_count >= training_max:
                    training_mode = False
                    print(f"[INFO] Training for {training_target} finished ({training_count} picture).")

                    # === Retrain model otomatis setelah dataset baru ===
                    faces, labels, label_dict = [], [], {}
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

                    if faces:
                        recognizer = cv2.face.LBPHFaceRecognizer_create()
                        recognizer.train(np.array(faces), np.array(labels))
                        recognizer.save("trainer.yml")
                        print("[INFO] The face model is updated after training.")
            # kirim frame ke browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue

        # === MODE NORMAL (PRESENSI) ===
        for (x, y, w, h) in faces_detected:
            roi_gray = gray[y:y+h, x:x+w]

            # Standarisasi sebelum predict
            roi_gray = cv2.resize(roi_gray, (200, 200))
            roi_gray = cv2.equalizeHist(roi_gray)

            if recognizer is not None:
                label, confidence = recognizer.predict(roi_gray)

                # ===== Akurasi LBPH =====
                # <50  = yakin (match)
                # 50-90 = ragu (maybe)
                # >90  = unknown

                THRESHOLD = 50

                if confidence < THRESHOLD:
                    nama = label_dict.get(label, "Unknown")
                    color = (0, 255, 0)  # hijau
                else:
                    nama = "Unknown"
                    color = (0, 0, 255)  # merah

                # Display teks
                cv2.putText(frame, f"{nama} ({confidence:.1f})",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # === Presensi hanya untuk 'nama valid', bukan Maybe/Unknown ===
                if nama not in ["Unknown", "Maybe"]:
                    if nama not in deteksi_waktu:
                        deteksi_waktu[nama] = time.time()

                    durasi = time.time() - deteksi_waktu[nama]
                    progress = min(int((durasi / 5) * 100), 100)

                    bar_x1, bar_y1 = x, y + h + 20
                    bar_x2 = x + int(w * (progress / 100))
                    cv2.rectangle(frame, (x, bar_y1), (x + w, bar_y1 + 10), (100, 100, 100), 2)
                    cv2.rectangle(frame, (x, bar_y1), (bar_x2, bar_y1 + 10), (0, 255, 255), -1)
                    cv2.putText(frame, f"{progress}%", (x+w+10, bar_y1+10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                    if progress >= 100 and nama not in presensi_dicatat:
                        simpan_presensi_ke_db(nama)
                        presensi_dicatat.add(nama)
                        print(f"[DEBUG] Attendance is automatically saved for {nama}")

            else:
                cv2.putText(frame, "No Model", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

        # Encode frame dan kirim ke browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# === ROUTE UNTUK STREAM VIDEO ===
@app.route('/video_feed')
def video_feed():
    from flask import Response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# === ROUTE UNTUK MULAI TRAINING DARI WEB ===
@app.route('/start_training', methods=['POST'])
def start_training():
    global training_mode, training_target, training_count, training_folder

    nama = request.form.get('nama')
    if not nama:
        return jsonify({"error": "Name cannot be blank"}), 400

    dataset_path = "dataset"
    training_folder = os.path.join(dataset_path, nama)
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)

    training_mode = True
    training_target = nama
    training_count = 0

    print(f"[INFO] Start facial training for {nama}")
    return jsonify({"message": f"Facial training for {nama} started"})

@app.route('/api/presensi-user-filter')
def get_presensi_user_filter():
    date = request.args.get("date")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    if date:
        cursor.execute("""
            SELECT id, nama, jam_masuk, jam_pulang, tanggal
            FROM presensi
            WHERE tanggal = %s
            ORDER BY jam_masuk ASC
        """, (date,))
    else:
        cursor.execute("""
            SELECT id, nama, jam_masuk, jam_pulang, tanggal
            FROM presensi
            ORDER BY tanggal DESC
        """)

    data = cursor.fetchall()
    cursor.close()
    conn.close()

    for row in data:
        row["jam_masuk"] = str(row["jam_masuk"]) if row["jam_masuk"] else "-"
        row["jam_pulang"] = str(row["jam_pulang"]) if row["jam_pulang"] else "-"

    return jsonify(data)

@app.route('/api/presensi_realtime')
def presensi_realtime():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, nama, jam_masuk, jam_pulang, tanggal FROM presensi ORDER BY id DESC")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # pastikan tipe data serializable ke JSON (jam bisa None)
    for row in data:
        row['jam_masuk'] = str(row['jam_masuk']) if row.get('jam_masuk') else "-"
        row['jam_pulang'] = str(row['jam_pulang']) if row.get('jam_pulang') else "-"
        row['tanggal'] = str(row['tanggal']) if row.get('tanggal') else "-"

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)