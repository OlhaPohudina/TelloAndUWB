import time
import threading
import datetime
import csv
import cv2 as cv
import numpy as np
import math
import os
import atexit
import json
import keyboard
from djitellopy import Tello

# === Initialization ===
# Parametri per il controllo del drone
X_DIST = 0.7  # Distanza target per la stabilizzazione
Y_DIST = 0    # Distanza laterale (nessuna correzione in Y)
ANGLE = 0     # Angolo di riferimento

tello = Tello()
tello.connect()
time.sleep(2)
position = [0.0, 0.0, 0.0]
last_time = time.time()
stop_flag = threading.Event()

# === Log file ===

log_folder = r"D:\tello\Alessia"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
imu_filename = os.path.join(log_folder, f"imu_log_{timestamp}.csv")
command_log_filename = os.path.join(log_folder, f"command_log_{timestamp}.txt")

imu_file = open(imu_filename, mode='w', newline='')
imu_writer = csv.writer(imu_file)
imu_writer.writerow(['Time', 'Ax', 'Ay', 'Az', 'Sx', 'Sy', 'Sz', 'x', 'y', 'z'])

# === Logging IMU ===
def log_imu_data():
    global last_time
    try:
        now = time.time()
        dt = now - last_time
        last_time = now
        tello.get_current_state()
        ax = tello.get_acceleration_x()
        ay = tello.get_acceleration_y()
        az = tello.get_acceleration_z()
        sx = tello.get_speed_x()
        sy = tello.get_speed_y()
        sz = tello.get_speed_z()
        position[0] += sx * dt
        position[1] += sy * dt
        position[2] += sz * dt
        imu_writer.writerow([now, ax, ay, az, sx, sy, sz, *position])
    except Exception as e:
        print("❌ Errore IMU:", e)

def imu_loop():
    while not stop_flag.is_set():
        log_imu_data()
        time.sleep(0.1)  # Підвищена частота логування

# === Logging commands ===
def log_command(cmd):
    with open(command_log_filename, 'a') as f:
        f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {cmd}\n")


class Pose3D:
    def __init__(self, x=0, y=0, z=0, theta=0):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

    def __repr__(self):
        return f"Pose3D(x={self.x}, y={self.y}, z={self.z}, theta={self.theta})"

#per stima posizioni
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, initial_estimate=0.0):
        self.estimate = initial_estimate
        self.error_estimate = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement):
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)
        self.estimate += kalman_gain * (measurement - self.estimate)
        self.error_estimate = (1 - kalman_gain) * self.error_estimate + abs(self.estimate - measurement) * self.process_variance
        return self.estimate


class Follower:
    def __init__(self):
        self.pose = Pose3D()
        self.kalman_theta = SimpleKalmanFilter()
        self.detector = cv.QRCodeDetector()

        battery = tello.get_battery()
        print(f"🔋 Battery level: {battery}%")
        if battery < 20:
            raise Exception("❗ Заряд батареї занадто низький (<20%)")

        try:
            tello.streamon()
            log_command("streamon")
        except Exception as e:
            print("❌ streamon не вдався:", e)
            raise

        self.frame_read = tello.get_frame_read()
        self.took_off = False
        self.running = True
        self.start_time = time.time()
        self.last_detection_time = time.time()
        self.pid_params = {
            'x': {'P': 0.7, 'I': 0.0012, 'D': 0.4},
            'y': {'P': 0.7,  'I': 0.0012, 'D': 0.4},
            'z': {'P': 0.75, 'I': 0.0015, 'D': 0.35},
            'yaw': {'P': 0.35, 'I': 0.0, 'D': 0.15}
           # 'x':   {'P': 0.84,  'I': 0.00108, 'D': 0.48},   # 0.7*1.2, 0.0012*0.9, 0.4*1.2
           # 'y':   {'P': 0.84,  'I': 0.00108, 'D': 0.48},
           # 'z':   {'P': 0.9,   'I': 0.00135, 'D': 0.42},   # 0.75*1.2, 0.0015*0.9, 0.35*1.2
           # 'yaw': {'P': 0.3,  'I': 0.0,     'D': 0.25}    # yaw чуть увеличен (≈1.2x)
        }


        self.log_file = open("log.txt", "w") 
        self.start_log_time = time.time()
        self._load_saved_pid()
        self.init_PID()

        self.is_taking_off = False  # Flag per il controllo della fase di decollo

    def set_pid_params(self, new_params):
        self.pid_params = new_params
        self.reset_PID()

    def _load_saved_pid(self):
        try:
            with open("pid_preset.json", "r") as f:
                saved_params = json.load(f)
            self.set_pid_params(saved_params)
            print("✅ PID iniziali caricati da pid_preset.json")
        except FileNotFoundError:
            print("⚠️ Nessun file pid_preset.json trovato. Uso i PID di default.")

    def update_pose_from_qr(self):
        frame = self.frame_read.frame
        if frame is None:
            return

        image = frame.copy()
        retval, bbox = self.detector.detect(image)

        if retval and bbox is not None and len(bbox) > 0:
            qr_size = 10.0 #QR DI 10 CM

            # half_size = qr_size / 2.0
            # obj_points = np.array([
            #     [-half_size, -half_size, 0],
            #     [ half_size, -half_size, 0],
            #     [ half_size,  half_size, 0],
            #     [-half_size,  half_size, 0]
            # ], dtype=np.float32)

            obj_points = np.array([  #definisco i 4 vertici in 3D del qr code 
                [0, 0, 0], #vertice in basso a sinistra
                [qr_size, 0, 0], #vertice in alto a sinistra
                [qr_size, qr_size, 0], #vertice in alto a destra
                [0, qr_size, 0] #vertice in basso a destra
            ], dtype=np.float32)
            
            h, w = image.shape[:2]
            camera_matrix = np.array([
                [900, 0, 481.5],
                [0, 900, 357.8],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.array([-0.03657185, 0.05602408, -0.00013781, -0.00215433, 0.01426604])

            if len(bbox) == 1 and len(bbox[0]) == 4:
                img_points = np.array(bbox[0], dtype=np.float32)
                success, rvec, tvec = cv.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                             
                if success: #se la stima della posizione del qr code è corretta 

                    #aggiorna la posizione stimata con i valori convertiti in metri, MAPPANDO COORDINATE DRONE E CAMERA
                    self.pose.x = tvec[2][0] / 100.0  # Z della camera corrisponde a X del drone (avanti/dietro)
                    self.pose.y = tvec[0][0] / 100.0  # X della camera corrsiponde a Y del drone (destra/sinistra)
                    self.pose.z = tvec[1][0] / 100.0  # Y della camera corrisponde a Z del drone (up/down)
                    self.last_detection_time = time.time()

                    R, _ = cv.Rodrigues(rvec)
                    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
                    yaw_rad = math.atan2(R[1, 0], R[0, 0]) if sy >= 1e-6 else 0
                    yaw_deg = math.degrees(yaw_rad)

                    # Aggiorna la stima di theta con il filtro di Kalman
                    self.pose.theta = self.kalman_theta.update(yaw_deg)
                    
                    #print(f"📍QR rilevato -  x={self.pose.x:.2f}m, y={self.pose.y:.2f}m, z={self.pose.z:.2f}m , yaw = {self.pose.theta:.2f}°\n")

    def init_PID(self):
        def proportional():
            Vx = Vy = Vz = Va = 0
            Ix = Iy = Iz = Ia = 0
            ex_prev = ey_prev = ez_prev = ea_prev = 0
            prev_time = time.time()

            while True:
                xoff, yoff, zoff, angleoff = yield Vx, Vy, Vz, Va
                current_time = time.time()
                dt = current_time - prev_time if current_time - prev_time > 0 else 1e-6

                ex = xoff - X_DIST
                ey = yoff - Y_DIST
                ez = zoff - 0
                ea = ANGLE - angleoff

                # PID x
                Px = self.pid_params['x']['P'] * ex
                Ix += self.pid_params['x']['I'] * ex * dt
                Dx = self.pid_params['x']['D'] * (ex - ex_prev) / dt

                # PID y
                Py = self.pid_params['y']['P'] * ey
                Iy += self.pid_params['y']['I'] * ey * dt
                Dy = self.pid_params['y']['D'] * (ey - ey_prev) / dt

                # PID z
                Pz = self.pid_params['z']['P'] * ez
                Iz += self.pid_params['z']['I'] * ez * dt
                Dz = self.pid_params['z']['D'] * (ez - ez_prev) / dt

                # PID yaw con zona morta e timer
                yaw_deadzone = 5 #5 gradi di tolleranza 
                if self.is_taking_off:  # Se il drone sta decollando, non correggere troppo velocemente yaw
                    Va = 0
                else:
                    if abs(ea) < yaw_deadzone:
                        Ia = 0
                        Va = 0
                    else:
                        Pa = self.pid_params['yaw']['P'] * ea
                        Ia += self.pid_params['yaw']['I'] * ea * dt
                        Da = self.pid_params['yaw']['D'] * (ea - ea_prev) / dt
                        Va = Pa + Ia + Da

                # Somma controlli
                Vx = Px + Ix + Dx
                Vy = Py + Iy + Dy
                Vz = Pz + Iz + Dz

                # Clamp
                Vx = np.clip(Vx, -0.7, 0.7)
                Vy = np.clip(Vy, -0.7, 0.7)
                Vz = np.clip(Vz, -0.6, 0.6)
                Va = np.clip(Va, -0.3, 0.3)

                ex_prev, ey_prev, ez_prev, ea_prev = ex, ey, ez, ea
                prev_time = current_time
                yield Vx, Vy, Vz, Va

        # ⚠️ Queste due righe devono essere FUORI dalla funzione proportional()
        self.PID = proportional()
        next(self.PID)

        # Scrivi intestazione tabellare nel log
        self.log_file.write(f"{'time':>6} | {'Px':>6} | {'Py':>6} | {'Pz':>6} | {'Pa':>6} | {'Vx':>6} | {'Vy':>6} | {'Vz':>6} | {'Va':>6}\n")
            
    def control_loop(self):
        current_time = time.time()
        if current_time - self.last_detection_time > 2: #se perde il qr 
            self.reset_PID()
            tello.send_rc_control(0, 0, 0, 0)
            
            tello.land()
            self.took_off = False
            return


        # Calcolo PID
        Vx, Vy, Vz, Va = self.PID.send((self.pose.x, self.pose.y, self.pose.z, self.pose.theta))


        # Tempo attuale
        t = time.time() - self.start_log_time

               # Definizione dei colori ANSI per il terminale
        RESET = "\033[0m"
        C_EX = "\033[96m"    # Azzurro
        C_EY = "\033[92m"    # Verde
        C_EZ = "\033[95m"    # Magenta
        C_EA = "\033[93m"    # Giallo
        C_VX = "\033[94m"    # Blu
        C_VY = "\033[91m"    # Rosso
        C_VZ = "\033[90m"    # Grigio
        C_VA = "\033[97m"    # Bianco

        # Stampa semplice risultati sul terminale
        #print(
        #    f"🧭 x=\033[96m{self.pose.x:+5.2f}\033[0m, Vx=\033[94m{Vx:+5.2f}\033[0m | "
        #    f"y=\033[92m{self.pose.y:+5.2f}\033[0m, Vy=\033[91m{Vy:+5.2f}\033[0m | "
        #    f"z=\033[95m{self.pose.z:+5.2f}\033[0m, Vz=\033[90m{Vz:+5.2f}\033[0m | "
        #    f"a=\033[93m{self.pose.theta:+6.2f}°\033[0m, Va=\033[97m{Va:+5.2f}\033[0m"
        #)
     
        # Stampa risultati su un file log.txt
        self.log_file.write(
            f"{t:6.2f} | {self.pose.x:+6.2f} | {self.pose.y:+6.2f} | {self.pose.z:+6.2f} | {self.pose.theta:+7.2f} | {Vx:+6.2f} | {Vy:+6.2f} | {Vz:+6.2f} | {Va:+6.2f}\n"
        )
        self.log_file.flush()


        # Blocco dinamico se troppo vicino
        if self.pose.x < 0.6 and Vx > 0:
            Vx = 0

        # Deadzone per evitare piccoli tremolii
        if abs(Vx) < 0.05: Vx = 0
        if abs(Vy) < 0.05: Vy = 0
        if abs(Vz) < 0.05: Vz = 0

        # Azzeramento traslazioni se Va significativo, altrimenti azzera Va (anti-jitter)
        # if abs(Va) > 0.05:
        #     Vx = 0
        #     Vy = 0
        #     Vz = 0
    

        tello.send_rc_control(int(Vy * 100), int(Vx * 100), int(-Vz * 100), 0)
        log_command(f"send_rc_control({int(Vy * 100)}, {int(Vx * 100)}, {int(-Vz * 100)}, 0)")
        time.sleep(0.03)  # Per stabilizzare il ciclo di controllo (~30 Hz)

    def reset_PID(self):
        self.init_PID()

    def takeoff(self):
        tello.takeoff()
        log_command("takeoff")
        imu_thread = threading.Thread(target=imu_loop)
        imu_thread.start()
        self.took_off = True
        self.is_taking_off = True  # Imposta il flag per la fase di decollo
        # Dopo un breve tempo, cambia il flag per permettere il controllo yaw
        time.sleep(1)  # Dà al drone il tempo di stabilizzarsi
        self.is_taking_off = False  # Setta a False dopo che il drone ha decollato

    def land(self):
        tello.land()
        log_command("land")
        self.took_off = False
        self.is_taking_off = False  # Reset flag dopo l'atterraggio

    def emergency(self):
        tello.emergency()
        log_command("emergency")
        self.took_off = False
        self.running = False
        if hasattr(self, 'log_file'):
            self.log_file.close()


# === Auto Mission ===
def auto_mission():
    log_command("takeoff")
    tello.takeoff()
    time.sleep(5)


# === Cleanup ===
def close_everything():
    try:
        if follower.took_off:
            print("🛬 Посадка перед завершенням...")
            tello.land()
            log_command("land")
            time.sleep(3)  # Дати час на посадку

        if not imu_file.closed:
            imu_file.close()

        # Спробувати вимкнути відеопотік лише якщо він активний
        try:
            tello.streamoff()
            log_command("streamoff")
        except Exception as e:
            print(f"⚠️ streamoff не вдалось: {e}")

        tello.end()
        print("🟢 Chiusura completata.")
    except Exception as e:
        print("❌ Errore in chiusura:", e)

atexit.register(close_everything)

# === MAIN ===
if __name__ == '__main__':
    follower = Follower()

    def on_takeoff(e):
        if not follower.took_off:
            print("🚀 Дрон злітає")
            follower.takeoff()

    def on_emergency(e):
        print("🛑 АВАРІЙНА ЗУПИНКА!")
        follower.emergency()
        stop_flag.set()

    # Прив’язка клавіш
    keyboard.on_press_key('t', on_takeoff)     # натисни 't' щоб злетіти
    keyboard.on_press_key('e', on_emergency)   # натисни 'e' для зупинки

    def main_loop():
        try:
            while follower.running and not stop_flag.is_set():
                if follower.took_off:
                    
                    follower.update_pose_from_qr()
                    follower.control_loop()
                else:
                    follower.update_pose_from_qr()
                time.sleep(0.03)
        except Exception as ex:
            print("❌ Помилка в головному циклі:", ex)
        finally:
            try:
                follower.log_file.close()
                imu_file.close()
                tello.end()
            except Exception as e:
                print("⚠️ Завершення з помилкою:", e)
            cv.destroyAllWindows()
            print("🟢 Завершення роботи")

    thread = threading.Thread(target=main_loop, daemon=True)
    thread.start()

    print("Натисни 't' для злету, 'e' для аварійної зупинки.")
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("❌ Перервано з клавіатури, посадка...")
        if follower.took_off:
            follower.land()
        stop_flag.set()
        thread.join()
        close_everything()
