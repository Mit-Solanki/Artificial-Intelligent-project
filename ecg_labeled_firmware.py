import serial
import time
import csv
from serial.tools import list_ports
import os
from datetime import datetime
import threading
import matplotlib.pyplot as plt
from collections import deque

# =========================
# CONFIGURATION
# =========================
BAUD = 115200
SESSION_DURATION = 300   # 5 minutes

DATA_PATHS = {
    "stress": r"D:\AuraPulse\data\Stress",
    "relaxed": r"D:\AuraPulse\data\Relaxed",
    "normal": r"D:\AuraPulse\data\Normal"
}

# Create folders
for path in DATA_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Globals
current_file = None
writer = None
session_start_time = None
current_label = None
current_folder = None

recording = False
enter_press_count = 0
exit_program = False

# =========================
# FIND ARDUINO
# =========================
def find_arduino_port():
    ports = list_ports.comports()
    for p in ports:
        desc = p.description.lower()
        if "arduino" in desc or "usb serial" in desc or "ch340" in desc:
            print(f"✅ Arduino detected on: {p.device}")
            return p.device
    return ports[0].device if ports else None

# =========================
# CREATE FILE
# =========================
def create_new_file():
    global current_file, writer, session_start_time

    if current_file:
        current_file.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ecg_{current_label}_{timestamp}.csv"
    filepath = os.path.join(current_folder, filename)

    current_file = open(filepath, 'w', newline='')
    writer = csv.writer(current_file)
    writer.writerow(['timestamp', 'raw_ecg', 'label'])

    session_start_time = time.time()
    print(f"\n📁 New file: {filepath}")

# =========================
# LABEL INPUT
# =========================
def get_valid_label():
    while True:
        label = input("Enter label (stress / relaxed / normal): ").strip().lower()
        if label in DATA_PATHS:
            return label
        else:
            print("❌ Invalid label!")

# =========================
# ENTER KEY CONTROL
# =========================
def toggle_recording():
    global recording, enter_press_count
    global exit_program, current_label, current_folder

    while True:
        input()
        enter_press_count += 1

        if enter_press_count == 1:
            current_label = get_valid_label()
            current_folder = DATA_PATHS[current_label]

            recording = True
            print(f"\n▶ Recording STARTED [{current_label.upper()}]")
            create_new_file()

        elif enter_press_count == 2:
            recording = False
            print("\n⏹ Recording STOPPED. Press ENTER again to exit.")

            if current_file:
                current_file.close()

        elif enter_press_count >= 3:
            print("\n🛑 Exiting...")
            exit_program = True
            break

threading.Thread(target=toggle_recording, daemon=True).start()

# =========================
# SERIAL SETUP
# =========================
port = "COM23"
if port is None:
    print("❌ No Arduino found.")
    exit()

ser = serial.Serial(port, BAUD)
time.sleep(2)
ser.reset_input_buffer()

print("\n💡 Controls:")
print("1st ENTER → Start recording")
print("2nd ENTER → Stop recording")
print("3rd ENTER → Exit")

# =========================
# LIVE GRAPH SETUP
# =========================
data_buffer = deque(maxlen=500)

plt.ion()
fig, ax = plt.subplots()
line_plot, = ax.plot([], [])
ax.set_ylim(0, 1023)
ax.set_title("Live ECG Signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# =========================
# MAIN LOOP
# =========================
try:
    while not exit_program:

        if ser.in_waiting == 0:
            time.sleep(0.001)
            continue

        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if "Raw_ECG:" in line:
            try:
                raw_val = int(line.split(',')[0].split(':')[1])
                t = time.time()

                # 🔴 LIVE TERMINAL OUTPUT
                print(f"\rECG: {raw_val}", end="")

                # 📈 UPDATE GRAPH
                data_buffer.append(raw_val)
                line_plot.set_ydata(data_buffer)
                line_plot.set_xdata(range(len(data_buffer)))

                ax.relim()
                ax.autoscale_view()

                plt.draw()
                plt.pause(0.001)

                # 💾 SAVE DATA
                if recording:

                    if time.time() - session_start_time >= SESSION_DURATION:
                        print("\n⏱ 5 min done → New file")
                        create_new_file()

                    if writer:
                        writer.writerow([t, raw_val, current_label])

            except:
                continue

except KeyboardInterrupt:
    print("\n🛑 Interrupted")

finally:
    print("\nCleaning up...")
    if current_file:
        current_file.close()
    ser.close()
    print("Done.")