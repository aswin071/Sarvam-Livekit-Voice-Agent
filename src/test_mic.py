import sounddevice as sd
import numpy as np

# Try WASAPI device (device 9) - more reliable on Windows
DEVICE_ID = 9  # Microphone Array (Intel Smart) WASAPI

print(f"Testing device {DEVICE_ID} - speak for 10 seconds...")

def callback(indata, frames, time, status):
    if status:
        print("Status:", status)
    volume = np.linalg.norm(indata) * 10
    bars = "|" * int(volume)
    print(f"{bars}  level={volume:.2f}")

with sd.InputStream(device=DEVICE_ID, callback=callback, channels=2, samplerate=16000):
    sd.sleep(10000)
