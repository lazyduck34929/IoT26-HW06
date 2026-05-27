from datetime import datetime
from pathlib import Path
import time

from picamera2 import Picamera2
from ultralytics import YOLO


# Create a folder for saved car detection images
output_dir = Path("car_results")
output_dir.mkdir(exist_ok=True)

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load the YOLO model
model = YOLO("yolo26n.pt")

last_saved_time = 0

print("Car detection started. Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()

        # Class 2 in the COCO dataset is "car"
        result = model(frame, classes=[2], verbose=False)[0]

        current_time = datetime.now()
        has_car = result.boxes is not None and len(result.boxes) > 0

        if has_car:
            car_count = len(result.boxes)
            print(f"[{current_time:%H:%M:%S}] Detected car: {car_count}", flush=True)

            # Save one annotated image every 3 seconds
            if time.time() - last_saved_time >= 3:
                filename = output_dir / f"car_{current_time:%Y%m%d_%H%M%S}.jpg"
                result.save(filename=str(filename))
                print(f"Saved image: {filename}", flush=True)
                last_saved_time = time.time()
        else:
            print(f"[{current_time:%H:%M:%S}] No car detected", flush=True)

except KeyboardInterrupt:
    print("\nCar detection stopped.")

finally:
    picam2.stop()
