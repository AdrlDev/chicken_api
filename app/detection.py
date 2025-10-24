#detection.py

import cv2
from app.utils import yolo

stop_live = False

def _run_detection():
        global stop_live
        stop_live = False
        print("🎥 Starting live chicken detection...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera.")
            return

        while not stop_live:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, stream=True)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = yolo.names[cls]
                    color = (0, 255, 0) if label.lower() == "healthy" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("ChickenAI Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Stopping live detection...")
                break

        cap.release()
        cv2.destroyAllWindows()
        stop_live = True
        print("✅ Live detection ended.")