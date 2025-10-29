import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import math

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi Pose Detector
detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5, trackCon=0.5)

# Fungsi hitung sudut lutut dengan Dot Product
def calculate_angle(a, b, c):
    a = np.array(a)  # Hip
    b = np.array(b)  # Knee
    c = np.array(c)  # Ankle

    BA = a - b      # Vector BA
    BC = c - b      # Vector BC

    dot = np.dot(BA, BC)
    mag = np.linalg.norm(BA) * np.linalg.norm(BC)

    if mag == 0:
        return 0

    angle = math.degrees(math.acos(dot / mag))
    return int(angle)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=True, bboxWithHands=False)

    if lmList:
        # Landmark untuk lutut kiri
        hip = lmList[23][0:2]     # Pinggul kiri
        knee = lmList[25][0:2]    # Lutut kiri
        ankle = lmList[27][0:2]   # Pergelangan kaki kiri

        # Hitung sudut lutut
        knee_angle = calculate_angle(hip, knee, ankle)

        # Tentukan status
        status = "Berdiri" if knee_angle > 160 else "Jongkok"

        # Tampilkan teks ke layar
        cv2.putText(img, f"Sudut Lutut: {knee_angle} deg", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.putText(img, f"Status: {status}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,0) if status=="Berdiri" else (0,0,255), 3)

        # Gambar titik landmark
        for pt in [hip, knee, ankle]:
            cv2.circle(img, pt, 8, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Deteksi Berdiri / Jongkok", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
