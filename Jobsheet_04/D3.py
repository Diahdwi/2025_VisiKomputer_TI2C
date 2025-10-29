import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Landmark indeks mata kiri untuk EAR
# Vertikal: (159,145), Horizontal: (33,133)
L_TOP, L_BOTTOM = 159, 145
L_LEFT, L_RIGHT = 33, 133

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi FaceMeshDetector
detector = FaceMeshDetector(
    staticMode=False,
    maxFaces=2,
    minDetectionCon=0.5,
    minTrackCon=0.5
)

# Variabel kedipan
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3   # Minimal frame berturut-turut dianggap kedip
EYE_AR_THRESHOLD = 0.20       # Ambang EAR untuk menilai mata tertutup
is_closed = False

while True:
    ok, img = cap.read()
    if not ok:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]  # 468 titik wajah

        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])
        ear = v / (h + 1e-8)

        cv2.putText(img, f"EAR (Left): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Logika deteksi kedip
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True
        else:
            closed_frames = 0
            is_closed = False

        # Tampilkan jumlah kedip
        cv2.putText(img, f"Blink: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("FaceMesh + EAR Blink Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
