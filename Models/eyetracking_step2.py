import cv2
import mediapipe as mp
import numpy as np
import os
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# ====== CONFIG ======
IMAGE_FOLDER = r"C:/Users/mamil/OneDrive/Documents/autism_detection_project/Data/Saliency4ASD Dataset/Saliency4ASD/dataset/Images"
DISPLAY_TIME = 5
SAVE_FILE = "C:/Users/mamil/OneDrive/Documents/autism_detection_project/Models/user_gaze_data.csv"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# ====== Mediapipe Setup ======
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# =========================================================
# CALIBRATION FUNCTION
# =========================================================
def run_calibration(cap, face_mesh):

    x_positions = [int(SCREEN_WIDTH * 0.1),
                   int(SCREEN_WIDTH * 0.5),
                   int(SCREEN_WIDTH * 0.9)]

    y_positions = [int(SCREEN_HEIGHT * 0.1),
                   int(SCREEN_HEIGHT * 0.5),
                   int(SCREEN_HEIGHT * 0.9)]

    calibration_points = [(x, y) for y in y_positions for x in x_positions]

    print("Starting Improved 9-Point Calibration...")

    calibration_data = []

    for point in calibration_points:

        h_samples = []
        v_samples = []

        start_time = time.time()

        while time.time() - start_time < 2:

            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            radius = int(25 + 8 * np.sin(time.time() * 4))
            cv2.circle(screen, point, radius, (0, 255, 255), -1)

            if results.multi_face_landmarks:

                landmarks = results.multi_face_landmarks[0].landmark
                frame_h, frame_w, _ = frame.shape

                left_corner = landmarks[33]
                right_corner = landmarks[133]
                iris = landmarks[468]
                top_lid = landmarks[159]
                bottom_lid = landmarks[145]

                lx = left_corner.x * frame_w
                rx = right_corner.x * frame_w
                ix = iris.x * frame_w

                if abs(rx - lx) <= 1:
                    continue

                h_ratio = (ix - lx) / (rx - lx)

                eye_height = abs(top_lid.y - bottom_lid.y)

                # Blink filter
                if eye_height < 0.005:
                    continue

                eye_center_y = (top_lid.y + bottom_lid.y) / 2
                v_ratio = (iris.y - eye_center_y) / eye_height
                # Blink filter
                if eye_height < 0.005:
                    continue
               

                if 0 <= h_ratio <= 1 and -1 <= v_ratio <= 1:
                    h_samples.append(h_ratio)
                    v_samples.append(v_ratio)

            cv2.imshow("Calibration", screen)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if len(h_samples) > 10:
            h_median = np.median(h_samples)
            v_median = np.median(v_samples)
            calibration_data.append([h_median, v_median, point[0], point[1]])

    cv2.destroyWindow("Calibration")

    calibration_data = np.array(calibration_data)

    if len(calibration_data) < 6:
        print("Warning: Few calibration points collected.")

    X = calibration_data[:, :2]
    y_x = calibration_data[:, 2]
    y_y = calibration_data[:, 3]

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model_x = LinearRegression().fit(X_poly, y_x)
    model_y = LinearRegression().fit(X_poly, y_y)

    # ===== Calibration Quality Check =====
    errors = []

    for i in range(len(X)):
        sample = X[i].reshape(1, -1)
        sample_poly = poly.transform(sample)

        pred_x = model_x.predict(sample_poly)[0]
        pred_y = model_y.predict(sample_poly)[0]

        true_x = y_x[i]
        true_y = y_y[i]

        error = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        errors.append(error)

    mean_error = np.mean(errors)
    print(f"Mean Calibration Error: {mean_error:.2f} pixels")

    if mean_error > 120:
        print("⚠ Warning: Calibration accuracy is low but continuing session.")
    else:
        print("Calibration quality acceptable.")

    return model_x, model_y, poly
cap = cv2.VideoCapture(0)

model_x, model_y,poly = run_calibration(cap, face_mesh)

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png")]

all_gaze_data = []

print("Starting Stimulus Presentation...")

prev_x, prev_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
alpha = 0.3

for img_name in image_files[:5]:

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    stimulus = cv2.imread(img_path)

    if stimulus is None:
        continue

    stimulus = cv2.resize(stimulus, (SCREEN_WIDTH, SCREEN_HEIGHT))

    start_time = time.time()

    while time.time() - start_time < DISPLAY_TIME:

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        gaze_x, gaze_y = None, None

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark
            frame_h, frame_w, _ = frame.shape

            left_corner = landmarks[33]
            right_corner = landmarks[133]
            iris = landmarks[468]
            top_lid = landmarks[159]
            bottom_lid = landmarks[145]

            # ----- Horizontal Ratio -----
            lx = left_corner.x * frame_w
            rx = right_corner.x * frame_w
            ix = iris.x * frame_w

            if abs(rx - lx) > 1:
                horizontal_ratio = (ix - lx) / (rx - lx)
            else:
                horizontal_ratio = 0.5

            # ----- Amplified Vertical Ratio -----
            eye_center_y = (top_lid.y + bottom_lid.y) / 2
            # vertical_ratio = (iris.y - eye_center_y) * 10 + 0.5
            eye_height = abs(top_lid.y - bottom_lid.y)
            vertical_ratio = (iris.y - eye_center_y) / eye_height
            if eye_height < 0.005:
                cv2.imshow("Stimulus", stimulus)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # ----- Prediction -----
            features = poly.transform([[horizontal_ratio, vertical_ratio]])
            pred_x = model_x.predict(features)[0]
            pred_y = model_y.predict(features)[0]

            pred_x = np.clip(pred_x, 0, SCREEN_WIDTH)
            pred_y = np.clip(pred_y, 0, SCREEN_HEIGHT)

            # ----- Smoothing -----
            gaze_x = alpha * pred_x + (1 - alpha) * prev_x
            gaze_y = alpha * pred_y + (1 - alpha) * prev_y

            prev_x, prev_y = gaze_x, gaze_y
            norm_x = gaze_x / SCREEN_WIDTH
            norm_y = gaze_y / SCREEN_HEIGHT
            pixel_x = int(gaze_x)
            pixel_y = int(gaze_y)

            cv2.circle(stimulus, (pixel_x, pixel_y), 6, (0, 0, 255), -1)      
        if results.multi_face_landmarks and gaze_x is not None:

            timestamp = time.time()

            all_gaze_data.append({
                "image": img_name,
                "timestamp": timestamp,
                "gaze_x": norm_x,
                "gaze_y": norm_y
            })

        cv2.imshow("Stimulus", stimulus)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print(f"Finished viewing: {img_name}")

cap.release()
cv2.destroyAllWindows()

# ====== Save CSV ======
df = pd.DataFrame(all_gaze_data)
print("Total rows collected:", len(all_gaze_data))
df.to_csv(SAVE_FILE, index=False)
print("Saved CSV at:", os.path.abspath(SAVE_FILE))
print("Session Complete.")
print("X range:", df["gaze_x"].min(), df["gaze_x"].max())
print("Y range:", df["gaze_y"].min(), df["gaze_y"].max())