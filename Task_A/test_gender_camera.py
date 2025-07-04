import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


model = load_model('best_gender_model.h5')
class_labels = ['Female', 'Male']  # FEMALE = 0, MALE = 1


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture an image or 'q' to quit.")
captured_img = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured.")
        break

    cv2.imshow("Camera - Press 'c' to capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        captured_img = frame.copy()
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if captured_img is None:
    print("No image captured.")
    exit()


gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) == 0:
    print("No face detected.")
    exit()


x, y, w, h = faces[0]
face_img = captured_img[y:y+h, x:x+w]


img_resized = cv2.resize(face_img, (224, 224))
img_array = img_to_array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)


prediction = model.predict(img_array)
predicted_class = int(prediction[0][0] > 0.5)
confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
gender = class_labels[predicted_class]


print(f"Predicted Gender: {gender} ({confidence * 100:.2f}% confidence)")


cv2.rectangle(captured_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(captured_img, f"{gender} ({confidence * 100:.1f}%)", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Prediction", captured_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
