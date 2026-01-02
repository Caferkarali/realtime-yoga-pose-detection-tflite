import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="yoga_pose_model.tflite")
interpreter.allocate_tensors()

# Modelin giriş ve çıkış detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Modelin beklediği giriş boyutunu yazdır
input_shape = input_details[0]['shape']
print("Model input shape:", input_shape)

# Sınıf isimlerini tanımla
class_names = ["warrior2", "tree", "plank", "downdog", "goddess"]

# MediaPipe Pose ayarları
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamerayı başlat
cap = cv2.VideoCapture(0)
print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # BGR görüntüyü RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Anahtar noktaları çiz
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Anahtar noktaları vektöre dönüştür
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])

        # Anahtar noktalarından bir görüntü oluşturun (224x224 boyutunda boş bir siyah resim)
        keypoints_image = np.zeros((224, 224, 3), dtype=np.float32)

        # Anahtar noktalarını (x, y) koordinatları ile resme işaretler ekleyin
        for i in range(0, len(keypoints), 4):
            x = int(keypoints[i] * 224)  # x koordinatını normalize et
            y = int(keypoints[i+1] * 224)  # y koordinatını normalize et
            # Resme beyaz noktalar yerleştir
            cv2.circle(keypoints_image, (x, y), 3, (1, 1, 1), -1)

        # Görüntüyü modele uygun şekilde işleyin
        input_data = np.expand_dims(keypoints_image, axis=0)  # (1, 224, 224, 3)

        # Modelden tahmin al
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # En yüksek olasılıklı sınıfı belirle
        max_index = np.argmax(predictions)
        predicted_class = class_names[max_index]
        confidence = predictions[max_index] * 100

        # Sonuçları görüntüde göster
        text = f"{predicted_class}: {confidence:.2f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yoga Pose Detection", frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
