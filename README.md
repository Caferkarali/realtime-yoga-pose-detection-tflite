# 🧘 Real-Time Yoga Pose Detection with MediaPipe & TFLite

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/Library-MediaPipe-orange)
![TensorFlow Lite](https://img.shields.io/badge/Model-TFLite-yellow)
![Status](https://img.shields.io/badge/Status-Prototype-green)

## 📌 Project Overview
This project is a computer vision application that detects and classifies yoga poses in real-time using a webcam. It leverages **Google MediaPipe** for robust pose estimation and a lightweight **TensorFlow Lite** model for classification.

The system is designed to be efficient and privacy-preserving, as it processes video frames locally on the CPU.

## 🚀 Key Features
* **Real-Time Detection:** Processes video streams instantly using OpenCV.
* **Pose Estimation:** Extracts 33 3D landmarks of the human body using MediaPipe Pose.
* **Visual Preprocessing:** Converts numerical landmark coordinates into a synthetic "skeleton image" (224x224) to feed into a CNN-based TFLite classifier.
* **Lightweight Model:** Uses `.tflite` format for fast inference on edge devices or standard laptops.

## 🧘 Supported Poses
The model is trained to recognize the following 5 specific yoga poses:
1.  **Warrior II** (Virabhadrasana II)
2.  **Tree Pose** (Vrikshasana)
3.  **Plank** (Phalakasana)
4.  **Downward Dog** (Adho Mukha Svanasana)
5.  **Goddess Pose** (Utkata Konasana)

## 🛠 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/realtime-yoga-pose-detection-tflite.git
    cd realtime-yoga-pose-detection-tflite
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    Ensure your webcam is connected and run:
    ```bash
    python main.py
    ```
    *Press 'q' to exit the application.*

## 🧠 How It Works (Technical Detail)
1.  **Frame Capture:** OpenCV captures the video frame.
2.  **Landmark Extraction:** MediaPipe Pose predicts the body landmarks (shoulders, elbows, knees, etc.).
3.  **Feature Transformation:** instead of feeding raw (x,y) coordinates directly, the code generates a blank black image (224x224) and draws the keypoints as white dots. This spatial representation allows the CNN model to learn geometric patterns.
4.  **Inference:** The processed image is passed to the TFLite interpreter.
5.  **Visualization:** The predicted class and confidence score are overlaid on the original video feed.

## 📸 Demo
*(Add a screenshot of yourself or a test subject doing a yoga pose here)*

## 🤝 Contribution
Contributions are welcome! If you want to add more poses or improve the model accuracy, feel free to submit a Pull Request.

## 📝 License
This project is open-source and available under the MIT License.
