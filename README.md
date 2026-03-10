# AI-Powered Football Analysis System ⚽️🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-orange)](https://opencv.org/)

A comprehensive Computer Vision and Machine Learning pipeline designed to extract tactical insights from raw football match footage. This system automates player tracking, team identification, and spatial analytics to calculate real-world metrics like ball possession, player speed, and total distance covered.



---

## 📌 Project Overview
The core challenge in sports analytics is converting 2D pixels into 3D spatial data. This project solves that by integrating multiple CV techniques:
* **Object Detection:** Multi-class entity tracking (Players, Referees, Ball).
* **Team Identification:** Color-based clustering for automated team assignment.
* **Spatial Mapping:** Perspective transformation to map pixels to field coordinates.
* **Motion Correction:** Optical flow to decouple camera movement from player movement.

---

## 🛠️ Technical Architecture & Features

### 1. Multi-Entity Tracking (YOLOv8 & ByteTrack)
* Utilized **YOLOv8** for high-performance object detection, fine-tuned to distinguish between players, referees, and the ball in varying lighting conditions.
* Integrated **ByteTrack** and **Supervision** to maintain unique IDs for players even during high-occlusion events (e.g., player huddles or tackles).
* **Ball Interpolation:** Implemented a **Pandas-based linear interpolation** logic to estimate ball position in frames where it is occluded by players or moving at high velocity.



### 2. Team Assignment (K-Means Clustering)
* Automated jersey color segmentation using **K-Means Clustering**.
* The system crops player bounding boxes, segments the foreground (jersey) from the background (grass), and clusters RGB centroids to assign team IDs without manual labeling.

### 3. Perspective Transformation (Homography)
Standard broadcast angles distort the field view. I implemented a **Perspective Transform** to map the trapezoidal image view into a birds-eye-view rectangle.
* **Mathematical Mapping:** Pixel coordinates are converted into real-world meter metrics.
* **Result:** Provides accurate data for speed (km/h) and distance regardless of camera angle.



### 4. Camera Motion Compensation
* Calculated camera panning and zooming using **Optical Flow** features.
* By isolating global camera movement, the system ensures that "Distance Covered" metrics are based solely on the player's physical exertion on the pitch, not the camera's movement.

---

## 📊 Analytics Output
* **Team Possession:** Real-time calculation of ball acquisition percentage.
* **Player Performance:** Per-player speed tracking (km/h) and total distance (meters) covered during the clip.
* **Tactical Overlays:** Automated annotations including team circles, ball pointers, and speed labels.

---

## 💻 Tech Stack
* **Core:** Python 3.x
* **Vision:** OpenCV, Ultralytics (YOLOv8), Supervision
* **Data:** NumPy, Pandas, Scikit-learn (K-Means)
* **Development:** Google Colab (GPU training), Jupyter Notebooks

---

## 🚀 Installation & Usage

1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Pratyush1110/Football_Analysis.git](https://github.com/Pratyush1110/Football_Analysis.git)
   cd Football_Analysis
