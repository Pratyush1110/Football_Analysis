# AI-Powered Football Analysis System ⚽🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/Library-OpenCV-orange)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io/)

A comprehensive Computer Vision and Machine Learning pipeline that extracts tactical insights from raw football match footage, now with an **interactive Streamlit dashboard** for video upload, live analytics, and downloadable reports.

---

## 📌 Project Overview

The system automates:
- **Object Detection** – Multi-class entity tracking (players, referees, ball) via YOLOv8 + ByteTrack
- **Team Identification** – Color-based K-Means clustering for automatic team assignment
- **Spatial Mapping** – Perspective transformation to convert pixels → real-world meter coordinates
- **Motion Correction** – Optical flow to decouple camera movement from player movement
- **Analytics** – Per-player speed, distance, possession; per-team aggregates; match summary

---

## 🖥️ Streamlit Dashboard

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Pratyush1110/Football_Analysis.git
cd Football_Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your model weights
#    Place best.pt inside the models/ directory

# 4. Launch the dashboard
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Dashboard Features

| Feature | Description |
|---|---|
| 📁 Video Upload | Drag-and-drop MP4 / AVI / MOV / MKV |
| 🚀 One-click Analysis | Full pipeline runs with a progress bar |
| 🎬 Annotated Video | In-browser playback of the processed output |
| 📊 Live Analytics | Match summary cards, team table, per-player table |
| ⬇️ Downloads | JSON stats · CSV stats · PDF report · annotated video |

---

## 📊 Analytics Output

### Match Summary
| Metric | Description |
|---|---|
| Duration | Total clip length (MM:SS) |
| Players Detected | Unique tracked player IDs |
| Team 1 / 2 Possession | Ball possession split (%) |
| Overall Avg Speed | Across all players (km/h) |

### Per-Player Stats
| Field | Description |
|---|---|
| Player ID | Unique ByteTrack ID |
| Team | 1 or 2 |
| Total Distance (m) | Real-world meters covered |
| Avg / Max Speed | km/h |
| Possession Frames | Frames where player held the ball |
| Tracked Frames | Total frames the player appeared in |

### Per-Team Stats
Aggregated distance, average speed, possession percentage, and player count.

---

## 🛠️ Technical Architecture

```
app.py                          ← Streamlit dashboard entry point
analytics_collector.py          ← Harvests stats from processed tracks
report_generator.py             ← JSON / CSV / PDF export layer
│
├── trackers/                   ← YOLOv8 + ByteTrack
├── team_assigner/              ← K-Means jersey color clustering
├── player_ball_assigner/       ← Ball-to-player assignment
├── camera_movement_estimator/  ← Optical flow compensation
├── view_transformer/           ← Homography / perspective transform
├── speed_and_distance_estimator/  ← Real-world speed & distance
└── utils/                      ← Video I/O, bbox helpers
```

---

## 🚀 Installation & Usage (CLI)

```bash
git clone https://github.com/Pratyush1110/Football_Analysis.git
cd Football_Analysis
pip install -r requirements.txt

# Place model weights
cp /path/to/best.pt models/best.pt

# Place input video
cp /path/to/match.mp4 input_videos/

# Run CLI pipeline (no UI)
python main.py

# Run Streamlit dashboard
streamlit run app.py
```

---

## 💻 Tech Stack

| Category | Library |
|---|---|
| Core Language | Python 3.8+ |
| Vision | OpenCV, Ultralytics YOLOv8, Supervision |
| Data | NumPy, Pandas, Scikit-learn |
| Dashboard | Streamlit |
| PDF Reports | ReportLab |
| Training | Google Colab (GPU), Jupyter Notebooks |

---

## 📁 Project Structure

```
Football_Analysis/
├── app.py                          # 🆕 Streamlit dashboard
├── analytics_collector.py          # 🆕 Stats collection layer
├── report_generator.py             # 🆕 PDF / JSON / CSV export
├── main.py                         # CLI pipeline entry point
├── requirements.txt                # Updated with new deps
├── models/
│   └── best.pt                     # YOLOv8 weights (add manually)
├── input_videos/                   # Drop your .mp4 here
├── output_videos/                  # Processed videos saved here
├── stubs/                          # Pickle stubs for dev speed
├── trackers/
├── team_assigner/
├── player_ball_assigner/
├── camera_movement_estimator/
├── view_transformer/
├── speed_and_distance_estimator/
├── utils/
└── training/
    └── football_training_yolo_v5.ipynb
```
