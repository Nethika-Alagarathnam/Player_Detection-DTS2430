<img width="1213" height="606" alt="image" src="https://github.com/user-attachments/assets/236f6ef6-dc78-4307-8867-355cabadbb7d" /># DS5216 - Artificial Intelligence | Programming Assignment 02
## Football Player Detection, Tracking & Pose Estimation using YOLOv8

---

## Overview of my Project

This project builds a computer vision pipeline to detect and track football players in video clips, and estimate their body keypoints. Two YOLOv8 model variants (nano and small) are fine-tuned on a custom-labeled football dataset and compared across detection performance metrics and inference speed.

The pipeline has four notebooks

| Notebook | Purpose |
|---|---|
| 01_extract_frames.ipynb | Extract frames from raw video for labeling |
| 02_train_yolov8.ipynb | Auto-label, split dataset, and train YOLOv8n |
| 03_keypoint_pose.ipynb | Run YOLOv8n-pose for skeleton/keypoint estimation |
| 04_model_comparison.ipynb | Train YOLOv8s and compare both models |

---

## Dataset

**12 football video clips** were collected (Football 01.mp4 through Football 12.mp4), each range 10 seconds in length. Videos were sourced from YouTube.

- **Google Drive link:** *(https://drive.google.com/drive/folders/1PCjux_xYi3og3Vlcb5sOzWfYW6FoL95x?usp=sharing)*
- The Drive folder contains below lists
  - 01 - Dataset - raw video clips
  - 02 - Sports_player_detection_Yolov8n - nano model training outputs
  - 03 - Sports_player_detection_Yolov8s - small model training outputs
  - 04 - Comparison Details - comparison charts and CSV
  - 05 - Output_keypoints - annotated pose estimation videos and screenshots
  - 06 - Player Detection [Bounding Box].avi - final detection output video (43.6 MB)

Frame extraction was done with OpenCV, sampling **every 10th frame** from each video using "cv2.VideoCapture". Frames were saved as ".jpg" files named with the sport label and clip name (football_Football_01_frame0010.jpg).

Labeling was done using **LabelImg** with YOLO format. To bootstrap labels, YOLOv8n was run in prediction mode (conf=0.25) on the extracted frames, and only class 0 (person) detections were retained as initial annotations.

---

## Environment

| Component | Version / Detail |
|---|---|
| OS | Windows |
| Python | 3.10 |
| Device | CPU only |
| Framework | PyTorch via Ultralytics |
| Key libraries | ultralytics, opencv-python, numpy, pandas, matplotlib, seaborn, labelImg |



## Notebook 1 - Frame Extraction (01_extract_frames.ipynb)

Reads each video from the "Football/" folder and extracts one frame every 10 frames using OpenCV. Saves frames to "dataset/images/" with structured filenames. After extraction, images are labeled using LabelImg in YOLO format (bounding box coordinates saved as ".txt" files in "dataset/labels/").

---

## Notebook 2 - YOLOv8n Training (02_train_yolov8.ipynb)

**Dataset split**

| Split | Ratio |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

**Training configuration (YOLOv8n)**

| Parameter | Value |
|---|---|
| Base model | yolov8n.pt |
| Epochs | 50 |
| Image size | 416 |
| Batch size | 4 |
| Device | CPU |
| Early stopping patience | 15 epochs |
| Class | player (1 class) |

A data.yaml file is generated automatically with paths to train/val/test splits.

**Outputs generated automatically by Ultralytics**
- results.png - loss curves, precision, recall, mAP over epochs
- BoxPR_curve.png - Precision-Recall curve
- confusion_matrix.png - confusion matrix
- weights/best.pt - best checkpoint
- weights/last.pt - final epoch checkpoint

Test set evaluation is run at the end using model.val(split = "test"), reporting mAP@0.50, mAP@0.50:0.95, Precision, and Recall. Results are saved to "test_results.csv".

---

## Notebook 3 - Pose / Keypoint Estimation (03_keypoint_pose.ipynb)

Uses the pre-trained **yolov8n-pose.pt** model to detect 17 body keypoints per person across all 12 football videos.

**COCO 17-keypoint skeleton**

```
Nose, Left Eye, Right Eye, Left Ear, Right Ear,
Left Shoulder, Right Shoulder,
Left Elbow, Right Elbow,
Left Wrist, Right Wrist,
Left Hip, Right Hip,
Left Knee, Right Knee,
Left Ankle, Right Ankle
```

**Inference settings**

| Parameter | Value |
|---|---|
| Confidence threshold | 0.30 |
| Keypoint visibility threshold | 0.40 |
| Frame skip | Every 2nd frame (for speed) |

**What the notebook produces**
- Annotated output video per clip saved to "output_keypoints/" with bounding boxes, player labels (P1, P2...), and colour-coded skeleton overlays
- Screenshot saved at frame 20 of each clip to "screenshots/"
- "keypoint_summary.csv" - per-clip summary of avg persons detected, max persons, avg keypoints visible, total frames
- "keypoint_results.png" - 3-panel chart: avg keypoints per sport, keypoint visibility over time, persons-per-frame distribution
- "per_keypoint_confidence.png" - horizontal bar chart of average confidence per keypoint across all clips

Skeleton limbs are drawn with distinct colours by body region (head- yellow, torso- cyan, left arm- green, right arm- orange, hips/core- purple, left leg- light blue, right leg- pink).

---

## Notebook 4 - Model Comparison (04_model_comparison.ipynb)

Trains a second model (**YOLOv8s - small**) on the same dataset and compares it against YOLOv8n.

**YOLOv8s training configuration**

| Parameter | Value |
|---|---|
| Base model | yolov8s.pt |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Device | CPU |
| Early stopping patience | 15 epochs |

Both models are evaluated on the same test split using model.val(split = "test").

**Comparison metrics collected**

- Precision, Recall, mAP@0.50, mAP@0.50:0.95
- Inference speed (ms/image)

**Comparison outputs saved to "comparison/"**

| File | Description |
|---|---|
| 01_metric_comparison.png | Grouped bar chart - all 4 metrics side by side |
| 02_loss_curves.png | Box loss, mAP@0.50, and Precision training curves for both models |
| 03_pr_curves.png | Precision-Recall curves (side by side) |
| 04_speed_vs_accuracy.png | Scatter plot - inference speed vs mAP@0.50 (bubble size = model size) |
| comparison_results.csv | Full numeric results table |


---

## Repository Structure

```
Code_PAS02/
│
├── 01_extract_frames.ipynb       # Frame extraction from raw videos
├── 02_train_yolov8.ipynb         # Dataset prep + YOLOv8n training
├── 03_keypoint_pose.ipynb        # Pose estimation with YOLOv8n-pose
├── 04_model_comparison.ipynb     # YOLOv8s training + model comparison
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml                 
│   ├── test_results.csv          # YOLOv8n test set metrics
│   ├── runs/
│   │   ├── sports_player_detection/       # YOLOv8n training outputs
│   │   └── sports_player_detection_small/ # YOLOv8s training outputs
│   └── comparison/               # Cross-model comparison outputs
│
├── Football/
│   ├── Football 01.mp4 ... Football 12.mp4
│   ├── output_keypoints/         # Pose-annotated videos
│   └── screenshots/              # Keypoint frame screenshots
│
└── Report/                       # Report (PDF)
```

---

## How to Run

> **Important** All paths in the notebooks are set to absolute Windows paths under "D:\MSc\DS 5216 - Artificial Intelligence\PAS02_DTS2430\". Update these directory.

Run the notebooks in order

```
1. 01_extract_frames.ipynb    
2. 02_train_yolov8.ipynb     
3. 03_keypoint_pose.ipynb  
4. 04_model_comparison.ipynb 
```

No GPU is required. All training and inference runs on CPU.

---

## Results Summary

Detailed performance metrics, loss curves, Precision-Recall curves, and qualitative detection screenshots are included in the project report and the Google Drive folder.

Key outputs to review
- **runs/sports_player_detection/results.png** - YOLOv8n training metrics
- <img width="1207" height="600" alt="image" src="https://github.com/user-attachments/assets/42d768a9-2b8d-43cd-aa6c-0956ee5ea0da" />

- **runs/sports_player_detection_small/results.png** - YOLOv8s training metrics
- <img width="1213" height="606" alt="image" src="https://github.com/user-attachments/assets/799027fb-3198-433e-9d85-217ba8f7a024" />

- **comparison/01_metric_comparison.png** - side-by-side model comparison
- <img width="1012" height="607" alt="image" src="https://github.com/user-attachments/assets/b0624abf-f31c-4cc5-a3a9-85c4e7fc374e" />

- **output_keypoints/keypoint_results.png** - pose estimation summary
- <img width="197" height="248" alt="image" src="https://github.com/user-attachments/assets/b48c08e7-a67b-4280-9a5e-f6e6d1342857" />
- <img width="541" height="450" alt="image" src="https://github.com/user-attachments/assets/41243596-6b33-46c3-9f14-70406ecae3c3" />

- **06 - Player Detection [Bounding Box].avi** (Drive) - final annotated video
- <img width="427" height="252" alt="image" src="https://github.com/user-attachments/assets/6bf7c9f9-b487-4010-806d-d3e591db0872" />
  



---

## References

- Ultralytics YOLOv8 - https://github.com/ultralytics/ultralytics
- LabelImg - https://github.com/HumanSignal/labelImg
- COCO Keypoint format - https://cocodataset.org/#keypoints-2017
