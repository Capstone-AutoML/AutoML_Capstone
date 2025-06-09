# Introduction

Wildfires pose a critical threat to ecosystems, infrastructure, and human life. Timely and accurate detection is essential for effective intervention and mitigation. However, developing high-performing object detection models for wildfire detection is often constrained by the lack of labeled data and the time-intensive process of manual annotation.

This project presents an **end-to-end AutoML pipeline** for wildfire detection using a **CI/CD/CT** (Continuous Integration, Continuous Deployment, and Continuous Training) architecture. The pipeline automates the entire lifecycle of a detection model. It starts from raw image collection and continues through pre-labeling, human validation, training, distillation, quantization, and deployment.

## Motivation

Manual labeling of wildfire imagery is time-consuming and error-prone. In addition, models degrade over time as environmental conditions and data distributions shift. Our system aims to continuously learn from new data using a scalable, semi-supervised approach. It automates as much of the machine learning workflow as possible and involves human review only when necessary.

## Key Features

- Automated pre-labeling using YOLOv8 and Grounding DINO  
- Model matching and validation using IoU and confidence thresholds  
- Human-in-the-loop review for mismatches via Label Studio  
- Image augmentation to improve generalization  
- End-to-end training, distillation, and quantization  
- CI/CD/CT-compatible design for regular updates and retraining  

## Updated Pipeline Overview

We revised the original proposal to improve model matching accuracy and simplify dependency management. Notably, we replaced **SAM** (Segment Anything Model) with **Grounding DINO** for zero-shot object detection using natural language prompts. This allowed us to generate diverse bounding box predictions with minimal manual intervention.

### Workflow Overview

1. **Data Collection**  
   Unlabeled wildfire images are collected from remote sensors and placed into a raw image directory.

2. **Pre-labeling (YOLO and Grounding DINO)**  
   Both models generate bounding boxes independently. YOLO is fast and lightweight. Grounding DINO supports natural language prompts.

3. **Matching**  
   Predictions from both models are matched using class name and IoU. Unmatched results are flagged for human review.

4. **Human-in-the-Loop Review**  
   Label Studio is used to manually verify or correct mismatched results.

5. **Augmentation**  
   Verified labeled images are augmented to enrich the dataset.

6. **Training**  
   A new YOLO model is trained on the augmented dataset.

7. **Distillation and Quantization**  
   The full model is distilled into a lightweight version and then quantized for deployment.

8. **Model Registry Update**  
   Trained models are stored in the registry and used for future pre-labeling.

---

This pipeline ensures **scalability**, **adaptability**, and **model freshness** without relying heavily on constant manual labeling. The integration of human review only when needed helps balance efficiency with accuracy.
