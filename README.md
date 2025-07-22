# Emotion-Based Music Recommender System

**Team Members:** Muhammad Hanzala, Ahmed Medhat, Huda Mawood  
**Date:** June 26, 2025

---

## Contents

1. Project Overview  
2. Model Selection Process  
   2.1 ResNet50  
   2.2 Xception (Pretrained on FER2013)  
   2.3 ResNet (Pretrained on FER2013)  
   2.4 EfficientNetB0 Model  
   2.5 EfficientNetB3 Model  
3. Dataset Structure  
4. Data Preprocessing  
5. Model Architecture  
6. Training Configuration  
7. Model Evaluation  
8. Emotion Prediction from Image  
9. Music Recommendation System  
10. Graphical User Interface (GUI)  
11. Real-time Webcam Detection  
12. Libraries Used  
13. Conclusion  
14. References  

---

## 1 Project Overview

This system detects emotions from human facial expressions using a deep learning model, then plays music based on the predicted emotion. The goal is to provide a mood-aware recommendation experience that connects emotion recognition with audio response.

---

## 2 Model Selection Process

We tested several deep learning models with different datasets and configurations. Below is a summary of our experimentation process.

### 2.1 ResNet50
- Model: ResNet50.h5  
- Dataset: Custom (841 images)  
- Source: Preprocessed dataset from Roboflow: https://universe.roboflow.com/hipster-pi5hv/custom-workflow-object-detection-kuaeb  
- Problem: Overfitting due to limited data. Validation accuracy was poor.

### 2.2 Xception (Pretrained on FER2013)
- Model: Xception.h5  
- Dataset: FER2013  
- Source: https://www.kaggle.com/datasets/msambare/fer2013  
- Issue: Poor generalization despite using data augmentation and optimizer tuning.

### 2.3 ResNet (Pretrained on FER2013)
- Model: resnet-fer2013.h5  
- Dataset: FER2013  
- Observation: Continued misclassifications, especially with similar expressions like sad and neutral.

### 2.4 EfficientNetB0 Model
- Model: EfficientNetB0.h5  
- Dataset: AffectNet  
- Source: https://www.kaggle.com/datasets/mstjebashazida/affectnet  
- Performance: Most stable predictions  
- Selected as final model

### 2.5 EfficientNetB3 Model
- Model: EfficientNetB3.h5  
- Dataset: Translated and structured Roboflow dataset (Spanish to English)  
- Source: https://universe.roboflow.com/reconocimiento-facial-irbfo/emociones-qbf5i  
- Result: Handled more classes with fine-tuned precision, especially in expressions with subtle variations.

---

## 3 Dataset Structure

Our primary training dataset is organized by emotion class:

