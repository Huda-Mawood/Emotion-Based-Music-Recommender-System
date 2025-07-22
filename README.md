# Emotion-Based Music Recommender System

**Team Members:** Muhammad Hanzala, Ahmed Medhat, Huda Mawood  
**Date:** June 26, 2025

---

## Contents

1. Project Overview  
2. Model Selection Process  
3. Dataset Structure  
4. Data Preprocessing  
5. Model Architecture  
6. Training Configuration  
7. Model Evaluation and Final Model Choice  
8. Emotion Prediction from Image  
9. Music Recommendation System  
10. Graphical User Interface (GUI)  
11. Real-time Webcam Detection  
12. Libraries Used  
13. Conclusion  
14. References  

---

## 1 Project Overview

This system detects emotions from human facial expressions using deep learning, then recommends music based on the predicted emotion. The goal is to provide a personalized mood-aware music experience.

---

## 2 Model Selection Process

We experimented with multiple deep learning models, including:

- ResNet50 trained on a custom dataset (841 images)  
- Xception pretrained on FER2013  
- ResNet pretrained on FER2013  
- EfficientNetB3 fine-tuned on a translated Roboflow dataset  
- EfficientNetB0 trained on AffectNet  

Among these, **EfficientNetB0** was chosen as the final model due to its:

- Stable and consistent predictions  
- Balanced performance across emotion classes  
- Suitability for real-time deployment with reasonable computational cost  

This selection was based on quantitative evaluations including accuracy, loss, and class-wise precision and recall.

---

## 3 Dataset Structure

The primary dataset is organized by emotion classes as follows:

dataset
├── Train
│ ├── happy
│ ├── sad
│ ├── angry
│ ├── neutral
│ ├── fear
│ ├── surprise
│ ├── contempt
│ └── disgust


---

## 4 Data Preprocessing

Applied data augmentation and preprocessing including:

- Image rescaling to [0,1]  
- Rotation, zoom, and shift transformations  
- Horizontal flips  
- Validation split of 15%

---

## 5 Model Architecture

- Base: EfficientNetB0 (include_top=True)  
- GlobalAveragePooling2D  
- Dropout layers (0.5 and 0.3)  
- Dense layer with 128 units and ReLU activation  
- Final Dense layer with Softmax activation for classification  

---

## 6 Training Configuration

- Optimizer: Adam (learning rate = 1e-5)  
- Loss: Categorical Crossentropy  
- Epochs: 15  
- Batch Size: 32  
- Class weights calculated to address class imbalance  

---

## 7 Model Evaluation and Final Model Choice

- Final validation accuracy reached approximately 61%  
- Confusion mainly between subtle emotions like neutral, sad, and contempt  
- Model showed good performance in real-time webcam testing with low latency (~1.5 sec per frame)  
- EfficientNetB0 balanced accuracy and computational efficiency, making it ideal for integration  

---

## 8 Emotion Prediction from Image

The model takes a facial image input, processes it through the CNN, and outputs the predicted emotion. This emotion is then mapped to corresponding music genres.

---

## 9 Music Recommendation System

| Emotion  | Recommended Music Genres      |
|----------|------------------------------|
| Anger    | Rock, Metal                  |
| Contempt | Classical, Jazz              |
| Disgust  | Metal, Rock                  |
| Fear     | Classical                   |
| Happy    | Pop, Disco                  |
| Neutral  | Hip-Hop, Reggae             |
| Sad      | Blues, Jazz                 |
| Surprise | Pop, Disco                  |

---

## 10 Graphical User Interface (GUI)

Built using tkinter, the GUI:

- Displays the detected emotion  
- Shows recommended music tracks  
- Allows user to play or stop tracks  

---

## 11 Real-time Webcam Detection

- Captures live frames using OpenCV  
- Performs emotion prediction in real-time  
- Displays results through the GUI  

---

## 12 Libraries Used

- TensorFlow, Keras  
- OpenCV  
- Numpy, Matplotlib, Seaborn  
- Tkinter  
- Pygame  

---

## 13 Conclusion

Through comprehensive experimentation, EfficientNetB0 was identified as the optimal model balancing accuracy and speed. Our system effectively connects facial emotion recognition with personalized music recommendations in real time. Future improvements include expanding the dataset diversity and enhancing accuracy for subtle emotions.

---

## 14 References

- DeepFace Library: https://github.com/serengil/deepface  
- EfficientNetB0 Dataset (AffectNet via Kaggle): https://www.kaggle.com/datasets/mstjebashazida/affectnet  
- FER2013 Dataset: https://www.kaggle.com/datasets/msambare/fer2013  
- Roboflow Dataset for ResNet50: https://universe.roboflow.com/hipster-pi5hv/custom-workflow-object-detection-kuaeb  
- Roboflow Dataset for EfficientNetB3: https://universe.roboflow.com/reconocimiento-facial-irbfo/emociones-qbf5i  
- EfficientNet Paper: https://arxiv.org/abs/1905.11946  
- Xception Paper: https://arxiv.org/abs/1610.02357  
- OpenCV Haar Cascades: https://github.com/opencv/opencv/tree/master/data/haarcascades  
- TensorFlow: https://www.tensorflow.org/  
- Keras: https://keras.io/  
- Pygame Documentation: https://www.pygame.org/docs/
