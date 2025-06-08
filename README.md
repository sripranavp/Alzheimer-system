# Alzheimer's Disease Detection using Deep Learning

This project focuses on the **early detection of Alzheimer's Disease (AD)** using deep learning models trained on structural MRI images. We implement a hybrid approach combining **CNN**, **MobileNet**, and **DenseNet121** architectures, and fuse their predictions using **ensemble learning** to improve accuracy and sensitivity.

## 🧠 Problem Statement

Traditional methods for Alzheimer's detection (clinical exams, cognitive tests, and neuroimaging) often lack early-stage accuracy, are subjective, and are not widely accessible. This project aims to build a **cost-effective, automated detection system** using machine learning.

---

## 🎯 Objectives

- Preprocess and balance an MRI dataset of Alzheimer's cases.
- Train and evaluate CNN, MobileNet, and DenseNet121 models.
- Apply **Soft Voting Ensemble** to combine predictions.
- Build a **web-based interface** for user-friendly prediction.

---

## 📂 Dataset

- MRI Brain Scans of:
  - **Non-Demented** – 2500 images
  - **Very Mild Demented** – 1000 images
  - **Mild Demented** – 1700 images
  - **Moderate Demented** – 50 images

### Preprocessing
- Image resizing to **176×176**
- Normalization using **rescale = 1/255**
- Applied **SMOTE** for class balancing

---

## 🧠 Model Architectures

### 1. Convolutional Neural Network (CNN)
- 3 Convolutional layers (3×3 kernels)
- 3 MaxPooling layers (2×2)
- Fully Connected Dense Layer (1024 neurons)
- ReLU + Softmax for classification

### 2. MobileNet
- 13 Depthwise Separable Convolution layers
- Global Average Pooling
- Softmax output for 4 classes

### 3. DenseNet121
- Pretrained on ImageNet
- Dense connectivity pattern for feature reuse
- GlobalAveragePooling2D + Dropout + Dense layers

---

## 🔀 Ensemble Model

**Soft Voting Ensemble**:  
Final prediction is the average of class probabilities from CNN, MobileNet, and DenseNet:

Then `argmax(P_ensemble)` is taken as the final class.

---

## 🖥️ Web Interface

- **Register/Login**
- Upload MRI scan
- Get prediction (AD class)

---

## 📈 Evaluation Metrics

- **Confusion Matrix**
- **Classification Report**
- **Accuracy, Precision, Recall, F1-Score**

---

## 🚀 Future Work

- Add PET/fMRI for multimodal analysis
- Integrate **Vision Transformers (ViTs)**
- Real-time mobile deployment using model quantization
- Longitudinal data analysis for disease progression

---

## 🧑‍💻 Authors

- **Sri Pranav P** – Developer & Researcher  
- **Dr. K. L. Nisha** – Project Guide, Amrita Vishwa Vidyapeetham

---

## 📚 References

- [Alzheimer's Research & Therapy, 2022](https://doi.org/10.1186/s13195-022-01047-y)  
- IEEE Access & Brain Sci. articles on AD detection using Deep Learning  
