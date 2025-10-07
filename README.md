# Brain-Tumor-Classification-Using-Deep-Learning-Techniques

## 📘 Project Overview
This project demonstrates how **deep learning** can be applied to automate brain tumor diagnosis from MRI images.  
By leveraging **transfer learning** on pre-trained models (VGG16 and DenseNet121) and integrating them into a **Flask web application**, this system enables fast, accurate, and accessible classification for medical professionals.

---

## 🎯 Problem Statement
Manual diagnosis of brain tumors from MRI scans is **time-consuming**, **labor-intensive**, and susceptible to human error.  
The goal of this project is to build a **deep learning-based classification model** that automates tumor detection, improving diagnostic speed and reliability.

---

## 🧩 Objectives
- Develop a CNN-based model to classify brain MRI images into four categories.
- Compare **VGG16** and **DenseNet121** architectures for accuracy and efficiency.
- Design and deploy a **Flask-based GUI** for real-time tumor prediction.
- Improve accessibility for medical practitioners through a simple web interface.

---

## 📚 Literature & Technology Review

### 🔬 Literature Review
Deep learning models, especially **Convolutional Neural Networks (CNNs)**, have revolutionized medical image classification.  
Pre-trained architectures like **VGG16** and **DenseNet** provide high accuracy through feature reuse and transfer learning.

### 🧰 Technology Stack
| Component | Technology Used |
|------------|----------------|
| **Programming Language** | Python |
| **Deep Learning Libraries** | TensorFlow, Keras |
| **Web Framework** | Flask |
| **Environment** | Google Colab |
| **Frontend** | HTML, CSS |
| **Models** | VGG16, DenseNet121 (Transfer Learning) |

---

## 🧠 Dataset Description
**Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Detail | Description |
|--------|--------------|
| **Size** | ~7,022 MRI images |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary Tumor |
| **Format** | Grayscale images (converted to RGB) |
| **Dimensions** | Resized to 150×150 pixels |

### 🔄 Data Augmentation
To improve generalization:
- Rescaling pixel values (0–1 range)
- Random rotations
- Brightness adjustments
- Shifts and flips



---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removed noise using bilateral filters  
- Converted grayscale → RGB  
- Normalized and resized all images to 150×150  

### 2️⃣ Model Architectures

#### **VGG16**
- 16 layers (13 conv + 3 dense)  
- Small 3×3 filters for spatial feature extraction  
- Max pooling (2×2) to reduce spatial dimensions  
- ReLU activation  
- Fully connected layers: 4096 → 4096 → 1000 (Softmax output)

#### **DenseNet121**
- Dense connections between layers for better gradient flow  
- Growth rate: 32  
- Includes bottleneck and transition layers  
- Global average pooling + Softmax classifier  
- Parameter efficient and faster convergence  


---

## 🧪 Model Training and Evaluation
| Parameter | Value |
|------------|--------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | Categorical Crossentropy |
| **Epochs** | 20 |
| **Batch Size** | 32 |

### 📈 Model Performance
| Model | Accuracy (Test Set) |
|--------|--------------------|
| **VGG16** | **97 %** |
| **DenseNet121** | 96 % |

Both models achieved strong accuracy, with **VGG16 performing slightly better** in this dataset.

---

## 🖥️ GUI Web Application
A **Flask-based GUI** was created to allow medical professionals to upload MRI images and receive instant predictions.

### Features
- 🖼️ **Image Upload:** Accepts `.png`, `.jpg`, `.jpeg`
- ⚙️ **Preprocessing:** Automatic resizing and normalization
- 🧠 **Prediction:** Displays predicted tumor category
- 🌐 **Web Deployment:** Runs locally or on a server for testing

**How It Works**
1. Upload MRI image via the web interface  
2. Model loads trained `.h5` files (`vgg_brain.h5` or `densenet_brain.h5`)  
3. System preprocesses image → runs prediction → displays output  



---

## 🚀 Code Execution

### 🔹 Model Development
1. Open **`brain_tumor.ipynb`** in Google Colab.  
2. Mount Google Drive and upload the dataset.  
3. Run all cells to train models and save `.h5` files.


