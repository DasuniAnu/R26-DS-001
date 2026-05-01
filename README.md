# R26-DS-001
Hate Speech and Deep Fake Identification for Sinhala and Tamil Low Recourse Languages.
# Multimodal Hate Speech, Deepfake Detection & Trend Analysis System

## 📌 Overview

This project focuses on detecting harmful and misleading content using a **multimodal approach**, combining text, audio, and video analysis. The system primarily targets **YouTube content**, including video audio and user comments, and is designed for **low-resource languages such as Tamil and Sinhala**.

Additionally, a **Chrome Extension** is developed to identify and highlight harmful content in real-time, supporting safer online interactions.

---

## 🎯 Objectives

* Detect hate speech in Tamil and Sinhala (text and comments)
* Analyze audio from videos to identify aggressive or harmful tone
* Detect deepfake content in videos
* Monitor harmful trends over time
* Provide early escalation for critical content
* Enable real-time detection through a browser extension

---

## 🧩 Project Components

### 🔹 1. Tamil Hate Speech Detection (Text + Audio)

* Text classification using ML models (SVM, Logistic Regression, Naive Bayes)
* Detection of hate speech in YouTube comments
* Audio analysis using MFCC feature extraction
* Multimodal fusion of text and audio predictions

---

### 🔹 2. Sinhala Hate Speech Detection

* Text preprocessing and normalization
* Hate speech detection in Sinhala comments
* Model training using machine learning techniques

---

### 🔹 3. Deepfake Detection (Video)

* Extraction of video frames
* Detection of manipulated or synthetic media
* Classification using video analysis techniques

---

### 🔹 4. Early Escalation System

* Identify high-risk or severe hate content
* Trigger alerts based on content severity
* Support moderation and decision-making

---

### 🔹 5. Trend Analysis Module

* Analyze hate speech patterns over time
* Identify rising harmful trends
* Generate insights for monitoring and reporting

---

### 🔹 6. Chrome Extension (YouTube Integration)

* Detect hate speech in YouTube comments in real-time
* Analyze video audio content for harmful speech
* Highlight or flag harmful content to users
* Provide a user-friendly interface for quick detection

---

### 🔹 7. System Integration

* Combine outputs from all modules
* Generate final predictions and insights
* Manage communication between components

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Librosa (Audio Processing)
* OpenCV (Video Processing)
* JavaScript (Chrome Extension)
* Streamlit / Web UI

---

## 📊 Methodology

1. Data collection from YouTube (comments, audio, video)
2. Data preprocessing and cleaning
3. Feature extraction (TF-IDF, MFCC, video features)
4. Model training and evaluation
5. Multimodal integration
6. Trend analysis and escalation handling
7. Deployment via Chrome Extension
8. Testing and validation

---

## 📁 Project Structure

```text id="yt8f3c"
/data
  /text
  /audio
  /video
/models
/extension
/notebooks
/src
README.md
```

---

## 🚀 Features

* Multilingual support (Tamil & Sinhala)
* YouTube comment and video analysis
* Multimodal detection (Text + Audio + Video)
* Chrome Extension for real-time detection
* Early warning and escalation system
* Trend analysis and reporting
* Modular and scalable design

---

## 📈 Future Improvements

* Improve accuracy using deep learning models
* Real-time streaming analysis
* Expand to more languages
* Deploy as a full cloud-based system

---

## 👥 Team Members

* Member 1 – Tamil Hate Speech 
* Member 2 – Sinhala Hate Speech
* Member 3 – Deepfake Detection
* Member 4 – Escalation & Trend Analysis

---

## 📌 Conclusion

This project demonstrates how multimodal AI can be applied to real-world platforms like YouTube to detect harmful content, provide early warnings, and analyze trends, especially in low-resource language contexts.

---
