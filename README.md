Hello

In this project, I developed an end-to-end audio emotion recognition system that analyzes speech signals and classifies emotional states using machine learning.

Project Overview

🔹 Model Architecture:
I implemented a Multi-Layer Perceptron (MLPClassifier) to classify emotions based on extracted acoustic features.

🔹 Objective:
To automatically detect and classify emotions from .wav audio files using signal processing and supervised learning techniques.

Technical Pipeline

🔹 Audio Feature Engineering:
Extracted high-level acoustic features using MFCC (Mel-Frequency Cepstral Coefficients), Chroma, and Mel Spectrogram representations via librosa.
These features were aggregated using mean pooling to form a compact numerical representation of each audio sample.

🔹 Data Preprocessing:
Applied StandardScaler normalization to ensure stable and efficient convergence of the MLP model.

🔹 Dataset Handling:
Structured dataset parsing with automatic emotion label extraction from file naming conventions.

🔹 Model Training:
Split dataset into 80% training and 20% testing using stratified sampling to maintain class balance.
Trained the MLP model with adaptive learning rate and optimized hyperparameters.

🔹 Evaluation:
Performance measured using:

Accuracy Score

Classification Report (Precision, Recall, F1-Score)

🔹 Model Persistence:
Saved trained model and scaler objects using joblib for future inference or deployment.

Machine Learning Approach

Unlike deep learning-based CNN audio models, this system focuses on:

Signal processing-based feature extraction

Classical neural network classification (MLP)

Efficient and lightweight architecture suitable for structured ML pipelines

 Tech Stack

Python, NumPy, Librosa, Scikit-learn, Joblib, Logging

Feel free to review the architecture and share your thoughts or improvement suggestions.

Author: Mustafa Alpergün

Merhaba

Bu projede, konuşma verilerinden duygusal durumu analiz eden ve sınıflandıran uçtan uca bir Ses Tabanlı Duygu Tanıma Sistemi geliştirdim.

Proje Özeti

🔹 Model Mimarisi:
Ses özelliklerini kullanarak duygu sınıflandırması yapan bir Çok Katmanlı Algılayıcı (MLPClassifier) modeli tasarladım.

🔹 Amaç:
.wav formatındaki konuşma kayıtlarından duygusal ifadeleri otomatik olarak tespit etmek.

 Teknik Süreç

🔹 Özellik Çıkarımı (Feature Extraction):
Librosa kütüphanesi ile:

MFCC

Chroma

Mel Spectrogram

özellikleri çıkarıldı ve her ses dosyası için sayısal özellik vektörleri oluşturuldu.

🔹 Veri Normalizasyonu:
MLP algoritmasının sağlıklı çalışması için StandardScaler ile standardizasyon uygulandı.

🔹 Veri Bölme:
Veri seti %80 eğitim, %20 test olacak şekilde dengeli biçimde ayrıldı.

🔹 Model Eğitimi ve Değerlendirme:
Model doğruluk oranı ve detaylı sınıflandırma raporları ile analiz edildi.

🔹 Model Kaydetme:
Eğitilmiş model ve ölçekleyici joblib kullanılarak diske kaydedildi.

 Kullanılan Teknolojiler

Python, NumPy, Librosa, Scikit-learn, Joblib

Kod yapısını incelemek ve geliştirme önerilerinizi paylaşmak isterseniz geri bildirimlerinizi memnuniyetle karşılarım.

👇 Yazar: Mustafa Alpergün
