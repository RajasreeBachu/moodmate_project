🎵 MoodMate: AI Emotion Detection & Music Recommendation System

An intelligent system that detects user emotions and recommends music in real time.


---

📌 Project Overview

MoodMate is an AI-powered application that predicts a user’s emotional state from facial expressions or text input and recommends music that aligns with or improves their mood.
It integrates Computer Vision / NLP with Music Recommendation Systems to deliver an interactive, emotion-aware experience.


---

🎯 Objectives

Detect emotions using facial images (FER-2013) or text input.

Build a content-based music recommendation engine.

Map user emotions to suitable music genres / tags.

Develop an interactive UI for real-time emotion → music suggestions.

Deploy a complete end-to-end system.



---

✅ Key Outcomes

Hands-on experience in image preprocessing, emotion classification, and content-based recommendation.

Working knowledge of MobileNetV2 / CNN / BERT (optional).

Complete integration of ML models with a user-friendly interface.

A fully functional prototype with real-time predictions.



---

📂 Datasets Used

1. Emotion Recognition

FER-2013 (Kaggle)
Grayscale 48×48 facial emotion dataset
Classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral


2. Music Recommendation

Million Song Dataset (subset)

Last.fm Tags Dataset

Optional: RAVDESS for multimodal audio emotion analysis



---

🧱 System Architecture

+-------------------------+
               |   User Input (Image/Text)|
               +-----------+-------------+
                           |
                           v
                +-----------------------+
                | Emotion Detection     |
                | (CNN / MobileNetV2)   |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Emotion → Music Mapping|
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Recommendation Engine |
                | (TF-IDF / Cosine Sim) |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Streamlit UI Output   |
                +-----------------------+


---

🧩 Modules Implemented

1️⃣ Data Collection & Preprocessing

Load and clean FER-2013 CSV dataset

Convert pixel strings → 48×48 grayscale images

Normalize and split into Train / Validation / Test

Process Last.fm music tags, genres, moods

Extract features: TF-IDF tags, tempo, energy, valence



---

2️⃣ Emotion Detection Module

Method 1: CNN from scratch

Method 2 (recommended): MobileNetV2 pretrained model

Resize images to 224×224×3

Freeze + fine-tune layers

Achieve 75–85% accuracy




---

3️⃣ Music Recommendation Engine

Content-based filtering

TF-IDF vectorization of music tags

Cosine similarity for ranking songs

Emotion → Tag mapping

Happy    → pop, dance, upbeat  
Sad      → acoustic, mellow, low-energy  
Angry    → rock, metal, fast-tempo  
Calm     → ambient, instrumental, soft



---

4️⃣ UI & Real-Time Integration

Interactive app built using Streamlit

Features:

Upload face image

Detect emotion

Generate recommended playlist

Display top 5–10 songs with metadata




---

5️⃣ Deployment & Final Output

Export model: mobilenetv2_emotion.keras

Streamlit front-end

Documentation + demo video
