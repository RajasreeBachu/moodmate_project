import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

# --- Load model ---
model = tf.keras.models.load_model(r"C:\Users\Admin\Desktop\MoodMate\notebooks\models\mobilenet_emotion_model.keras")

# --- Emotion labels ---
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# --- Load Spotify dataset ---
spotify_df = pd.read_csv(r"C:\Users\Admin\Desktop\MoodMate\data\spotify_tracks.csv")

# --- Emotion → Genre Mapping ---
def recommend_songs_by_emotion(emotion, num_recs=5):
    emotion_to_genres = {
        "happy": ["pop", "dance", "house"],
        "sad": ["acoustic", "soul", "piano"],
        "angry": ["rock", "metal"],
        "fear": ["ambient", "instrumental"],
        "surprise": ["edm", "party"],
        "disgust": ["grunge", "hardcore"],
        "neutral": ["chill", "indie", "lo-fi"]
    }

    genres = emotion_to_genres.get(emotion, [])
    filtered = spotify_df[spotify_df['genre'].isin(genres)]
    filtered = filtered.sort_values(by='popularity', ascending=False)

    # Add Spotify clickable link column
    filtered['spotify_link'] = filtered['name'].apply(
        lambda x: f'<a href="https://open.spotify.com/search/{x.replace(" ", "%20")}" target="_blank">🎧 Listen</a>'
    )

    recs = filtered[['name', 'artists', 'genre', 'popularity', 'spotify_link']].head(num_recs)
    return recs

# --- Text prompt → Emotion ---
def detect_emotion_from_text(prompt):
    prompt = prompt.lower()
    if any(word in prompt for word in ["happy", "joy", "excited", "fun", "good"]):
        return "happy"
    elif any(word in prompt for word in ["sad", "lonely", "cry", "kill" "down", "tired", "dying"]):
        return "sad"
    elif any(word in prompt for word in ["angry", "mad", "furious", "irritated"]):
        return "angry"
    elif any(word in prompt for word in ["afraid", "scared", "nervous", "fear"]):
        return "fear"
    elif any(word in prompt for word in ["wow", "surprised", "unexpected"]):
        return "surprise"
    elif any(word in prompt for word in ["disgust", "gross", "yuck"]):
        return "disgust"
    else:
        return "neutral"

# --- Streamlit UI ---
st.set_page_config(page_title="AI MoodMate", page_icon="🎵", layout="centered")
st.title("🎵 AI MoodMate – Emotion-Based Music Recommender")
st.markdown("Upload your face image *or* type how you feel to get personalized music 🎧")

option = st.radio("Choose input mode:", ["Upload Image", "Text Prompt"])

# --- IMAGE MODE ---
# --- IMAGE MODE ---
if option == "Upload Image":
    st.markdown("📸 Capture from webcam or upload a face image below:")

    # Webcam and upload options
    camera_image = st.camera_input("Take a picture")
    uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "png"])

    image_data = None
    if camera_image is not None:
        image_data = camera_image
    elif uploaded_file is not None:
        image_data = uploaded_file

    if image_data is not None:
        # Load and preprocess image
        image = tf.keras.utils.load_img(image_data, color_mode='grayscale', target_size=(48,48))
        image = tf.keras.utils.img_to_array(image) / 255.0
        st.image(image.squeeze(), caption="Captured/Uploaded Image", use_column_width=True)

        # Convert grayscale → RGB for MobileNet
        img_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
        img_rgb = tf.image.resize(img_rgb, (224,224))
        img_rgb = tf.expand_dims(img_rgb, 0)

        # Predict emotion
        preds = model.predict(img_rgb)
        emotion = emotion_labels[np.argmax(preds)]
        st.subheader(f"Detected Emotion: {emotion.capitalize()}")

        # Recommend songs
        recs = recommend_songs_by_emotion(emotion)
        st.write("### 🎧 Recommended Songs:")
        st.write(recs.to_html(escape=False, index=False), unsafe_allow_html=True)
        # Predict emotion
        img_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image))
        img_rgb = tf.image.resize(img_rgb, (224,224))
        img_rgb = tf.expand_dims(img_rgb, 0)
        preds = model.predict(img_rgb)
        emotion = emotion_labels[np.argmax(preds)]
        st.subheader(f"Detected Emotion: {emotion.capitalize()}")

        # Recommend songs
        recs = recommend_songs_by_emotion(emotion)
        st.write(recs.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.write("### 🎧 Recommended Songs:")

        # Display Spotify-style cards
        for _, row in recs.iterrows():
            st.markdown(f"""
            🎶 *{row['name']}*  
            👤 {row['artists']}  
            🏷 {row['genre']}  
            ⭐ Popularity: {row['popularity']}  
            [🎧 Listen on Spotify](https://open.spotify.com/search/{row['name'].replace(' ', '%20')})  
            ---
            """)

# --- TEXT MODE ---
elif option == "Text Prompt":
    from textblob import TextBlob

    st.subheader("💬 Or enter a text prompt for emotion detection")
    user_text = st.text_input("Type how you feel (e.g., 'I feel sad today')")

    emotion = None

    if user_text:
        blob = TextBlob(user_text)
        sentiment = blob.sentiment.polarity  # -1 (negative) → +1 (positive)
        if sentiment > 0.2:
            emotion = "happy"
        elif sentiment < -0.2:
            emotion = "sad"
        else:
            emotion = "neutral"
        st.write(f"Detected emotion from text: *{emotion}*")

        recs = recommend_songs_by_emotion(emotion)
        st.write("### 🎧 Recommended Songs:")
        st.write(recs.to_html(escape=False, index=False), unsafe_allow_html=True)
