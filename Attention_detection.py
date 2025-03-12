import cv2
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Emotion to engagement score mapping
EMOTION_SCORES = {
    "happy": 90, "neutral": 80, "surprise": 75, 
    "fear": 60, "sad": 50, "angry": 40, "disgust": 30
}

# Data storage
engagement_data = []

def analyze_attention():
    """Capture frame, analyze emotions, facial features, and calculate attention score."""
    global engagement_data
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            break
        
        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Initialize attention metrics
        brow_furrowed = False
        eyes_open = False
        looking_away = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get key facial landmarks
                left_brow = face_landmarks.landmark[70]  # Left eyebrow inner
                right_brow = face_landmarks.landmark[300]  # Right eyebrow inner
                left_eye = face_landmarks.landmark[159]  # Left eye
                right_eye = face_landmarks.landmark[386]  # Right eye
                nose = face_landmarks.landmark[1]  # Nose tip
                
                # Brow furrow detection (y-position difference)
                if abs(left_brow.y - right_brow.y) < 0.02:
                    brow_furrowed = True  # Indicates concentration/confusion
                
                # Eye openness detection (height difference)
                if abs(left_eye.y - face_landmarks.landmark[145].y) > 0.01 or \
                   abs(right_eye.y - face_landmarks.landmark[374].y) > 0.01:
                    eyes_open = True  # Eyes are open (attentive)

                # Head tilt detection (nose x-position)
                if nose.x < 0.3 or nose.x > 0.7:  
                    looking_away = True  # Student is not facing the screen
        
        try:
            # Emotion analysis using DeepFace
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            emotion_score = EMOTION_SCORES.get(emotion, 50)

            # Compute final attention score
            attention_score = emotion_score
            if brow_furrowed: attention_score += 5  # Slight boost if focused
            if not eyes_open: attention_score -= 20  # Deduction if eyes closed
            if looking_away: attention_score -= 15  # Deduction if not facing screen
            
            # Ensure score stays in range [0,100]
            attention_score = max(0, min(100, attention_score))

            # Timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store data
            engagement_data.append([timestamp, emotion, attention_score])
            print(f"[{timestamp}] Emotion: {emotion} | Attention Score: {attention_score}")

            # Save to CSV
            df = pd.DataFrame(engagement_data, columns=["Timestamp", "Emotion", "Score"])
            df.to_csv("engagement_scores.csv", index=False)

        except Exception as e:
            print("Error detecting face/emotion:", str(e))

        # Wait 4 minutes before next analysis
        time.sleep(240)

# Run analysis in a separate thread
emotion_thread = threading.Thread(target=analyze_attention)
emotion_thread.daemon = True
emotion_thread.start()

# Display webcam feed (optional)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam - Engagement Tracker", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
