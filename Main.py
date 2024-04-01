import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained facial expression recognition model
model = load_model(r'path_to_your_pretrained_model')

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open the default camera
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to match the model's input size
        face_roi = cv2.resize(face_roi, (64, 64))

        # Normalize pixel values
        face_roi = face_roi.astype('float32') / 255.0

        # Reshape the face ROI to match the input shape of the model
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Perform facial expression recognition
        predictions = model.predict(face_roi)
        emotion_index = np.argmax(predictions)
        emotion = emotions[emotion_index]

        # Draw bounding box around the face and show the predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
