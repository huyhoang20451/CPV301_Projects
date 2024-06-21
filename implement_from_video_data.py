import cv2
import numpy as np

# Load the trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face_recognizer.yml')

# Load the label dictionary
label_dict = {}
with open('label_dict.txt', 'r') as f:
    for line in f:
        name, label = line.strip().split(',')
        label_dict[int(label)] = name

# Function to predict and track faces in a video
def video_face_tracking(video_source):
    # Open the video source (0 for webcam, or 'path/to/video' for a file)
    cap = cv2.VideoCapture(video_source)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        
        # Predict each face found in the frame
        for (x, y, w, h) in faces:
            # Resize the detected face to match the training data
            resized_face = cv2.resize(gray_frame[y:y+h, x:x+w], (100, 100))
            # Predict the label of the face
            label, confidence = recognizer.predict(resized_face)
            # Get the name corresponding to the predicted label
            person_name = label_dict[label]
            
            # Draw a rectangle around the face and put the person's name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{person_name} - {confidence:.2f}", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Example usage
#Đây là video để test nhận diện khuôn mặt.
if __name__ == '__main__':
    video_source = 'Video_detect_test.mp4'  # Use 0 for webcam or 'path/to/video.mp4' for a video file
    # video_source = 0
    video_face_tracking(video_source)
