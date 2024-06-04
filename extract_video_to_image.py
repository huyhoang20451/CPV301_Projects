import os
import cv2

# List of video names
videos = ['Huy_Hoang', 'Quoc_Dat', 'Vu_Hai']

# Loop through each video
for video in videos:
    # Create a directory for each video if it doesn't exist
    output_dir = f'./images/{video}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(f'./Dataset_video/{video}.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open video {video}.mp4")
        continue
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load the face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Error: Could not load face cascade.")
            break
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Loop through detected faces and draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Save the detected face image
            face_filename = f'{output_dir}/{video}_{count}.jpg'
            cv2.imwrite(face_filename, roi_gray)
            print(f"Saved face image {face_filename}")
            count += 1
    
    # Release the video capture object
    cap.release()

print("Processing complete.")
