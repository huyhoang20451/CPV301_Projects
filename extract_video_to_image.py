import os
import cv2

# List of video names
videos = ['Hoang', 'Dat', 'Hai']
id = 1

if id == 1:
    # Initialize face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_id = input('\n nhap ID khuon mat :')
    face_name = input('\n nhap ten khuon mat :')

    # Create a directory for the new face if it doesn't exist
    output_dir = f'images/{face_name}'
    os.makedirs(output_dir, exist_ok=True)

    print('\n Dang khoi tao Camera...')

    num_img = 0 
    #cam setup     
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    while True:
        # Setup camera
        ret, frame = cam.read()

        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            num_img += 1
            
            # Save the detected face image
            file_name = f'{output_dir}/{face_name}_{num_img}.jpg'
            cv2.imwrite(file_name, gray[y:y+h, x:x+w])
            
            cv2.imshow('Detecting', frame)
        
        if num_img >= 20:
            break
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n Exitting...")

    cam.release()
    cv2.destroyAllWindows()

elif id == 2:
    # Loop through each video
    for video in videos:
        # Create a directory for each video if it doesn't exist
        output_dir = f'./images/{video}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the video file
        video_path = f'./Dataset_video/{video}.mp4'
        cap = cv2.VideoCapture(video_path)
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
                
                # Save the detected face image with Vietnamese filename
                face_filename = f'{output_dir}/{video}_{count}.jpg'
                cv2.imwrite(face_filename, roi_gray)
                print(f"Saved face image {face_filename}")
                count += 1
        
        # Release the video capture object
        cap.release()

    print("Processing complete.")
