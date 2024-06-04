import numpy as np
import cv2
import os

# Load the images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        if person_name not in label_dict:
            label_dict[person_name] = current_label
            current_label += 1

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                resized_img = cv2.resize(img, (100, 100))
                images.append(resized_img)
                labels.append(label_dict[person_name])

    return images, labels, label_dict

# Train the model
def train_face_recognizer(data_folder):
    images, labels, label_dict = load_images_from_folder(data_folder)

    # Convert labels to numpy array
    labels = np.array(labels)

    # Create the face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the face recognizer
    face_recognizer.train(images, labels)

    return face_recognizer, label_dict

# Save the recognizer
def save_recognizer(recognizer, output_file):
    recognizer.save(output_file)

# Save the label dictionary
def save_label_dict(label_dict, label_dict_path):
    with open(label_dict_path, 'w') as f:
        for name, label in label_dict.items():
            f.write(f"{name},{label}\n")

# Main function
if __name__ == '__main__':
    # Define the data folder
    data_folder = './images'

    # Train the face recognizer
    face_recognizer, label_dict = train_face_recognizer(data_folder)

    # Save the face recognizer
    save_recognizer(face_recognizer, 'face_recognizer.yml')

    # Save the label dictionary
    save_label_dict(label_dict, 'label_dict.txt')

    print("Training complete.")
    
    #Test the result
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognizer.yml')
    label_dict = {}
    with open('label_dict.txt', 'r') as f:
        for line in f:
            name, label = line.strip().split(',')
            label_dict[int(label)] = name
    