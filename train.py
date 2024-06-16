import numpy as np
import cv2
import os

# Load the images from folder
def load_images_from_folder(folder):
    #Định nghĩa list các image và label lưu vào images và labels
    images = []
    labels = []
    label_dict = {}
    #Định nghĩa label hiện tại
    current_label = 0

    #Duyệt qua các folder trong thư mục data_folder
    for person_name in os.listdir(folder):
        #Đường dẫn đến thư mục của mỗi người
        person_folder = os.path.join(folder, person_name)
        #Nếu không phải là thư mục thì bỏ qua
        if not os.path.isdir(person_folder):
            continue
        
        #Nếu person_name chưa có trong label_dict thì thêm vào
        if person_name not in label_dict:
            #Thêm person_name vào label_dict và gán label tương ứng
            label_dict[person_name] = current_label
            #Increment label - Tăng label lên 1
            current_label += 1

        #Duyệt qua các file trong thư mục của mỗi người nếu đã có trong label_dict
        for filename in os.listdir(person_folder):
            #Đường dẫn đến ảnh
            img_path = os.path.join(person_folder, filename)
            #Đọc ảnh trắng đen trong thư mục hình đã xử lý sang grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #Resize ảnh về kích thước 100x100 nếu ảnh không rỗng
            if img is not None:
                #Resize ảnh về kích thước 100x100
                resized_img = cv2.resize(img, (100, 100))
                #Thêm ảnh vào list images
                images.append(resized_img)
                #Thêm label tương ứng vào list labels
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

# Lưu model để chuẩn bị cho test
def save_recognizer(recognizer, output_file):
    recognizer.save(output_file)

# Save the label dictionary
def save_label_dict(label_dict, label_dict_path):
    # Save the label dictionary to a file
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
    
    #Load model và label_dict để lưu thành 1 file có tên label_dict.txt và face_recognizer.yml
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognizer.yml')
    label_dict = {}
    with open('label_dict.txt', 'r') as f:
        for line in f:
            name, label = line.strip().split(',')
            label_dict[int(label)] = name
    