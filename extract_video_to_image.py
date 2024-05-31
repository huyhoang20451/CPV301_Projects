import cv2
import os

# Hàm trích xuất khung hình từ video
def extract_frames(video_path, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    
    # Mở tệp video
    cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra xem video có mở được không
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Khởi tạo bộ đếm khung hình
    frame_number = 0

    # Vòng lặp để đọc khung hình
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lưu khung hình thành tệp ảnh
        frame_filename = os.path.join(output_folder, f'frame_{frame_number:04d}.png')
        cv2.imwrite(frame_filename, frame)

        # Tăng bộ đếm khung hình
        frame_number += 1

    # Giải phóng đối tượng capture
    cap.release()

    print(f"Extracted {frame_number} frames from {video_path}.")

# Đường dẫn tới các video
videos = ['Dataset/Huy_Hoàng.mp4', 'Dataset/Quốc_Đạt.mp4', 'Dataset/Vũ_Hải.mp4']

# Thư mục đầu ra cho từng video
output_folders = ['Dataset_frame_extract/frames_video_HuyHoang', 'Dataset_frame_extract/frames_video_QuocDat', 'Dataset_frame_extract/frames_video_VuHai']

# Trích xuất khung hình cho từng video
for video, folder in zip(videos, output_folders):
    extract_frames(video, folder)
