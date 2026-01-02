import cv2
import os
from ultralytics import YOLO

# Configuration
MODEL_PATH = '../model/best.pt'
IMAGE_PATH = '../data/129ec4f74dc25caa.jpg'
VIDEO_PATH = '../data/test-video.mp4'
OUTPUT_DIR = '../data'


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)

    if os.path.exists(IMAGE_PATH):
        file_name = os.path.basename(IMAGE_PATH)
        name, ext = os.path.splitext(file_name)
        image_save_path = os.path.join(OUTPUT_DIR, f"{name}(labeled){ext}")

        results = model.predict(source=IMAGE_PATH, save=False, conf=0.5, device='cuda:0')

        # Plot results and save
        annotated_img = results[0].plot()
        cv2.imwrite(image_save_path, annotated_img)
        print(f"Image saved successfully to: {image_save_path}")

    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        video_name = os.path.basename(VIDEO_PATH)
        v_name, v_ext = os.path.splitext(video_name)
        video_save_path = os.path.join(OUTPUT_DIR, f"{v_name}(labeled).mp4")

        # Initialize Video Writer (MP4V codec)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))

        print(f"Processing video... Saving to: {video_save_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, save=False, conf=0.5, imgsz=640, device='cuda:0')
            annotated_frame = results[0].plot()

            # Write the frame to the output file
            out.write(annotated_frame)

            cv2.imshow("Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")


if __name__ == '__main__':
    main()