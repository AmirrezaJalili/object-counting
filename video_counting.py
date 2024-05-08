from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os

# Set environment variable KMP_DUPLICATE_LIB_OK to TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------------

path = 'C:/Users/ASUS/Desktop/counting_yolov/'
model = YOLO(path + "yolov8n.pt")
cap = cv2.VideoCapture(path + "SampleVideo.MP4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 800), (1000, 800), (1000, 600), (20, 600)]
line_points = [(0, 500), (1000, 500)]

# Video writer
video_writer = cv2.VideoWriter(path + "object_counting_output3.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 #reg_pts=region_points,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, save_txt = True, save = True)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()