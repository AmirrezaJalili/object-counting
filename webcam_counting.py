from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import os
import time

current_time = time.time()

# Set environment variable KMP_DUPLICATE_LIB_OK to TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------------

path = 'C:/Users/ASUS/Desktop/counting_yolov/'
model = YOLO(path + "yolov8n.pt")
cap = cv2.VideoCapture(0)

'''
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w, h, fps)
# Define region points
region_points = [(20, 800), (1000, 800), (1000, 600), (20, 600)]
'''

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter('webcam_counting.avi',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 10, size)
region_points = [(0, 380), (640, 380), (640, 450), (0, 450)]
line_points = [(0, 400), (640, 400)]

# Video writer
'''
video_writer = cv2.VideoWriter( "'C:/Users/ASUS/Desktop/counting_webcam/'object_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))
'''

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 #reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True, save_txt = True, save = True)

    im0 = counter.start_counting(im0, tracks)

    #in_counts = counter.in_counts
    #out_counts = counter.out_counts
    #print(f"In_Counts - {in_counts} and Out_Counts - {out_counts}")

    result.write(im0)
    
    if time.time() > current_time + 20:
        break

cap.release()
result.release()
cv2.destroyAllWindows()
        
