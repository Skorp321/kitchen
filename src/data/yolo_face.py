from ultralytics import YOLO
import cv2
import numpy as np
from tqdm.auto import tqdm
import ffmpegcv

yolo_face = 'models/yolov8l-face.pt'
path_video = 'data/raw/for_test.mp4'

model = YOLO(yolo_face)

vidin = ffmpegcv.VideoCaptureNV(path_video)
vidout = ffmpegcv.VideoWriterNV('output.mp4', 'h264', vidin.fps)  #NVIDIA-GPU

count = 1
pbar = tqdm(total=30000)
with vidin, vidout:
    for frame in vidin:
        frame = cv2.resize(frame, (1280, 800), interpolation = cv2.INTER_AREA)
        
        result = model(frame, imgsz=1280, max_det=10, device=0,)
        
        for box in result:
            if box:
                data = box.boxes.xyxy[0]
                cv2.rectangle(frame, (int(data[0]), int(data[1])), (int(data[2]), int(data[3])), (0,0,255), 2)
            
        vidout.write(frame)
        count += 1
        pbar.update(1)
        
pbar.close()
vidin.release()
vidout.release()
cv2.destroyAllWindows()        