#!/usr/bin/env python3

from pathlib import Path
import cv2
import pandas as pd
from ultralytics import YOLO
import torchreid
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
from torchreid.reid.utils import FeatureExtractor

def box2xyxy(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    
    return (x1, y1), (x2, y2) 

video_path = "/media/skorp321/datasets/datasets/camera30.avi"
model_weights = Path("models/osnet_ain_x1_0_citchen.pt")

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

new_width = 800
scale = new_width / width
new_height = int(height * scale)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

model = YOLO('yolov8l.pt') 

'''model_reid = ReIDDetectMultiBackend(
            weights=model_weights,
            device="cuda"
            )'''

extractor = FeatureExtractor(
    model_name='osnet_ain_x1_0',
    model_path='models/model.pth.tar-5',
    device='cuda'
)

out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc( * 'XVID'), fps, (new_width, new_height))

KNeighbors_model = load('models/KNeighbors.joblib')
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    color = (0,0,255)
        
    lw = line_thickness or max(round((sum(frame).shape) / 2 * 0.003), 2)

    # Здесь ваш код обработки изображения
    results = model.predict(frame, classes=0, imgsz=1280, device=0, max_det=10)

    for result in results:

        cv2.putText(frame, 'frame: %d fps: %.2f num: %d' % (count, fps, len(result.boxes.data)),
                (0, int(15 * (text_scale))), cv2.FONT_HERSHEY_PLAIN, 2, color, thickness=2)
        
        if result.boxes.xywh.shape[0] == 0:
            continue
        
        frames = []
        for box in result.boxes.data:
            x1y1, x2y2 = box2xyxy(box)  
            frames.append(frame[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]])
        
        features = extractor(frames)
        features = features.cpu().numpy()
        classes = KNeighbors_model.predict(features)
        probabilities = KNeighbors_model.predict_proba(features)

        for box, id in zip(result.boxes.data, classes):

            x1y1, x2y2 = box2xyxy(box)            
            '''            
            tf = max(lw - 1, 1) 
            sf = lw / 3  
            
            w, h = cv2.getTextSize(str(id), 0, fontScale=sf, thickness=tf)[0]
            outside = y1 - h >= 3
            p2 = x1 + w, y1 - h - 3 if outside else y1 + h + 3           
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=lw, lineType=cv2.LINE_AA)           
            cv2.rectangle(frame, (x1, y1), p2, color, -1, cv2.LINE_AA)
            cv2.putText(frame,
                            str(label), (x1, y1 - 2 if outside else y1 + h + 2),
                            0,
                            sf,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)'''
                        
            cv2.rectangle(frame, x1y1, x2y2, (0,0,255), thickness=5)
            cv2.putText(frame, str(id), x1y1, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5, cv2.LINE_AA)

    frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow('Video', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count > 3000:
        break
out.release()
cap.release()
cv2.destroyAllWindows()