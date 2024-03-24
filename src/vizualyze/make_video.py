import numpy as np
import pandas as pd
import cv2

num = 0
video_path = "/media/skorp321/datasets/datasets/camera30.avi"
anno_path = "data/output.txt"

def box2xyxy(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(x1 + box[2])
    y2 = int(y1 + box[3])
    
    return (x1, y1), (x2, y2) 

cap = cv2.VideoCapture(video_path)

new_width = 640
new_height = 480

anno_data = pd.read_table(anno_path, header=None, delimiter=" ")
max_len = anno_data.iloc[-1, 0]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    anno = anno_data.loc[anno_data.iloc[:, 0]==num]
    if len(anno) != 0:
        
        for box, id in zip(anno.iloc[:, 2:].values, anno.iloc[:, 1].values):
            x1y1, x2y2 = box2xyxy(box)
            #x1y1, x2y2 = box[:2], box[2:]
            cv2.rectangle(frame, x1y1, x2y2, (0,0,255), thickness=2)
            cv2.putText(frame, str(id), x1y1, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 1, cv2.LINE_AA)
            
    frame = cv2.resize(frame, (new_width, new_height))
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    num += 1
cap.release()
cv2.destroyAllWindows()