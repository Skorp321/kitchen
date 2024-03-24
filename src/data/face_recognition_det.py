import math
import ffmpegcv
import cv2
import numpy as np
from tqdm.auto import tqdm

import numpy as np
import cv2
import torch
import face_recognition
from PIL import Image
import dlib

dlib.DLIB_USE_CUDA = True
print(f'Num devices: {dlib.DLIB_USE_CUDA}')

path_to_video = '/media/skorp321/datasets/datasets/camera30.avi'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

vidin = ffmpegcv.VideoCaptureNV(path_to_video)
vidout = ffmpegcv.VideoWriterNV('output_fr.mp4', 'h264', vidin.fps)  #NVIDIA-GPU

detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')

count = 1

for frame in tqdm(vidin):
    frame = cv2.resize(frame, (1280, 800))

    img_copy = frame.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    #img_copy = Image.fromarray(img_copy)

    #face_locations = face_recognition.face_locations(frame, model="cnn")
    face_locations = detector(img_copy)
    
    #print(f"I found {len(face_locations)} face(s) in this photograph.")
    #print(face_locations)
    if len(face_locations) > 0:
        box = face_locations[0]
        frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    #cv2.imshow('frame', frame)
    
    vidout.write(frame)
    if count > 3000:
        break
    count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidin.release()
vidout.release()