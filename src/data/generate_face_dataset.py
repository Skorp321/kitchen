import math
import shutil
import ffmpegcv
import cv2
import numpy as np
from tqdm.auto import tqdm

import numpy as np
import cv2
import torch
import face_recognition
from PIL import Image
from deepface import DeepFace

path_to_video = 'data/raw/for_test.mp4'

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

vidin = ffmpegcv.VideoCaptureNV(path_to_video)

count = 1

for frame in tqdm(vidin):
    
    frame = cv2.resize(frame, (1280, 800))
    
    face_objs = DeepFace.extract_faces(img_path = frame, 
        target_size = (224, 224), 
        detector_backend = backends[6],
        enforce_detection = False
    )
    
    if (face_objs[0]['facial_area']['left_eye'] is not None) | (face_objs[0]['facial_area']['right_eye'] is not None):
        for face in face_objs:
            print(face['face'])
            '''image = Image.fromarray(face['face'])
            print(image)
            image.save(f'output_image_{count}.jpg')'''
            