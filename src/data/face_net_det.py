import math
import ffmpegcv
import cv2
import numpy as np
from tqdm.auto import tqdm

from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from IPython import display

path_to_video = '/media/skorp321/datasets/datasets/camera30.avi'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on device: {device}')

mtcnn = MTCNN(
    margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

vidin = ffmpegcv.VideoCaptureNV(path_to_video)
vidout = ffmpegcv.VideoWriterNV('output.mp4', 'h264', vidin.fps)  #NVIDIA-GPU

count = 1

for frame in tqdm(vidin):
    frame = cv2.resize(frame, (1280, 800))
    
    img_copy = frame.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)    
    img_copy = Image.fromarray(img_copy)
    boxes, _ = mtcnn.detect(img_copy)
    
    if boxes is not None:
        for box in boxes:
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    #frame = np.array(frame)
    vidout.write(frame)
    if count > 30000:
        break
    count += 1

vidin.release()
vidout.release()

