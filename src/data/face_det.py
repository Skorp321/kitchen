import math
import ffmpegcv
import cv2
import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO
import face_recognition
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


path_to_video = '/media/skorp321/datasets/datasets/camera30.avi'
model_path = 'models/detector.tflite'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

vidin = ffmpegcv.VideoCapture(path_to_video)
vidout = ffmpegcv.VideoWriterNV('output.mp4', 'h264', vidin.fps)  #NVIDIA-GPU

count = 1
pbar = tqdm(total=3000)
with vidin, vidout:
    
    for frame in vidin:
        
        detection_result = detector.detect(frame)
        
        image_copy = np.copy(frame.numpy_view())
        
        #frame = np.array(frame)
        frame = cv2.resize(frame, (1280, 800), interpolation = cv2.INTER_AREA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        #face_locations = face_recognition.face_locations(frame)
        predictions = facedetection.detect(mp_image)
        print(predictions.detections)
        if predictions.detections:
            for detection in predictions.detections:
                bboxC = detection.bounding_box
                #bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                #frame = cv2.Mat(frame)
                x, y, w, h = int(bboxC.origin_x), int(bboxC.origin_y), int(bboxC.width), int(bboxC.height)
                print(f'{x}, {y}, {w}, {h}')
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if count > 3000:
            break
        vidout.write(frame)
        count += 1
        pbar.update(1)
print(count)
pbar.close()
vidin.release()
vidout.release()
cv2.destroyAllWindows()