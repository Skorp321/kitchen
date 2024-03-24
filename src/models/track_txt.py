#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torchreid
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from sklearn.neighbors import KNeighborsClassifier
from joblib import load

def box2xyxy(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    
    return x1, y1, x2, y2 

KNeighbors_model = load('models/KNeighbors.joblib')

file_path = "data/raw/arr_embs_id_1.npy"
anno_path = "data/raw/camera30.avi.txt"

file_data = np.load(file_path, allow_pickle=True)
df = pd.DataFrame(file_data)

anno_data = pd.read_table(anno_path, header=None, delimiter=" ")
max_len = anno_data.iloc[-1, 0]

def int_num(x):

    return int(x.split('_')[-1])

df.iloc[:, 2] = list(map(int_num, df.iloc[:, 2]))

with open('data/output.txt', 'w') as file:
    
    for i in tqdm(range(430, max_len)):
        x = []
        dets = anno_data.loc[anno_data.iloc[:, 0] == i]
        dets = dets.iloc[:, 2:6]
        embeddet = pd.DataFrame(df.loc[df.iloc[:, 0] == i].values)
        if len(embeddet) == 0:
                continue
        classes = KNeighbors_model.predict(list(embeddet.iloc[:, -1]))

        for result, id in zip(dets.values, classes):
            
            if len(result) == 0:
                continue

            x1, y1, x2, y2 = box2xyxy(result)
            file.write(f'{i} {id} {x1} {y1} {x2} {y2}\n')