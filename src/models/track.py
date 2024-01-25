#!/usr/bin/env python3

# sourcery skip: avoid-builtin-shadow
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

from boxmot import StrongSORT

annotation = []

tracker = StrongSORT(
    model_weights=Path("osnet_x0_25_msmt17.pt"),  # which ReID model to use
    device="cuda:0",
    fp16=False,
)

file_path = "data/raw/arr_embs_id_1.npy"
anno_path = "data/raw/camera30.avi.txt"

file_data = np.load(file_path, allow_pickle=True)
df = pd.DataFrame(file_data)

anno_data = pd.read_table(anno_path, header=None, delimiter=" ")
max_len = anno_data.iloc[-1, 0]

for i in tqdm(range(430, max_len)):
    x = []
    dets = anno_data.loc[anno_data.iloc[:, 0] == i]
    dets = dets.iloc[:, 1:7]
    embeddet = df.loc[df.iloc[:, 0] == i].values
    if (dets.shape[0] != 0) & (embeddet.shape[0] != 0) & (dets.shape[0] == embeddet.shape[0]):
        dets = dets.values
        dets = dets.astype(np.float32)
        for det in dets:
            id = int(det[0])
            emb_id = embeddet[embeddet[:, 1] == str(id)]
            x.append(emb_id[0][3])
        x = np.array(x)

        
        #embeddet = embeddet.astype(np.float32)
        #print(embeddet[0])
        tracks = tracker.update(dets, x)

        annotation.append(tracks)

pd.DataFrame(annotation).to_csv("test.txt", header=None, index=None, sep=" ")