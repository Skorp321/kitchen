from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend
from boxmot.utils.matching import NearestNeighborDistanceMetric
import pandas as pd
import numpy as np
import cv2
import numpy as np
from pathlib import Path
import torch

from boxmot import StrongSORT
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.motion.cmc import get_cmc_method


def xywh2tlwh(x):
    """
    Convert bounding box coordinates from (x c, y c, w, h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height...

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # xc --> t
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # yc --> l
    y[..., 2] = x[..., 2]  # width
    y[..., 3] = x[..., 3]  # height
    return y


class CastomStrongSORT(StrongSORT):
    def __init__(self,
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_dist=0.7,
                 max_age=70,
                 max_unmatched_preds=7,
                 n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):
        super(StrongSORT, self).__init__(model_weights, device, fp16)
        
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_dist=max_iou_dist, max_age=max_age, n_init=n_init, max_unmatched_preds=max_unmatched_preds, mc_lambda=mc_lambda, ema_alpha=ema_alpha)


    def update(self, dets, embeddet):
        # dets = dets.data
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]

        tlwh = xywh2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(
                tlwh, confs, clss, det_ind, embeddet
            )
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(
                    ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                ).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])