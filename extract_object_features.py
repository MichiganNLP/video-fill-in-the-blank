import torchvision
from PIL import Image
import numpy as np
import torch
import os
import pickle

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/activitynet_frames/"
features = {}
THESHROLD = 10
def getObjectFeature(self, input, output):
    global feature
    feature = output.data

def getPrediction(self, input, output):
    global scores
    global cl
    scores, cl = torch.max(output.data, axis=1)

def getBBox(self, input, output):
    global bbox
    bbox = output.data

for video in os.listdir(folder):
    frame_num = len(os.listdir(f"{folder}{video}"))
    features[video] = []
    for i in range(frame_num):
        frame_name = '0' * (6-len(str(i + 1))) + str(i + 1) + '.jpg'
        image = Image.open(f'{folder}{video}/{frame_name}')
        img_np = np.asarray(image) / 255
        img_tensor = torch.FloatTensor(img_np)
        img_tensor = img_tensor.permute(2, 0, 1)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.roi_heads.box_roi_pool.register_forward_hook(getObjectFeature)
        model.roi_heads.box_predictor.cls_score.register_forward_hook(getPrediction)
        model.roi_heads.box_predictor.bbox_pred.register_forward_hook(getBBox)
        model.eval()
        pred = model([img_tensor])
        # select top THRESHOLD
        _, idx = torch.topk(scores, THESHROLD)
        feature = torch.index_select(feature, 0, idx)
        # scores = torch.index_select(scores, 0, idx)
        cl = torch.index_select(cl, 0, idx)
        bboxes = torch.zeros(THESHROLD, 4)
        for i in range(len(idx)):
            bboxes[i, :] = bbox[4 * idx[i] : 4 * idx[i] + 4]


        features[video].append([feature, bboxes, cl])

with open("video_features.pkl", 'wb') as f:
    pickle.dump(features, f)