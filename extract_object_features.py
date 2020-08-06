import torchvision
from PIL import Image
import numpy as np
import torch
import os

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/activitynet_frames/"
features = {}
def getObjectFeature(self, input, output):
    global feature
    feature = output.data

def getPrediction(self, input, output):
    global scores
    scores = output.data

for video in os.listdir(folder):
    frame_num = len(os.listdir(f"{folder}{video}"))
    for i in range(frame_num):
        frame_name = '0' * (6-len(str(i + 1))) + str(i + 1) + '.jpg'
        image = Image.open(f'{folder}{video}/{frame_name}')
        img_np = np.asarray(image) / 255
        img_tensor = torch.FloatTensor(img_np)
        img_tensor = img_tensor.permute(2, 0, 1)

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.roi_heads.box_roi_pool.register_forward_hook(getObjectFeature)
        model.roi_heads.box_predictor.cls_score.register_forward_hook(getPrediction)
        model.eval()

        pred = model([img_tensor, img_tensor])
        pass