import torchvision
from PIL import Image
import numpy as np
import torch
import os

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/activitynet_frames/"

def getObjectFeature(self, input, output):
    print(output.data)

image = Image.open(f'{folder}v_qkN9uA8izVE/000001.jpg')
img_np = np.asarray(image) / 255
img_tensor = torch.FloatTensor(img_np)
img_tensor = img_tensor.permute(2, 0, 1)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

pred = model([img_tensor])