import torchvision
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torch
import os
import pickle
from object_detect_utils import BoxCoder

folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/5fps_Videos/"
output_folder = "/scratch/mihalcea_root/mihalcea1/shared_data/ActivityNet_Captions/latest_data/multimodal_model/object_detection_features/"

# Hyperparams copied from torchvision faster rcnn model default config
box_score_thresh = 0.05
nms_thresh = 0.5
detections_per_img = 100

def ROIHeadsHook(self, input, output):
    global prop
    global img_shapes
    prop = input[1]
    img_shapes = input[2]

def ROIHeads_BoxPredictorHook(self, input, output):
    global cl
    global box_reg
    cl, box_reg = output

def RPN_ClslogitsHook(self, input, output):
    global box_features
    # This layer contains two linear layers
    # We may want to train new linear layers so we used the input of this layer
    box_features = input[0]
    

def postprocess_detections(class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes,     # type: List[Tuple[int, int]]
                               box_features,     # type: Tensor, size: N * C * H * W (N is total box numbers in all images in one batch)
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        
        bbox_reg_weights = (10., 10., 5., 5.)
        box_coder = BoxCoder(bbox_reg_weights)

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        box_feature_list = box_features.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_box_features = []
        for boxes, scores, image_shape, box_feat in zip(pred_boxes_list, pred_scores_list, image_shapes, box_feature_list):
            # boxes : N * num_classes * 4
            # scores : N * nums_classes
            # box_feat: N * C * H * W
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > box_score_thresh).squeeze(1)
            boxes, scores, labels, box_feat = boxes[inds], scores[inds], labels[inds], box_feat[inds//(num_classes - 1)]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, box_feat = boxes[keep], scores[keep], labels[keep], box_feat[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:detections_per_img]
            boxes, scores, labels, box_feat = boxes[keep], scores[keep], labels[keep], box_feat[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_box_features.append(box_feat)

        return all_boxes, all_scores, all_labels, all_box_features

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
if torch.cuda.is_available:
    model = model.cuda()
model.roi_heads.register_forward_hook(ROIHeadsHook)
model.roi_heads.box_predictor.register_forward_hook(ROIHeads_BoxPredictorHook)
model.roi_heads.box_head.register_forward_hook(RPN_ClslogitsHook)
model.eval()

# output layer structure of faster-rcnn:
# (roi_heads): RoIHeads(
#     (box_roi_pool): MultiScaleRoIAlign()
#     (box_head): TwoMLPHead(
#       (fc6): Linear(in_features=12544, out_features=1024, bias=True)
#       (fc7): Linear(in_features=1024, out_features=1024, bias=True)
#     )
#     (box_predictor): FastRCNNPredictor(
#       (cls_score): Linear(in_features=1024, out_features=91, bias=True)
#       (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
#     )
#   )

for video in os.listdir(folder):
    if (os.path.exists(f"{output_folder}{video}.pkl")):
        continue
    print(video)
    frame_num = len(os.listdir(f"{folder}{video}"))
    features = []
    with torch.no_grad():
        for i in range(frame_num):
            # Model input is a list of images, here we input images one at each time
            image_list = []
            frame_name = '0' * (6-len(str(i + 1))) + str(i + 1) + '.jpg'
            with Image.open(f'{folder}{video}/{frame_name}') as image:
                img_np = np.asarray(image) / 255
            img_tensor = torch.FloatTensor(img_np)
            if torch.cuda.is_available:
                img_tensor = img_tensor.cuda()
            img_tensor = img_tensor.permute(2, 0, 1)        
            image_list.append(img_tensor)
            
            pred = model(image_list)
            all_boxes = pred[0]['boxes']
            all_boxes[:, [0, 2]] /= img_tensor.shape[2]
            all_boxes[:, [1, 3]] /= img_tensor.shape[1]
            all_boxes = [all_boxes]
            del image_list
            # All outputs are lists, one element corresponds to one image 
            _, all_scores, all_labels, all_box_features = postprocess_detections(cl, box_reg, prop, img_shapes, box_features)

            features.append([all_boxes, all_box_features, all_scores, all_labels])

    with open(f"{output_folder}{video}.pkl", 'wb') as f:
        pickle.dump(features, f)