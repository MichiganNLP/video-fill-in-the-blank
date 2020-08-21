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
features = {}
THESHROLD = 16
box_score_thresh = 0.05
nms_thresh = 0.5
detections_per_img = 100

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

def ROIHeadsHook(self, input, output):
    global prop
    global img_shapes
    prop = input[1]
    img_shapes = input[2]
    

def ROIHeads_BoxPredictorHook(self, input, output):
    global cl
    global box_reg
    box_feature = input[0]
    cl, box_reg = output

def RPN_ClslogitsHook(self, input, output):
    pass

def postprocess_detections(class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
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

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
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
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

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
        model.roi_heads.register_forward_hook(ROIHeadsHook)
        model.roi_heads.box_predictor.register_forward_hook(ROIHeads_BoxPredictorHook)
        model.rpn.head(RPN_ClslogitsHook)
        model.eval()
        pred = model([img_tensor])
        # # select top THRESHOLD
        # _, idx = torch.topk(scores, THESHROLD)
        # feature = torch.index_select(feature, 0, idx)
        # # scores = torch.index_select(scores, 0, idx)        
        # bboxes = torch.zeros(THESHROLD, 4)
        # for i in range(len(idx)):
        #     bboxes[i, :] = bbox[idx[i], cl[idx[i]] * 4 : cl[idx[i]] * 4 + 4]
        # cl = torch.index_select(cl, 0, idx)

        all_boxes, all_scores, all_labels = postprocess_detections(cl, box_reg, prop, img_shapes)

        # features[video].append([feature, bboxes, cl])

with open("video_features.pkl", 'wb') as f:
    pickle.dump(features, f)