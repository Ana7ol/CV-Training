# Object detection and nonmaximum suppression
import torch
from torchvision.ops import box_iou, nms
boxes = torch.tensor([[0.,0.,10.,10.],
                      [1.,1.,11.,11.],
                      [20.,20.,30.,30.]])
scores = torch.tensor([.9, .8, .7])
print("IoU matrix:", box_iou(boxes, boxes))
keep = nms(boxes, scores, iou_threshold=.5)
print("Retained indices:", keep)  # [0, 2]
