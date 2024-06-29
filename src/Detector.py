
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torchvision
import numpy as np

class Detector:
  def __init__(self):
    self.cfg = get_cfg()
    print(self.cfg.MODEL.DEVICE)
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    self.predictor = DefaultPredictor(self.cfg)

  def detectBoxes(self, img):
    predictions = self.predictor(img)

    instances = predictions["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor
    classes = instances.pred_classes
    scores = instances.scores

    keep_indices = [i for i, cls in enumerate(classes) if cls in [2,3,7]]
    filtered_boxes = boxes[keep_indices]
    filtered_scores = scores[keep_indices]

    keep = torchvision.ops.nms(filtered_boxes, filtered_scores, 0.6)
    
    return filtered_boxes[keep].numpy().astype(np.int32).tolist()