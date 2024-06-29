import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import math
from random import randint
import torchvision

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

class Tracker:
  def __init__(self, boundary, threshold):
    self.prevBoxes = []
    self.currBoxes = []
    self.threshold = threshold
    self.boundary = boundary
    self.trackID = 0
    self.initializedIDs = 0
    self.trackingObjects = {}

  def setBoxes(self, boxes):
    boxes = [self.adjustToBoundary(box) for box in boxes]
    self.prevBoxes = self.currBoxes.copy()
    self.currBoxes = boxes
    
  def adjustToBoundary(self,box):
    boundaryX, boundaryY = self.boundary[0]
    box = (box[0] + boundaryX, box[1] + boundaryY, box[2] + boundaryX, box[3] + boundaryY)
    return box


  
  def calculate_iou(self, box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection_width = max(0, x2_min - x1_max)
    intersection_height = max(0, y2_min - y1_max)
    intersection_area = intersection_width * intersection_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area != 0 else 0

  def track(self, boxes):
    self.setBoxes(boxes)
    self.initializedIDs += 1

    if self.initializedIDs == 2:
      for curr in self.currBoxes:
        for prev in self.prevBoxes:
          iou = self.calculate_iou(curr,prev)
          if iou > 0.3:
            self.trackingObjects[self.trackID] = curr
            self.trackID += 1
      
    else:

      for id, box in self.trackingObjects.copy().items():
        exists = False
        for curr in self.currBoxes.copy():
          iou = self.calculate_iou(curr,box)
          print(iou)
          if iou > 0.3:
            self.trackingObjects[id] = curr
            if curr in self.currBoxes:
              self.currBoxes.remove(curr)
            exists = True
            continue
        
        if not exists:
          self.trackingObjects.pop(id)
      
      for box in self.currBoxes:
        self.trackingObjects[self.trackID] = box
        self.trackID += 1

    return self.trackingObjects
    
  def visualize(self, frame, trackingObjects):

    for id, box in trackingObjects.items():
        midpoint = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.circle(frame, midpoint, 3, (0, 255, 255), -1)
        cv2.putText(frame, str(id), (midpoint[0] + 2, midpoint[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame
  

class CarCounter:
  def __init__(self, boundary, threshold):
    self.boundary = boundary
    self.threshold = threshold
    self.carCount = 0
    self.count = 0

    self.detector = Detector()
    self.tracker = Tracker(self.boundary, self.threshold)

    self.prevMidPoints = {}
    self.currMidPoints = {}
    

  def drawBoundaryAndThreshold(self, frame):
    cv2.rectangle(frame, self.boundary[0], self.boundary[1], (0,255,0), 1, cv2.LINE_AA)
    cv2.line(frame, self.threshold[0], self.threshold[1], (0,0,255), 1,cv2.LINE_AA)


  def update(self,frame):
    self.drawBoundaryAndThreshold(frame)

    roi = frame[self.boundary[0][1]:self.boundary[1][1], self.boundary[0][0]:self.boundary[1][0]]

    vehiclesBB = self.detector.detectBoxes(roi)
    trackingObjects = self.tracker.track(vehiclesBB)
    frame = self.tracker.visualize(frame, trackingObjects)
    self.updateCount(trackingObjects)
    frame = self.writeCount(frame)


    return self.writeCount(frame)


  def setMidPoints(self, trackingObjects):
    self.prevMidPoints = self.currMidPoints.copy()
    self.currMidPoints = {}

    for id, box in trackingObjects.items():
        midpoint = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        self.currMidPoints[id] = midpoint

  def writeCount(self, frame):
    frame = cv2.rectangle(frame, (37, 216), (230, 260), (250,0,255), -1)
    text = f"Car Count: {self.count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)
    thickness = 2
    text_x = 45
    text_y = 250
    return cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

  def updateCount(self, trackingObjects):
    self.setMidPoints(trackingObjects)
    

    for id, currMidpoint in self.currMidPoints.items():
        if id in self.prevMidPoints:
            prevMidpoint = self.prevMidPoints[id]
            if (prevMidpoint[1] <= self.threshold[1][1] and currMidpoint[1] > self.threshold[1][1]) or (prevMidpoint[1] >= self.threshold[1][1] and currMidpoint[1] < self.threshold[1][1]):
                print("here")
                self.count += 1

