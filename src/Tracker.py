import torchvision
import cv2

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