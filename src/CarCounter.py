import cv2
from src.Detector import Detector
from src.Tracker import Tracker

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

