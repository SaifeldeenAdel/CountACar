import cv2
import numpy as np
import copy

class BoundarySetter:
  def __init__(self):
    self.boundary = None
    self.p1, self.p2 = None, None
    self.carThreshold = None
    self.isCarThresholdSet = False
    self.og = None

    # Setting the mouse callback to track mouse movement
    cv2.setMouseCallback("CarCount", self.mouseCallback)

  def isAllSet(self) -> bool:
    return self.getBoundary() and self.isCarThresholdSet
  
  def getBoundary(self) -> tuple:
    return self.boundary

  def getThreshold(self) -> tuple:
    return self.carThreshold

  def update(self,frame) -> np.array:
    self.modified = frame
    if self.p1 and self.p2:
      cv2.rectangle(self.modified, self.p1, self.p2, (0,255,0), 1,cv2.LINE_AA)
      
      if self.carThreshold:
        cv2.line(self.modified, self.carThreshold[0], self.carThreshold[1], (0,0,255), 1,cv2.LINE_AA)

    return self.modified

  
  def mouseCallback(self, event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
      if not self.p1:
        self.p1 = (x,y)
      if self.boundary and not self.isCarThresholdSet:
        self.isCarThresholdSet = True

    if event == cv2.EVENT_LBUTTONUP:
      self.boundary = (self.p1, self.p2)

    if event == cv2.EVENT_MOUSEMOVE:
      if self.p1 and not self.boundary:
        self.p2 = (x,y)
      elif self.boundary and not self.isCarThresholdSet:
        if y > self.p2[1]:
          y = self.p2[1]
        elif y < self.p1[1]:
          y = self.p1[1]

        self.carThreshold = ((self.p1[0], y), (self.p2[0], y))
