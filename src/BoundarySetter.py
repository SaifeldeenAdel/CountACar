import cv2
import copy

class BoundarySetter:
  def __init__(self, path):
    self.boundary = None
    self.p1, self.p2 = None, None

    self.carThreshold = None
    self.isCarThresholdSet = False
    
    self.modified = None

    # Reading first frame and setting the mouse callback to track mouse movement
    self.cap = cv2.VideoCapture(path)
    _, self.frame = self.cap.read()
    self.og = cv2.resize(self.frame, (self.frame.shape[1]//2, self.frame.shape[0]//2))
    cv2.namedWindow("CarCount")
    cv2.setMouseCallback("CarCount", self.mouseCallback)

    # Method that does all the boundary logic
    self.setBoundary(self.frame)

  def setBoundary(self, frame):
    self.modified = cv2.resize(self.frame, (self.frame.shape[1]//2, self.frame.shape[0]//2))

    while not self.boundary or not self.isCarThresholdSet:
      if self.p1 and self.p2:
        # p1 and p2 updated in mouse callback
        cv2.rectangle(self.modified, self.p1, self.p2, (0,255,0), 1,cv2.LINE_AA)
        
        if self.carThreshold:
          cv2.line(self.modified, self.carThreshold[0], self.carThreshold[1], (0,0,255), 1,cv2.LINE_AA)

      cv2.imshow("CarCount", self.modified)
      cv2.moveWindow("CarCount", 0,0)
      if cv2.waitKey(5) == 27:
        break
      # print(not self.boundary and not self.isCarThresholdSet)

    cv2.destroyAllWindows()
  
  def mouseCallback(self, event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
      if not self.p1:
        self.p1 = (x,y)

    if event == cv2.EVENT_LBUTTONUP:
      self.boundary = (self.p1, self.p2)
      self.carThreshold = (())

    if event == cv2.EVENT_MOUSEMOVE:
      self.modified = self.og.copy()
      if self.p1 and not self.boundary:
        self.p2 = (x,y)
      elif self.boundary and not self.isCarThresholdSet:
        if x > self.p2[0]:
          x = self.p2[0]
        elif x < self.p1[0]:
          x = self.p1[0]

        self.carThreshold = ((x, self.p1[1]), (x,self.p2[1]))
