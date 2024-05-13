import cv2


class BoundarySetter:
  def __init__(self, path):
    self.boundary = None
    self.carThreshold = None
    self.p1, self.p2 = None, None

    # Reading first frame and setting the mouse callback to track mouse movement
    self.cap = cv2.VideoCapture(path)
    _, self.frame = self.cap.read()
    cv2.namedWindow("CarCount")
    cv2.setMouseCallback("CarCount", self.mouseCallback)

    # Method that does all the boundary logic
    self.setBoundary(self.frame)

  def setBoundary(self, frame):
    modified = cv2.resize(self.frame, (self.frame.shape[1]//2, self.frame.shape[0]//2))

    while not self.boundary and  not self.carThreshold:
      if self.p1 and self.p2:
        # Draw rectangle and clear the old ones
        modified = cv2.rectangle(modified, self.p1, self.p2, (0,255,0), 1,cv2.LINE_AA)
        

      cv2.imshow("CarCount", modified)
      cv2.moveWindow("CarCount", 0,0)
      if cv2.waitKey(5) == 27:
        break

    cv2.destroyAllWindows()
  
  def mouseCallback(self, event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDBLCLK:
      self.p1 = (x,y)
    if event == cv2.EVENT_MOUSEMOVE:
      if self.p1:
        self.p2 = (x,y)
