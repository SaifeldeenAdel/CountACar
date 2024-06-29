import cv2

from BoundarySetter import BoundarySetter
from CarCounter import CarCounter

path = "../video/carStocksEdit.mp4"

def main():

  cv2.namedWindow("CarCount")
  cv2.moveWindow("CarCount", 400,200)

  cap = cv2.VideoCapture(path)

  ## BOUNDARY SETTING
  img = None
  boundarySetter = BoundarySetter()

  ret, frame = cap.read()
  frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
  while not boundarySetter.isAllSet():
      if not ret:
        continue

      img = boundarySetter.update(frame.copy())

      cv2.imshow("CarCount", img)
      if cv2.waitKey(10) == 27:
        break

  boundary = boundarySetter.getBoundary()
  carThreshold = boundarySetter.getThreshold()

  carCounter = CarCounter(boundary, carThreshold)
  frmcount = 0

  ## DETECTION AND COUNTING
  while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))

    if frmcount % 5 == 0:
      img = carCounter.update(frame)
    frmcount += 1
    
    cv2.imshow("CarCount", img)
    key = cv2.waitKey(1) 

    if key == 32:
      continue
    if key == 27:
      break
    
  
  cv2.destroyAllWindows()
  
  


if __name__ == "__main__":
  main()