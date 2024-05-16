import cv2

from BoundarySetter import BoundarySetter
from CarCounter import CarCounter

path = "../video/cars2.MOV"

def main():

  cv2.namedWindow("CarCount")
  cv2.moveWindow("CarCount", 400,200)

  cap = cv2.VideoCapture(path)

  # Run boundary setting
  img = None
  boundarySetter = BoundarySetter()
  
  while not boundarySetter.isAllSet():
      _, frame = cap.read()
      frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
      img = boundarySetter.update(frame)

      cv2.imshow("CarCount", img)
      if cv2.waitKey(10) == 27:
        break


  while True:
      _, frame = cap.read()
      frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
      boundary = boundarySetter.getBoundary()
      carThreshold = boundarySetter.getThreshold()

      # frame = frame[p1[1]:p2[1],p1[0]:p2[0]]
      # img = CarCounter.update(frame, boundary, threshold)

      cv2.imshow("CarCount", img)
      if cv2.waitKey(10) == 27:
        break
  
  cv2.destroyAllWindows()
  
  
  carThreshold = boundarySetter.getThreshold()
  
  
  ## Run detection


if __name__ == "__main__":
  main()