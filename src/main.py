import cv2

from BoundarySetter import BoundarySetter
from CarCounter import CarCounter

def main():
  path = "../video/cars.MOV"

  # Run boundary setting
  boundary, carThreshold = BoundarySetter(path) 
  
  ## Run detection


if __name__ == "__main__":
  main()