# Count A Car

A from-scratch Object Tracking and Counting from scratch implementation.

Tracking and counting cars on a highway with an OpenCV graphical interface that allows users to choose a specific region to track and count.

![](./demo.gif)

## How it's done?

1. **Boundary and Threshold setting**

You're first presented with the first frame of the video you're working on. Drag to select a region where you'd like the detection and tracking to run then move vertically to choose a threshold to count cars from if crossed.

2. **Object Tracking**

By utilising one of Detectron2's object detection models, we are able to detect all cars in the chosen region and by utilizing IOU calculations between frames and assigning unique IDs to each matching car from the previous frame, we are able to build an object tracker that works quite well given that there isn't much occlusion of the tracked object.

3. **Object Counting**

Using the threshold set by the user, we can identify whether a car has crossed the threshold by carrying out a simple calculation using the detection midpoints in the current and previous frames.

## To Run

If it's all setup, simply run `main.py` to start up the interface. 

Unfortunately, this was more of a personal project, getting it up and running isn't that hard however it will need some tinkering

-   Editing the video path in `main.py` and placing your own video path

-   Figuring out the right resize ratio for your video frame (I was working with large videos so I had to resize to 1/4th the original size)

-   Based on your architecture, detection/tracking might be very slow.
