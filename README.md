# Lane Finding Project

In this project I build an algorithm to detect road lane by processing a video stream of a forward
facing camera mounted centeraly in a moving vehicle. The algorithm extracts information form each frame to find the **Peaks in a Histogram** which
represents the position of the lane lines. Window search is used to track the lines and fit the extracted points to line.

## Project files

[`calibrate.py`](calibrate.py) : gets the calibration matrix and store in pickle file

[`lane_finding.py`](lane_finding.py) : preprocesses the video's frames and implements lane detection algorithm   

## project pipeline

- **Camera Calibration** - Calibrate the camera to get rid of image distortions. Using chessboard images to calibrate the camera.
OpenCv library provides `cv2.calibrateCamera()` whcih extract the calibration matrix and distortion of the camera. Then we can save that
matrix and use it later.

- **Perspective Transform** - To detect the curvature of the lane we use "bird’s-eye view transform" which give us a top view of the lanes.
This way we can compute the Histogram of each image and extract the peaks.

- **Histogram Peaks** - We compute the histogram of the image and find its peaks which represents the base of the lane lines.
Then we apply a window search to extract the pixels of these lines.

- **lines fitting** - After extracting the pixels, we fit these pixels to the best line that go through the pixels. Then we can measure these
distance between the two line to find the center of the lane.

## Project output

The output folder contains some output image, while the output video can be found [here](https://www.youtube.com/watch?v=J40jRx7ykrI&fbclid=IwAR1Yyh-ISBQEXSza-M24DA-t4XGRHV4N02oUuSb3laz5j0YHlAa2ysYBlaM)

![alt text](output/project-video-output.gif)
