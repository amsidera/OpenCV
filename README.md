# OpenCV


Projects using OpenCV for the understanding of algotihms and techniques used in computer vision.

1. OpenCV: The program should load an image by either reading it from a file or capturing it directly
from a camera. When the user presses a key perform the operation corresponding to the
key on the original image.


2. Camera Calibration: implementation of a camera calibration algorithm, estimating the camera matrix 
parameters under the assumption of noisy data. Successive I use a Ransac algorithm to eliminate the outliers, 
therefore try to obtain a robust estimation.


3. Corner Detection: the detection of corners and features in two different images and match the same points 
in the images.


4. Epipolar lines estimation: calculate the fundamental matrix, the epipole and the epipolar lines.


5. Spatial Pyramid Matching for recognizing images: research project using [Beyond Bags of Features: Spatial Pyramid Matching
for Recognizing Natural Scene Categories]. In this paper the authors present a method for recognizing scene 
categories based on approximate global geometric correspondence. The main algorithm for this is made a partition
in the image, decreasing in each step the region.

[Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories]: <http://ieeexplore.ieee.org/document/1641019/>
