## Writeup 

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image0]: ./camera_cal/corners_found1.jpg "Found Corners for Calibration"
[image1]: ./output_images/Undistorted.png "Undistorted"
[image2a]: ./output_images/Undistorted_Road.png "Road Undistorted"
[image2b]: ./output_images/Undistorted_Bridge.png "Bridge Undistorted"
[image3]: ./output_images/thresholded_binary.png "Binary edge detection"
[image4a]: ./output_images/Warped_Lanelines.png "Warp lane lines"
[image4b]: ./output_images/Warped_Chessboard.png "Warp chessboard"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image5a]: ./output_images/polyfit1.png "polyfit1"
[image5b]: ./output_images/polyfit2.png "polyfit2"
[image5c]: ./output_images/polyfit1_fit.png "polyfit1_fit"
[image5d]: ./output_images/polyfit2_fit.png "polyfit2_fit"
[image6]: ./output_images/pipeline_output.png "Output"
[image7a]: ./output_images/polyfit_mytest4.jpg "fail1"
[image7b]: ./output_images/polyfit_mytest5.jpg "fail2"
[image7c]: ./output_images/polyfit_test5.jpg "fail3"
[image7d]: ./output_images/polyfit_test1.jpg "fail4"

[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  The python notebook file `./Advanced_Lane_Finding.ipynb` contains all the code for this project.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the jupyter notebook located in `./Advanced_Lane_Finding.ipynb`. Since this is the preparation for the following work and only executes once before the pipeline, this part is named `0. Camera Calibration`.  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  `imgpoints` are the detected corners of the chessboard image (9x6 in the below picture), and can be obtained by calling `ret, corners = cv2.findChessboardCorners(gray, (9,6), None)` .

![alt text][image0]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to test image `./test_images/test1.jpg`. The undistortion is implemented by calling `undst = cv2.undistort(img, mtx, dist, None, mtx)` function with the camera correction matrix obtained in the previous step. We can observe there is very minor difference before and after undistortion of this picture. 
![alt text][image2a]

However, for pictures with long straight lines, we can observe significant difference, e.g., the picture below with a wide bridge. It is clear that the undistorted picture has a straight bridge while the bridge in the original picture is bended.
![alt text][image2b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image and locate all the edges.  The steps are described in the function named `edgefinding()` in section `2. Color/Gradient Threshold`. After trying RGB and HLS color spaces and both gradient and thresholding methods, two edge detection algorithms are used to get the best results. The first one is based on the horizontal (x direction) gradient using `r channel` data and `cv2.Sobel()` function. The gradient threshold is (20, 100). The second one is based on `s channel` values and a simple thresholding.  The threshold is (170, 255). Here is an example of my output for this step using the test image.  Green is detected by sobel_x and blue is detected by s channel thresholding. By taking the advantages of the two detection results, the combined image on the right detects clear lane lines and objects like trees and cars, which can be easily removed by applying a detection window later.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform starts with generating a warp/unwarp matrix. This matrix is obtained by using a picture with straight lane lines and matching them with assigned positions assuming looking down vertically. In that case, the straight lane lines in the pictures correspond to a square after the perspective transformation. I selected four corner points along the lane line in picture `./test_images/straight_lines1.jpg` and matching them with assigned location in the table below (image size is 1280x720 pixels).  Then the perspective transform matrix and its inverse can be easily obtained using `M = cv2.getPerspectiveTransform(src, dst)` and `Minv = cv2.getPerspectiveTransform(dst, src)`.  The related code is in the ipython notebook function named `warp()`.

This source and destination points are selected below using: 
```python
src = np.float32([[294,659], [1012,659], [683,449], [597,449]])
dst = np.float32([[400, 720], [880, 720], [880, 0], [400, 0]])
```

|  Source  | Destination |
| :------: | :---------: |
| 294,659  |  400, 720   |
| 1012,659 |  880, 720   |
| 683,449  |   880, 0    |
| 597,449  |   400, 0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4a]

I also repeated the same procedure to do perspective transform for a chessboard image. The result is below.

![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the class, it is demonstrated the lane lines can be expressed using 2nd order polynomial like this:

![alt text][image5]

In order to fit the polynomial, the first step is to identify the active pixels that belong to left/right lane lines. In my code, there are two approaches to find those pixels. The first approach (`find_lane_pixels()`) uses histogram and searches from the bottom of the image using a sliding window. The two peak positions that include most pixels are identified as the start of the left and right lane lines. Then the window is gradually moved up and keep tracking the position that contains most of the active pixels. The other approach (`search_around_poly()`) still searches for the cluster of active points but it is based on the fitted curve position from previous frame, instead of from the bottom of the graph and moving up. The later one is simpler and also more robust to noisy pixels.

| using sliding window |  using previous fit  |
| :------------------: | :------------------: |
| ![alt text][image5a] | ![alt text][image5b] |

After identifying the pixels that belong to lane lines, the polynomial generation is straight forward by calling the function `left_fit = np.polyfit(lefty, leftx, 2)` and `right_fit = np.polyfit(righty, rightx, 2)`. This part is implemented in the python notebook function named `fit_polynomial()`.  The fitted curves are highlighted in yellow below. 

| curve fitting using sliding window | curve fitting using previous fit|
| :--------------------------------: | :------------------------------:|
|        ![alt text][image5c]        |       ![alt text][image5d]      |

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature can be easily calculated using the equations given in the course note and the fitted curve obtained from the previous step. The value needs to be converted from pixel to the real length. This part is implemented in function `measure_curvature_real()`.  The vehicle offset can be calculated by function `find_lane_base_real()`. It compares the center of the vehicle to the center of the lane lines, generating the vehicle offset.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `unwarp()` which transform the fitted curve from the top-down view to the camera view. 

All the previous steps are sequenced as a pipeline in the final function `draw_lane_lines()`. It takes an image as an input and output the image with lane area highlighted. Notes are also printed on top of the image to display curvature and vehicle offset in real time. Here is an example of my result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The developed pipeline worked pretty well for the test images and project video. There were some little glitches when there is a shade or road color change, but the impact is insignificant and considered acceptable. 
2. The current pipeline did not work quite well with the challenge videos. For some frames, the lane detection failed for at least one of the lane lines. I have picked some difficult images as my test set, and name them as `mytest*.jpg` in the test image folder. Below I listed some failure cases and reasons. It can be observed it is very critical to have a binary graph with few noise points. For the case there are areas of clustered noise pixels, the detection could be misled. It helps to search based on the fitted polynomial in previous frames, as it provides a baseline to start with. Nevertheless, a clean binary picture is important.
3. Another method suggested by the project guideline is to store the lane detection information history in a data structure and use it as a filter to ensure the wrong detection can be ruled out. In this project, I also stored the lane detection in previous frames in the `Lane` class. But due to time limit, I have not developed a comprehensive algorithm to use these information. This could be a future direction to improve the performance, and solve the challenge projects.


| correct detection with big shading | right lane detection failed with shading |
| :--------------------------------: | :------------------------------:|
|        ![alt text][image7a]        |       ![alt text][image7b]      |
| right lane distored with shading   | left lane incorrect detection   |
|        ![alt text][image7c]        |       ![alt text][image7d]      |



