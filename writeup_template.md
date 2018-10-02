## Project 2: Advanced Lane Finding

---

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
* Define the final pipeline of processing image
* Define the final video pipeline

[//]: # (Image References)

[image01]: ./camera_cal/calibration1.jpg "Undistorted Chess Board"
[image02]: ./output_images/calibration1.jpg "Undistorted Chess Board"
[image1]: ./output_images/undist_image.jpg "Undistorted"
[image2]: ./output_images/combined_binary.jpg "Combined Binary"
[image3]: ./output_images/binary_warped.jpg "Binary Warped Image"
[image4]: ./output_images/fit_poly_img.jpg "fit poly img"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/poly_line.jpg "Poly lines"
[image7]: ./output_images/line_on_image.jpg "line_on_image"
[image8]: ./output_images/data_on_image.jpg "data_on_image"
[video1]: ./output_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I used function `cal_undist_params()` to calculate the mtx and dist needed for camera calibration. 
In this function, I iterated each chesse board image, transform them to grayscale, then use `cv2.findChessboardCorners` function to find ret and all corners. 
Then I used `cv2.calibrateCamera()` to generate `mtx` and `dist` parms needed for calibrating camera. 
Then I used those params to distort the image, using function `undist_single_img`, Result example is like below:

Raw image:
![alt text 1][image01]  

Undist image:
![alt text 2][image02]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Then I used `undist_single_img()` function to correct real image:

![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I defined several functions to find generate binary image:
* `abs_sobel_thresh()` is using binary threshold depends on x or y;
* `mag_thresh()` is for calculating gradient magnitude
* `dir_threshold()` is for direction gradient
* `hls_select()` is for s channel color threshold in HLS color space

Then I combined the results of those functions in `pipeline()` function:

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`

It takes an binary image and performs a perspective transform

```python
    src = np.float32([[1280,720],[720, 450], [560, 450], [0,720]])
    dst = np.float32([[1280,720],[1280,0],[0,0],[0,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1280,720      | 1280,720        | 
| 720, 450      | 1280,0      |
| 560, 450      | 0,0      |
| 0, 720        | 0,720        |

The perspective transform result is:

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

The code is in `find_lane_pixels()` function and `fit_polynomial()` function. 

What the do is taking the binary warped image and find all pixels that belongs to right or left line. In detail:
Firstly it find the bottom pixels, then use sliding windows method to move up and shift some pixels to find all points.
![alt text][image4]

Then we use `np.polyfit` in `fit_polynomial` function to fit a second order polynomial for each line. Then use y value in points to caculate x value, and draw on the image: 
![alt text][image6]

 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use `measure_curvature_real()` function to detect the lane curvature and distance to center of lane.

Basically it takes in the binary warped image and use `fit_polynomial` to caculate the fit for lines, the use the poly to calculate the radius of curvature.

used 
```python
ym_per_pix = 3.048/100 # meters per pixel in y dimension
xm_per_pix = 3.7/900 # meters per pixel in x dimension
```
as transform params to real life.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I used `draw_lane()` method, taking the original image, binary image and left and right fit, transform the lines back to normal size and shape , then add it on to orinial image:
![alt text][image7]

Lastly I draw the calculated data on to the output image:
 
![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found this project has more challenge than the first one. But it has more fun!

Problem I had:

* I spent too much time on python's libraries.
* I had a bad time of using the workspace.
* I should use more code wrote by myself instead of using code from lessons.
* I want to spend more time to polish the code, so I'll come back again!