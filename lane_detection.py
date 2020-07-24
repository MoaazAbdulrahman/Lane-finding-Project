import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import timeit
import time
from moviepy.editor import VideoFileClip


# Calculate directional gradient
def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=255):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    max_value = np.max(abs_sobel)
    binary_output = np.uint8(255*abs_sobel/max_value)
    threshold_mask = np.zeros_like(binary_output)
    threshold_mask[(binary_output >= thresh_min) & (binary_output <= thresh_max)] = 1
    return threshold_mask

def get_thresholded_image(img):
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    # apply gradient threshold on the horizontal gradient
    sx_binary = abs_sobel_thresh(gray, 'x', 10, 200)

    # apply gradient direction threshold so that only edges closer to vertical are detected.
    dir_binary = dir_threshold(gray, thresh=(np.pi / 6, np.pi / 2))

    # combine the gradient and direction thresholds.
    combined_condition = ((sx_binary == 1) & (dir_binary == 1))

    # R & G thresholds so that yellow lanes are detected well.
    color_threshold = 150
    R = img[:, :, 0]
    G = img[:, :, 1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)

    # color channel thresholds
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    L = hls[:, :, 1]

    # S channel performs well for detecting bright yellow and white lanes
    s_thresh = (100, 255)
    s_condition = (S > s_thresh[0]) & (S <= s_thresh[1])

    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
    l_thresh = (120, 255)
    l_condition = (L > l_thresh[0]) & (L <= l_thresh[1])

    # combine all the thresholds
    # A pixel should either be a yellowish or whiteish
    # And it should also have a gradient, as per our thresholds
    color_combined[(r_g_condition & l_condition) & (s_condition | combined_condition)] = 1

    # apply the region of interest mask
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([[0, height - 1], [width / 2, int(0.5 * height)], [width - 1, height - 1]],
                                           dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(color_combined, mask)

    return thresholded
'''
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute values
    sobel = np.absolute(sobel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    # Return the result
    return binary_output
'''

# Calculate gradient magnitude
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    # Return the result
    return binary_output


# Calculate gradient direction
def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Error statement to ignore division and invalid errors
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely / sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
    # Return the result
    return dir_binary


# Do perspective transform
def birds_eye_view(gray):

    img_size = (gray.shape[1], gray.shape[0])

    offset = 0
    src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    '''
    source = np.float32([[500, 482], [780, 482],
                         [1250, 720], [40, 720]])
    destination = np.float32([[0, 0], [1280, 0],
                              [1250, 720], [40, 720]])
                              '''
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(gray, M, img_size)
    return warped, M, M_inv, img_size

def window(warped, leftx_base, rightx_base):
    out_img = np.dstack((warped, warped, warped)) * 255

    non_zeros = warped.nonzero()
    non_zeros_y = non_zeros[0]
    non_zeros_x = non_zeros[1]

    num_windows = 10
    num_rows = warped.shape[0]
    window_height = np.int(num_rows / num_windows)
    window_half_width = 50

    min_pixels = 100

    left_coordinates = []
    right_coordinates = []

    for window in range(num_windows):
        y_max = num_rows - window * window_height
        y_min = num_rows - (window + 1) * window_height

        left_x_min = leftx_base - window_half_width
        left_x_max = leftx_base + window_half_width

        cv2.rectangle(out_img, (left_x_min, y_min), (left_x_max, y_max), [0, 0, 255], 2)

        good_left_window_coordinates = (
                    (non_zeros_x >= left_x_min) & (non_zeros_x <= left_x_max) & (non_zeros_y >= y_min) & (
                        non_zeros_y <= y_max)).nonzero()[0]
        left_coordinates.append(good_left_window_coordinates)

        if len(good_left_window_coordinates) > min_pixels:
            leftx_base = np.int(np.mean(non_zeros_x[good_left_window_coordinates]))

        right_x_min = rightx_base - window_half_width
        right_x_max = rightx_base + window_half_width

        cv2.rectangle(out_img, (right_x_min, y_min), (right_x_max, y_max), [0, 0, 255], 2)

        good_right_window_coordinates = (
                    (non_zeros_x >= right_x_min) & (non_zeros_x <= right_x_max) & (non_zeros_y >= y_min) & (
                        non_zeros_y <= y_max)).nonzero()[0]
        right_coordinates.append(good_right_window_coordinates)

        if len(good_right_window_coordinates) > min_pixels:
            rightx_base = np.int(np.mean(non_zeros_x[good_right_window_coordinates]))

    left_coordinates = np.concatenate(left_coordinates)
    right_coordinates = np.concatenate(right_coordinates)

    out_img[non_zeros_y[left_coordinates], non_zeros_x[left_coordinates]] = [255, 0, 0]
    out_img[non_zeros_y[right_coordinates], non_zeros_x[right_coordinates]] = [0, 0, 255]

    left_x = non_zeros_x[left_coordinates]
    left_y = non_zeros_y[left_coordinates]

    polyfit_left = np.polyfit(left_y, left_x, 2)

    right_x = non_zeros_x[right_coordinates]
    right_y = non_zeros_y[right_coordinates]

    polyfit_right = np.polyfit(right_y, right_x, 2)

    y_points = np.linspace(0, num_rows - 1, num_rows)

    left_x_predictions = polyfit_left[0] * y_points ** 2 + polyfit_left[1] * y_points + polyfit_left[2]

    right_x_predictions = polyfit_right[0] * y_points ** 2 + polyfit_right[1] * y_points + polyfit_right[2]
    return polyfit_left, polyfit_right, non_zeros_y, non_zeros_x


def window2(polyfit_left, polyfit_right,non_zeros_y, non_zeros_x, num_rows,img, M_inv,img_size,warped ):
    margin = 50
    out_img = np.dstack((warped, warped, warped)) * 255

    left_x_predictions = polyfit_left[0] * non_zeros_y ** 2 + polyfit_left[1] * non_zeros_y + polyfit_left[2]
    left_coordinates = \
    ((non_zeros_x >= left_x_predictions - margin) & (non_zeros_x <= left_x_predictions + margin)).nonzero()[0]

    right_x_predictions = polyfit_right[0] * non_zeros_y ** 2 + polyfit_right[1] * non_zeros_y + polyfit_right[2]
    right_coordinates = \
    ((non_zeros_x >= right_x_predictions - margin) & (non_zeros_x <= right_x_predictions + margin)).nonzero()[0]

    out_img[non_zeros_y[left_coordinates], non_zeros_x[left_coordinates]] = [255, 0, 0]
    out_img[non_zeros_y[right_coordinates], non_zeros_x[right_coordinates]] = [0, 0, 255]

    left_x = non_zeros_x[left_coordinates]
    left_y = non_zeros_y[left_coordinates]

    polyfit_left = np.polyfit(left_y, left_x, 2)

    right_x = non_zeros_x[right_coordinates]
    right_y = non_zeros_y[right_coordinates]

    polyfit_right = np.polyfit(right_y, right_x, 2)

    y_points = np.linspace(0, num_rows - 1, num_rows)

    left_x_predictions = polyfit_left[0] * y_points ** 2 + polyfit_left[1] * y_points + polyfit_left[2]

    right_x_predictions = polyfit_right[0] * y_points ** 2 + polyfit_right[1] * y_points + polyfit_right[2]

    window_img = np.zeros_like(out_img)

    left_line_window_1 = np.array(np.transpose(np.vstack([left_x_predictions - margin, y_points])))

    left_line_window_2 = np.array(np.flipud(np.transpose(np.vstack([left_x_predictions + margin, y_points]))))

    left_line_points = np.vstack((left_line_window_1, left_line_window_2))

    cv2.fillPoly(window_img, np.int_([left_line_points]), [0, 255, 0])

    right_line_window_1 = np.array(np.transpose(np.vstack([right_x_predictions - margin, y_points])))

    right_line_window_2 = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions + margin, y_points]))))

    right_line_points = np.vstack((right_line_window_1, right_line_window_2))

    cv2.fillPoly(window_img, np.int_([right_line_points]), [0, 255, 0])

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    """"""""
    out_img = np.dstack((warped, warped, warped)) * 255

    y_points = np.linspace(0, num_rows - 1, num_rows)

    left_line_window = np.array(np.transpose(np.vstack([left_x_predictions, y_points])))

    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions, y_points]))))

    line_points = np.vstack((left_line_window, right_line_window))

    cv2.fillPoly(out_img, np.int_([line_points]), [0, 0, 255])

    unwarped = cv2.warpPerspective(out_img, M_inv, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
    """"""""
    print("--- %s seconds ---" % (time.time() - start_time))
    #plt.imshow(result)

    #plt.show()
    return result
'''
def inverse(left_x_predictions, right_x_predictions):
    out_img = np.dstack((warped, warped, warped)) * 255

    y_points = np.linspace(0, num_rows - 1, num_rows)

    left_line_window = np.array(np.transpose(np.vstack([left_x_predictions, y_points])))

    right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_x_predictions, y_points]))))

    line_points = np.vstack((left_line_window, right_line_window))

    cv2.fillPoly(out_img, np.int_([line_points]), [0, 255, 0])

    unwarped = cv2.warpPerspective(out_img, M_inv, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)

    plt.imshow(result)
'''

start_time = time.time()

image = cv2.imread('test_images/test2.jpg')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bin_img = get_thresholded_image(image)
warped, M, M_inv,img_size = birds_eye_view(bin_img)
num_rows = warped.shape[0]
histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)

    # Peak in the first half indicates the likely position of the left lane
half_width = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:half_width])

# Peak in the second half indicates the likely position of the right lane
rightx_base = np.argmax(histogram[half_width:]) + half_width

polyfit_left, polyfit_right, non_zeros_y, non_zeros_x = window(warped, leftx_base, rightx_base)
result = window2(polyfit_left, polyfit_right,non_zeros_y, non_zeros_x, num_rows,image, M_inv, img_size, warped)
filename = 'output/image3.jpg'
cv2.imwrite(filename, result)
plt.imshow(result)
plt.show()
'''
output = 'output/project_video_output.mp4'
clip1 = VideoFileClip("test_images/project_video.mp4")
write_clip = clip1.fl_image(f) #NOTE: this function expects color images!!
write_clip.write_videofile(output, audio=False)
'''
