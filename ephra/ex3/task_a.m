
## Task A a)
# read image and convert to greyscale 
img_color = imread("input_ex3.jpg");
img_grey = rgb2gray(img_color);

## Task A b)
#
source("assignment2.m");

[GoG_x, GoG_y] = GoG_filtering(img_grey, 0.5);