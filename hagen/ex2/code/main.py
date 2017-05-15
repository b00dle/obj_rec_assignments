import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def main():
    # load the image
    img = ndimage.imread('input.png')

    # display the initial image
    plt.figure(0)
    plt.imshow(img)
    plt.show()

    filter_gaussian(img)


def filter_gaussian(image_src, sigma=0.5):
    """
    filter image using the given 5x5 gaussian kernel
    :param image_src: 
    :param sigma: 
    :return: 
    """

    # hardcoded kernel
    kernel = [[0.0000, 0.0001, 0.0000, -0.0001, -0.0000],
              [0.0002, 0.0466, 0.0000, -0.0466, -0.0002],
              [0.0017, 0.3446, 0.0000, -0.3446, -0.0017],
              [0.0002, 0.0466, 0.0000, -0.0466, -0.0002],
              [0.0000, 0.0001, 0.0000, -0.0001, -0.0000]]

    print(kernel)

    pass


if __name__ == '__main__':
    main()
