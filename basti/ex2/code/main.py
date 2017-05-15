import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def main():
    # load the image
    img = ndimage.imread('input.png', flatten=True)

    # display the initial image
    '''
    plt.figure(0)
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    '''

    compute_gaussian_kernel(0.5)
    # filter_gaussian(img)


def compute_gaussian_kernel(sigma=0.5):
    c_x = np.asarray([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    ])

    c_y = c_x.transpose()

    print("======= c_x ======\n" + str(c_x))
    print("======= c_y ======\n" + str(c_y))

    # apply GoG to c_x
    gog_x, gog_y = gog_filter(c_x, c_y, sigma)

    plt.figure(0)
    plt.imshow(gog_x, cmap="Greys_r")
    plt.show()

def gog_filter(c_x, c_y, sigma=0.5):
    '''
    apply gradient of Gaussian to filter arrays.
    Assignment task A.a
    :param c_x: x filter array
    :param c_y: y filter array
    :param sigma: standard deviation
    :return: gaussian filtered arrays for c_x and c_y
    '''
    # compute gog_x
    factor_left = -1 * (c_x / 2 * np.pi * sigma ** 4)
    factor_right = np.exp(-1 * (c_x ** 2 + c_y ** 2) / 2 * sigma ** 2)
    gog_x = factor_left * factor_right

    # compute gog_x
    factor_left = -1 * (c_y / 2 * np.pi * sigma ** 4)
    factor_right = np.exp(-1 * (c_x ** 2 + c_y ** 2) / 2 * sigma ** 2)
    gog_y = factor_left * factor_right
    return gog_x, gog_y


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


if __name__ == '__main__':
    main()
