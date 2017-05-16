import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def main():
    # load the image
    img = ndimage.imread('input.png', flatten=True, mode="F")

    # display the initial image
    plt.figure(0)
    plt.imshow(img, cmap='Greys_r')

    ## computes the gaussian kernels
    img = apply_gog_filter(image_src=img, sigma=0.5)

    # show all figures
    plt.show()


def apply_gog_filter(image_src, sigma=0.5):
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

    # show kernel
    plt.figure(1)
    plt.imshow(gog_x, cmap="Greys_r")

    # get padding
    padding = int(gog_x.shape[0] / 2)

    # scale image to [0..1]
    image_src = image_src / 255

    # print(gog_x)
    # print(gog_x[padding - padding:padding + padding, padding - padding:padding + padding])

    # apply gog masks to image
    new_image = np.ndarray
    for r in range(padding, image_src.shape[0] - padding + 1):
        for c in range(padding, image_src.shape[1] - padding + 1):
            # select submatrix from image
            submat = image_src[r - padding:r + padding + 1, c - padding:c + padding + 1]
            print(submat.shape)

    return image_src


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


if __name__ == '__main__':
    main()
