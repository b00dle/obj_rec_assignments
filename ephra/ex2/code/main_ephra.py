import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def main():
    # load the image
    img = ndimage.imread('input.png', flatten=True, mode="F")

    # display the initial image
    plt.figure(0)
    plt.imshow(img, cmap='Greys_r')

    # computes the gaussian kernels
    img_x, img_y = apply_gog_filter(image_src=img, sigma=0.5)

    grey_image = img / 255

    calculate_autocorrelation_mat(grey_image, img_x, img_y)


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
    new_image_x = np.ndarray(image_src.shape)
    new_image_y = np.ndarray(image_src.shape)
    for r in range(padding, image_src.shape[0] - padding):
        for c in range(padding, image_src.shape[1] - padding):
            # select submatrix from image
            submat = image_src[r - padding:r + padding + 1, c - padding:c + padding + 1]
            new_image_x[r][c] = filter_center_value(submat, gog_x)
            new_image_y[r][c] = filter_center_value(submat, gog_y)

    plt.imshow(new_image_x, cmap="Greys_r")
    plt.imshow(new_image_y, cmap="Greys_r")

    return new_image_x, new_image_y


def filter_center_value(image_sub_mat, gog_mat):
    """
    Applies a gog filter mat to a similarly sized image_sub_mat.
    :param image_sub_mat: sub matrix containing image values 
    :param gog_mat: filter matrix to apply
    :return: center value from image_sub mat filtered by gog_mat
    """

    # for val in np.nditer(image_sub_mat):
    #     for c in range(0, image_sub_mat.shape[1]):
    #         sum += image_sub_mat[r][c] * gog_mat[r][c]

    sum = 0.0
    for img_val, gog_val in np.nditer([image_sub_mat, gog_mat]):
        sum += img_val * gog_val
    return sum / float(image_sub_mat.shape[0] * image_sub_mat.shape[1])


def gog_filter(c_x, c_y, sigma=0.5):
    """
    apply gradient of Gaussian to filter arrays.
    Assignment task A.a
    :param c_x: x filter ndarray
    :param c_y: y filter ndarray
    :param sigma: standard deviation
    :return: gaussian filtered arrays for c_x and c_y
    """
    # compute gog_x
    factor_left = -1 * (c_x / 2 * np.pi * sigma ** 4)
    factor_right = np.exp(-1 * (c_x ** 2 + c_y ** 2) / 2 * sigma ** 2)
    gog_x = factor_left * factor_right

    # compute gog_x
    factor_left = -1 * (c_y / 2 * np.pi * sigma ** 4)
    factor_right = np.exp(-1 * (c_x ** 2 + c_y ** 2) / 2 * sigma ** 2)
    gog_y = factor_left * factor_right
    return gog_x, gog_y


def calculate_autocorrelation_mat(grey_scale_image, i_x_mat, i_y_mat):
    i_x_mat_2 = i_x_mat * i_x_mat
    i_y_mat_2 = i_y_mat * i_y_mat
    i_x_y_mat = i_x_mat * i_y_mat

    # get padding
    padding = int(i_x_mat.shape[0] / 2)
    w_mat = np.zeros(i_x_mat.shape)
    print(w_mat)

    for r in range(padding, grey_scale_image.shape[0] - padding):
        for c in range(padding, grey_scale_image.shape[1] - padding):
            # select submatrix from image
            w_mat[r,c] = grey_scale_image[r - padding:r + padding + 1, c - padding:c + padding + 1]





if __name__ == '__main__':
    main()
    # a = np.arange(9).reshape(3,3)
    # print(a)
    # for val in np.nditer(a):
    #     print(val)
