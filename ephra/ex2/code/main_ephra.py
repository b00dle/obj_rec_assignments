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
    img_x, img_y = apply_gog_filter(image_src=img, sigma=0.5)

    ## compute auto correlation mat
    auto_cor_mat = compute_auto_correlation_matrix(img_x, img_y)

    # compute roundness and cornerness
    roundness, cornerness = compute_roundness_and_cornerness(auto_cor_mat)

    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.imshow(roundness, cmap="jet")
    plt.title("roundness")

    plt.subplot(1, 2, 2)
    plt.title("cornerness")
    plt.imshow(cornerness, cmap="jet")

    # compute interest points binary mask from roundness and cornerness
    mask = compute_binary_mask(roundness, cornerness)

    plt.figure(5)
    plt.title("binary interest point mask")
    plt.imshow(mask, cmap="Greys_r")

    # multiple mask with roundness and cornerness and get regional max
    mask_round = roundness * mask
    mask_corner = cornerness * mask
    mask_combined = mask_round * mask_corner
    mask_regio_max = ndimage.filters.maximum_filter(mask_combined, footprint=np.ones((3, 3)))

    plt.figure(6)
    plt.title("final interest points")
    plt.imshow(mask_regio_max, cmap="Greys_r")

    # show all figures
    plt.show()


def compute_binary_mask(roundness, cornerness, roundness_thresh=0.5, cornerness_thresh=0.004):
    """

    :param roundness: 
    :param cornerness: 
    :param roundness_thresh: 
    :param cornerness_thresh: 
    :return: 
    """
    mask = np.zeros(roundness.shape)
    max_round = -1000000.0
    max_corner = -1000000.0
    for r in range(0, mask.shape[0]):
        for c in range(0, mask.shape[1]):
            round = roundness[r][c]
            corner = cornerness[r][c]
            if round > max_round:
                max_round = round
            if corner > max_corner:
                max_corner = corner
            round_ok = round > roundness_thresh
            corner_ok = corner > cornerness_thresh
            if round_ok and corner_ok:
                mask[r][c] = 1.0
            else:
                mask[r][c] = 0.0
    return mask


def compute_roundness_and_cornerness(auto_cor_mat):
    """
    computes roundness for each pixel given 
    the auto correlation matrix of an image
    :param auto_cor_mat: 
    :return: 
    """
    padding = 2
    roundness = np.zeros(auto_cor_mat.shape)
    cornerness = np.zeros(auto_cor_mat.shape)
    for r in range(padding, auto_cor_mat.shape[0] - padding):
        for c in range(padding, auto_cor_mat.shape[1] - padding):
            m = auto_cor_mat[r, c]
            m_trace = np.trace(m) ** 2
            if m_trace == 0:
                m_trace = 0.000000001
            roundness[r, c] = (4 * np.linalg.det(m)) / m_trace
            cornerness[r, c] = (np.trace(m) / 2.0) - np.sqrt((np.trace(m) / 2.0) ** 2 - np.linalg.det(m))
    return roundness, cornerness


def apply_gog_filter(image_src, sigma=0.5):
    c_x = np.asarray([
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    ])

    c_y = c_x.transpose()

    # apply GoG to c_x
    gog_x, gog_y = gog_filter(c_x, c_y, sigma)

    # show kernel
    plt.figure(1)
    plt.imshow(gog_x, cmap="Greys_r")

    # get padding
    padding = int(gog_x.shape[0] / 2)

    # scale image to [0..1]
    image_src = image_src / 255

    # apply gog masks to image
    new_image_x = np.ndarray(image_src.shape)
    new_image_y = np.ndarray(image_src.shape)
    for r in range(padding, image_src.shape[0] - padding):
        for c in range(padding, image_src.shape[1] - padding):
            # select submatrix from image
            submat = image_src[r - padding:r + padding + 1, c - padding:c + padding + 1]
            # apply filter using gog_x matrix
            new_image_x[r][c] = filter_center_value(submat, gog_x)
            # apply filter using gog_y matrix
            new_image_y[r][c] = filter_center_value(submat, gog_y)

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(new_image_x, cmap="Greys_r")
    plt.title("x-direction filtered")

    plt.subplot(1, 2, 2)
    plt.title("y-direction filtered")
    plt.imshow(new_image_y, cmap="Greys_r")

    # compute gradient magnitude image (task A.c)
    new_image_grad_mag = np.sqrt(new_image_x ** 2 + new_image_y ** 2)
    plt.figure(3)
    plt.imshow(new_image_grad_mag, cmap="Greys_r")

    return new_image_x, new_image_y


def compute_auto_correlation_matrix(image_x, image_y):
    w = np.asarray([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    ix_2 = image_x ** 2
    ixy = image_x * image_y
    iy_2 = image_y ** 2

    padding = int(w.shape[0] / 2)

    auto_cor_mat = np.empty(image_x.shape, dtype=object)

    # auto_cor_mat = np.full(image_x.shape, np.zeros((2, 2)), dtype=object)
    #
    # for r in range(0, image_x.shape[0]):
    #     for c in range(0, image_x.shape[1]):
    #         auto_cor_mat[r, c] = np.zeros((2, 2))

    for r in range(padding, image_x.shape[0] - padding):
        for c in range(padding, image_x.shape[1] - padding):
            # select submatrix from images
            ix_2_sub = ix_2[r - padding:r + padding + 1, c - padding:c + padding + 1]
            iy_2_sub = iy_2[r - padding:r + padding + 1, c - padding:c + padding + 1]
            ixy_sub = ixy[r - padding:r + padding + 1, c - padding:c + padding + 1]
            # compute sum for sub matrices
            ix_2_sum = 0.0
            for val in np.nditer(ix_2_sub):
                ix_2_sum += val
            iy_2_sum = 0.0
            for val in np.nditer(iy_2_sub):
                iy_2_sum += val
            ixy_sum = 0.0
            for val in np.nditer(ixy_sub):
                ixy_sum += val
            # build local m
            M_2_2 = np.asarray([
                [ix_2_sum, ixy_sum],
                [ixy_sum, iy_2_sum]
            ])
            # compute w and sum up neighborhood M
            w = ix_2_sub + iy_2_sub + ixy_sub
            M = np.zeros(M_2_2.shape)
            for w_n in np.nditer(w):
                M += w_n * M_2_2
            auto_cor_mat[r, c] = M

    print(auto_cor_mat[0:5, 0:5])
    return auto_cor_mat


def filter_center_value(image_sub_mat, gog_mat):
    """
    Applies a gog filter mat to a similarly sized image_sub_mat.
    :param image_sub_mat: sub matrix containing image values 
    :param gog_mat: filter matrix to apply
    :return: center value from image_sub mat filtered by gog_mat
    """
    sum = 0.0
    for r in range(0, image_sub_mat.shape[0]):
        for c in range(0, image_sub_mat.shape[1]):
            sum += image_sub_mat[r][c] * gog_mat[r][c]
    return sum  # / float(image_sub_mat.shape[0]*image_sub_mat.shape[1])


def gog_filter(c_x, c_y, sigma=0.5):
    """
    apply gradient of Gaussian to filter arrays.
    Assignment task A.a
    :param c_x: x filter array
    :param c_y: y filter array
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


if __name__ == '__main__':
    main()
