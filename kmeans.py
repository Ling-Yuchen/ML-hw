import cv2
import numpy as np


def img_show(msg, img):
    cv2.imshow(msg, img)
    cv2.waitKey(0)


def input_img(img_path):
    _img = cv2.imread(img_path)
    _img = cv2.resize(_img, (256, 256))
    img_show('origin-img', _img)
    return _img


def origin_to_gray(_img):
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    img_show('gray-img', _img)
    return _img


def Gaussian_Blur(_img):
    _img = cv2.GaussianBlur(_img, (3, 3), 1)
    img_show('blur-img', _img)
    return _img


def remove_shadow(_img):
    pixel = int(np.mean(_img[_img > 140]))
    _img[_img > 10] = pixel
    img_show('remove-shadow-img', _img)
    return _img


def gen_center(data, k):
    n_sample, n_feature = data.shape
    f_mean = np.mean(data, axis=0).reshape((1, n_feature))
    f_std = np.std(data, axis=0).reshape((1, n_feature))
    centers = np.random.randn(k, n_feature) * f_std + f_mean  # (k, n_feature)
    return centers


def kmeans(img_path, k, use_gray=True, use_GaussianBlur=True, use_remove_shadow=False, use_cv2lib=False):
    """
    Use K-means algorithm to realize image segmentation.

    :param img_path: path of input image
    :param k: preset number of clusters
    :param use_gray: convert RGB-image to Gray-Scale-Image in preprocess
    :param use_GaussianBlur: use Gaussian Blur in preprocess
    :param use_remove_shadow: remove shadow in preprocess
    :param use_cv2lib: True if cv2.kmeans() is allowed, use pure numpy otherwise
    """
    # read image && resize to 256*256
    img = input_img(img_path)

    # preprocess
    if use_gray:
        img = origin_to_gray(img)
    if use_GaussianBlur:
        img = Gaussian_Blur(img)
    if use_remove_shadow:
        img = remove_shadow(img)

    # reshape
    if use_gray:
        data = img.reshape((-1, 1))
    else:
        data = img.reshape((-1, 3))
    data = np.float32(data)

    if use_cv2lib:
        compactness, clusters, centers = cv2.kmeans(
            data=data,
            K=k,
            bestLabels=None,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,  # type
                10,  # max_iter
                1.0  # epsilon
            ),  # define condition of ending
            attempts=10,
            flags=cv2.KMEANS_PP_CENTERS  # set initial centers
        )

        # reconstruct image
        centers = np.uint8(centers)
        clusters = clusters.flatten()

    else:
        n_sample, n_feature = data.shape
        clusters = np.zeros(n_sample)  # each sample corresponds with a cluster
        dist = np.zeros((n_sample, k))  # each sample has a distance from each center

        from copy import deepcopy
        cent_cur = gen_center(data, k)
        cent_pre = np.zeros(cent_cur.shape)
        cent_move = np.linalg.norm(cent_cur - cent_pre)  # 每轮迭代后质心的移动距离

        epsilon = 1e-3  # 质心需要移动的最小距离
        epoch = 0  # 当前迭代次数
        max_iter = 50  # 最大迭代次数
        while epoch < max_iter and cent_move > epsilon:
            epoch += 1

            # calculate distance from each sample to each center
            for i in range(k):
                dist[:, i] = np.linalg.norm(data - cent_cur[i], axis=1)

            # each sample belongs to cluster of the nearest center
            clusters = np.argmin(dist, axis=1)

            cent_pre = deepcopy(cent_cur)
            # calculate mean coordinate on each cluster, update center
            for i in range(k):
                cent_cur[i] = np.mean(data[clusters == i], axis=0)
            cent_move = np.linalg.norm(cent_cur - cent_pre)

        # reconstruct image
        centers = np.uint8(cent_cur)

    # reshape back && display
    res = centers[clusters]
    img = res.reshape(img.shape)
    img_show('result-img', img)


if __name__ == '__main__':
    kmeans('data/img/5.jpg', 4, use_gray=False, use_GaussianBlur=True, use_remove_shadow=False, use_cv2lib=False)
