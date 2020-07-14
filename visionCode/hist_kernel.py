# encoding: utf-8

import numpy as np
from PIL import Image
import time

# considering the conditions
def heaviside(v):
    return v if v >= 0 else 0

def P1(v):
    '''
    Equation (5)
    Laplacian distribution
    simulate bright layer

    sigma_b = 9  ( scale of distribution )
    :param v:
    :return: laplace operator
    '''

    # laplaceFrac = np.array([[1, 1, 1],
    #                         [1, -8, 1],
    #                         [1, 1, 1]])

    return float(1) / 9 * np.exp(-(255 - v) / float(9)) * heaviside(255 - v)


def P2(v):
    '''
    Equation (6)
    uniform distribution
    simulate (bright < mild < dark) mild layer

    u_a = 105
    u_b = 225  #  u_a and u_b defining the range of distribution
    :param v:
    :return:  mean operator
    '''
    return float(1) / (225 - 105) * (heaviside(v - 105) - heaviside(v - 225))


def P3(v):
    '''
    Equation (7)
    Gaussian distribution
    simulate dark layer

    sigma_d = 11  #  scale for distribution
    mu_d = 90     #  mean value
    :param v:
    :return:  Gauss operator
    '''

    # def gauss(x, y, sigma=1):
    #     return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))

    return float(1) / np.sqrt(2 * np.pi * 11) * np.exp(-((v - 90) ** 2) / float(2*(11 ** 2)))


def P(v, imtype="pencil"):
    '''
    Equation (4)
    apply Maximum Likelihood Estimation ( MLE ) to estimate weights
    BW pencil sketch uses : ( the most neat distribution )
        [omega_1 : omega_2 : omega_3 = 42 : 29 : 29]

    Color pencil : ( tried by own, param can be varied )
        [omega_1 : omega_2 : omega_3 = 62 : 30 : 5]

    :param v:
    :param imtype: "color" / "pencil"
    :return:
    '''

    return (42 * P1(v) + 29 * P2(v) + 29 * P3(v)) if imtype == 'pencil' else (62*P1(v) + 30*P2(v) + 5*P3(v))

    # if type == "color":
    #     return 62*P1(v) + 30*P2(v) + 5*P3(v)
    # else:
    #     return 76*P1(v) + 22*P2(v) + 2*P3(v)


def histogram_matching(img, imtype="pencil"):
    '''
    Adjust tone maps using histogram matching in all
    3 layers.

    :param img:     grayscale 0~255
    :param imtype:  "color" / "pencil"
    :return:        adjusted tone map
    '''

    r, c = img.shape

# -------- Start compute img histogram  --------------------------------------------------
    # compute acc histogram for input grayscale img
    gray_acc = np.zeros((1, 256))  # accumulator  #  0 ~ 255
    gray_pix = np.zeros((1, 256))  # indicate each grayscale correspond to how many pixels

    for i in range(256):
        gray_pix[0, i] = sum(sum(1 * img == i))   #  formulate grayscale array

    gray_pix /= float(sum(sum(gray_pix)))

    gray_acc[0, 0] = gray_pix[0, 0]
    for i in range(1, 256):
        gray_acc[0, i] = gray_acc[0, i-1] + gray_pix[0, i]
# -------- END  compute img histogram  ----------------------------------------------------

# -------- Start compute img w/ func( P(v) ) histogram  -----------------------------------
    ori_acc = np.zeros((1, 256))
    ori_pix = np.zeros((1, 256))

    # Equation (4)
    for i in range(256):
        ori_pix[0, i] = P(i, imtype)   #  formulate color array

    ori_pix /= float(sum(sum(ori_pix)))

    ori_acc[0] = ori_pix[0]
    for i in range(1, 256):
        ori_acc[0, i] = ori_acc[0, i-1] + ori_pix[0, i]
# -------- END compute img w/ func( P(v) ) histogram  ------------------------------------

# -------- Start Histogram Matching  -----------------------------------------------------
    adjusted = np.zeros((r, c))
    for x in range(r):
        for y in range(c):
            hist_val = gray_acc[0, int(img[x, y])]
            idx = (abs(ori_acc - hist_val)).argmin()  #  only take the closest point
            adjusted[x, y] = idx
# -------- END  Histogram  Matching  -----------------------------------------------------
    return adjusted / 255.

def main():
    start = time.time()
    imr = Image.open("../media/img/tested.jpg")
    im = imr.convert("L")
    J = np.array(im)
    out = histogram_matching(J)
    out = Image.fromarray((out*255).clip(0, 255).astype('uint8'))
    out.show()
    print('Used: {}s'.format(time.time() - start))


if __name__ == "__main__":
    main()