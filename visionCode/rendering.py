# encoding: utf-8

import numpy as np

#  -----------  Alpha Blending  --------------------------------
def horizontal_blend(img, width):
    '''
    Horizontally blend
    :param img:     input img
    :param width:   width for blending
    :return:        horizontally blended img
    '''
    Hblend = img
    while Hblend.shape[1] < width:

        kernel_size = int(round(img.shape[1] / 4))  #  make sure your kernel is in nxn shape

        left = img[:, (img.shape[1] - kernel_size) : img.shape[1]]
        right = img[:, 0:kernel_size]

        lr, lc = left.shape
        alpha_l = np.zeros((lr, kernel_size))
        alpha_r = np.zeros((lr, kernel_size))

        for i in range(kernel_size):
            alpha_l[:, i] = left[:, i] * (1 - float(i+1) / kernel_size)
            alpha_r[:, i] = right[:, i] * float(i+1) / kernel_size

        #  Blending
        Hblend = np.column_stack(
            ( Hblend[:, : (Hblend.shape[1] - kernel_size)],
              alpha_l + alpha_r,
              Hblend[:, kernel_size : Hblend.shape[1]] )
        )

    Hblend = Hblend[:, : width]
    return Hblend


def vertical_blend(img, height):
    '''
    vertically blending
    :param img:     input img
    :param height:  height limit for blending
    :return:        vertically blended img
    '''
    Vblend = img
    while Vblend.shape[0] < height:

        kernel_size = int(round(img.shape[0] / float(4)))

        up = img[(img.shape[0] - kernel_size) : img.shape[0], :]
        down = img[ : kernel_size, : ]

        ur, uc = up.shape
        alpha_up = np.zeros((kernel_size, uc))
        alpha_down = np.zeros((kernel_size, uc))

        for i in range(kernel_size):
            alpha_up[i, :] = up[i, :] * (1 - float(i+1) / kernel_size)
            alpha_down[i, :] = down[i, :] * float(i+1) / kernel_size

        #  Blending
        Vblend = np.row_stack(
            ( Vblend[0: Vblend.shape[0] - kernel_size, :],
              alpha_up + alpha_down,
              Vblend[kernel_size: Vblend.shape[0], :] )
        )

    Vblend = Vblend[ : height, : ]
    return Vblend