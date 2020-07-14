# encoding: utf-8

import numpy as np
from PIL import Image
import time
from pencilMagic import *

# ----------------------------------------------------------------------------------------------------------------------------------

'''
    Thanks to this public git thread for rgb2yuv/yuv2rgb by np.array
    Reference: https://gist.github.com/Quasimondo/c3590226c924a06b276d606f4f189639
'''

# input is a RGB numpy array with shape (height,width,3), can be uint,int, float or double, values expected in the range 0..255
# output is a double YUV numpy array with shape (height,width,3), values in the range 0..255
def RGB2YUV(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv


# input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
# output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    return rgb

# ----------------------------------------------------------------------------------------------------------------------------------

'''
    gammaE = 3 , gammaI = 0 , scale = 235 ->  bright color
    gammaE = 3 , gammaI = 1 , scale = 255 ->   mild  color
'''
def get_color(img, gammaE=3, gammaI=1):

    img = np.array(img)
    YUV = RGB2YUV(img)
    imtype = 'color'

    #  create pencil drawing here
    S = stroke(YUV[:, :, 0], gammaE=gammaE)
    T = tone_map(YUV[:, :, 0], imtype, gammaI=gammaI)
    R = S * T

    tempYUV = YUV[:, :, :]
    tempYUV[:, :, 0] = R * 255

    R = YUV2RGB(tempYUV)
    outImg = Image.fromarray(R.clip(0, 255).astype('uint8'))
    outImg.show()
    return outImg

if __name__ == "__main__":
    start = time.time()
    img = Image.open('../media/img/tested.jpg')
    out = get_color(img)
    print('Used: {}s'.format(time.time() - start))
