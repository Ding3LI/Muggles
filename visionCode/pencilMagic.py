# Magic for making real photo into pencil scratch

import cv2
import numpy as np
import sys, os

# local package
from util import *            #  type conversion , rotation
from hist_kernel import *     #  histogram matching
from rendering import *       #  alpha blend in directions

# image processing
from PIL import Image         #  image processing [ R, G, B ]

import scipy.signal as signal
from scipy.ndimage import interpolation
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csr_matrix as csr_matrix, spdiags as spdiags
from scipy.sparse.linalg import spsolve as spsolve

# ------------------ BEGIN Line Drawing --------------------------------
#  ratio => conv kernel size : image.size
len_divisor = 50  # the bigger the ratio, the more details can get
# gammaE = 1   --> thickness control

# image gradient -> line drawing ( strokes feature )
def stroke(img, gammaE=2):
    '''
    1. compute gradient
    2. convolve image in 8 directions, 45 degrees per each

    :param img:  type=PIL  np.array([Grayscale image])
    :param gammaE: the thickness of edge line
    :return: stokes image
    '''
    r, c = img.shape
    len_float = float(min(r, c)) / len_divisor
    line = int(len_float)
    line = line + (line % 2)
    half_line = line / 2

    # ------- start compute gradient in x , y directions --------
    doubleImg = im2double(img)
    Ix = np.column_stack((abs(doubleImg[:, :-1] - doubleImg[:, 1:]), np.zeros((r, 1))))
    Iy = np.row_stack((abs(doubleImg[:-1, :] - doubleImg[1:, :]), np.zeros((1, c))))
    Gimg = np.sqrt(Ix**2 + Iy**2)    #  Equation (1)
    # Gimg = Image.fromarray(Gimg*255)
    # Gimg.show()

    # ------- start convolve ---------------------------------
    # create a 3D with 8 pages for storing line segments from 8
    # directions.
    # ********************************************************************
    # ********************* IMPORTANT!!! *********************************
    #          line_seg[:, :, idx] -> idx+1 th direction line segment
    #          -> eg. line_seg[:, :, 1] = 2nd pos line segment
    # ********************************************************************
    # --> line_seg is a convolve kernel
    line_seg = np.zeros((line, line, 8))  #  relevant to { L }

    for i in range(8):
        for x in range(line):
            y = round(((x + 1) - half_line) * np.tan(np.pi / 8 * i))
            y = half_line - y
            if y > 0 and y <= line:
                line_seg[int(y - 1), x, i] = 1
            if i == 0:
                line_seg[:, :, 3] = rot90c(line_seg[:, :, 0])
            if i == 1:
                line_seg[:, :, 4] = rot90c(line_seg[:, :, 1])
            if i == 2:
                line_seg[:, :, 5] = rot90c(line_seg[:, :, 2])
            if i == 7:
                line_seg[:, :, 6] = rot90cc(line_seg[:, :, 7])

    #  ----------------------  Equation (2) ----------------------------------
    G = np.zeros((r, c, 8))
    for i in range(8):
        G[:, :, i] = signal.convolve2d(Gimg, line_seg[:, :, i], mode='same')
    #  ----------------------  Equation (2) ----------------------------------
    #  ----------------------  Equation (3) ----------------------------------
    #  we only take the direction with maximum
    Gmax = G.argmax(axis=2)  #  because we have 3D array
    C = np.zeros((r, c, 8))  #  C is array storing map_set
    for i in range(8):
        C[:, :, i] = Gimg * (Gmax == i)
    #  ----------------------  Equation (3) ----------------------------------

    #  Generate line segments
    temp_line = np.zeros((r, c, 8))
    for i in range(8):
        temp_line[:, :, i] = signal.convolve2d(C[:, :, i], line_seg[:, :, i], mode='same')

    # unifying
    temp = temp_line.sum(axis=2)
    temp = (temp - temp[:].min()) / (temp[:].max() - temp[:].min())
    output = (1 - temp) ** gammaE
    # output = Image.fromarray((output * 255).astype('uint8'))
    # output.show()
    # output.save('tifa.png')
    return output
# ------------------  END Line Drawing --------------------------------------

# ------------------  Start Create Pencil Texture  --------------------------

# loading pencil texture background
lamda = .8
texture_resize = 1.
texture_file = '../media/img/pencilText.jpg'
# gammaI = 1   --> darkness control

def tone_map(img, imtype='pencil', gammaI=1):
    '''
    This step is for tone rendering
    focus more on image's details:
    shadow, shape and shading.
    1. apply tone adjustment for grayscale img
    2. pencil texture rendering
    :param img:  type=PIL  np.array([Grayscale image])
    :param imtype:  type of image: "pencil" / "color"
    :param gammaI:  controlling the magnitude of color
    :return:  image with applied pencil texture upon tone map
    '''
    r, c = img.shape
    # output tone map image
    toneMap = histogram_matching(img, imtype=imtype) ** gammaI

    texture = Image.open(texture_file).convert('L')
    texture = np.array(texture)
    tr, tc = texture.shape
    #  take out a piece of pencil texture background
    texture = texture[99 : tr-100, 99 : tc-100]

    ratio = texture_resize * min(r, c) / float(1024)
    resize = interpolation.zoom(texture, (ratio, ratio))
    texture = im2double(resize)
    horizontal = horizontal_blend(texture, c)   #  horizontally rendering
    imgTexture = vertical_blend(horizontal, r)  #  vertically rendering

    temp = r * c

    ker = 2 * (temp - 1)
    x = np.zeros((ker, 1))
    y = np.zeros((ker, 1))
    z = np.zeros((ker, 1))
    for i in range(1, ker+1):
        x[i - 1] = int(np.ceil((i + 0.1) / 2)) - 1
        y[i - 1] = int(np.ceil((i - 0.1) / 2)) - 1
        z[i - 1] = -2 * (i % 2) + 1
    dx = csr_matrix((z.T[0], (x.T[0], y.T[0])), shape=(temp, temp))

    ker = 2 * (temp - c)
    x = np.zeros((ker, 1))
    y = np.zeros((ker, 1))
    z = np.zeros((ker, 1))
    for i in range(1, ker+1):
        x[i - 1] = int(np.ceil((i - 1 + 0.1) / 2) + (c * ( i % 2))) - 1
        y[i - 1] = int(np.ceil((i - 0.1) / 2)) - 1
        z[i - 1] = -2 * (i % 2) + 1
    dy = csr_matrix((z.T[0], (x.T[0], y.T[0])), shape=(temp, temp))

    imgTexture1D = np.log(np.reshape(imgTexture.T, (1, imgTexture.size), order='f') + 0.01)  #  +0.01 avoid 0 division
    imgSparse = spdiags(imgTexture1D, 0, temp, temp)
    adjusted1D = np.log(np.reshape(toneMap.T, (1, toneMap.size), order='f').T + 0.01)        #  +0.01 avoid 0 division

    baseimg = imgSparse.T.dot(adjusted1D)  #  ln( J(x) )
    val_sp = np.dot(imgSparse.T, imgSparse)
    val_x = dx.T.dot(dx)
    val_y = dy.T.dot(dy)
    imgMat = val_sp + lamda * (val_x + val_y)  #  ln( H(x) )

    #  ---------------  Equation (8)  --------------------------------
    # lnH(x) * beta(x) = lnJ(x) --> beta(x) = spsolve(lnH(x), lnJ(x))
    beta = spsolve(imgMat, baseimg)
    BETA = np.reshape(beta, (r, c), order='c')
    #  ---------------  Equation (8)  --------------------------------

    #  ---------------  Equation (9)  --------------------------------
    #  Lets add in shadow property by repeat line drawing
    render_shadow = imgTexture ** BETA
    render_shadow.clip(0, 255)
    # render_shadow = (render_shadow - render_shadow.min()) / (render_shadow.max() - render_shadow.min())
    #  ---------------  Equation (9)  --------------------------------

    ## Displaying output image  ################
    # outImg = Image.fromarray((render_shadow*255).clip(0, 255).astype('uint8'))
    # outImg.show()
    ###########################################
    return render_shadow

# ------------------    END Create Pencil Texture  --------------------------

# ------------------    Start Blending   ------------------------------------

def pencilMagic(img, gammaE=1, gammaI=1):
    img = np.array(img)
    S = stroke(img, gammaE=gammaE)
    T = tone_map(img, imtype='pencil', gammaI=gammaI)
    # ----------  Equation (10)  ---------------------
    pencil = S * T
    # ----------  Equation (10)  ---------------------
    outImg = Image.fromarray((pencil*255).clip(0, 255).astype('uint8'))
    outImg.show()

# ------------------    END   Blending   ------------------------------------

if __name__ =="__main__":

    img = Image.open('../media/img/test.bmp')
    img = img.convert('L')
    img = np.array(img)

    # out = stroke(img)
    # out = tone_map(img, 'pencil')
    pencilMagic(img)

