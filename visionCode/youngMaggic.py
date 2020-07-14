# Magic for becoming Younger

# face recognition
'''
    *** install dlib ( IMPORTANT install )
        -->  install face_recognition
    for certain version of python to install the package:
        python3.x -m pip install face_recognition
'''
import face_recognition

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import cv2
import time

def enhancer(img):
    '''
    This is used for improving image's color and contrast
    :param img:  type = BGR / RGB
    :return:     type = PIL
    '''
    # --------------  cv2 method for improving output image's color  ---------------
    # hsvImg = cv2.cvtColor(magic, cv2.COLOR_BGR2HSV)
    # hsvImg[:, :, 1] = hsvImg[:, :, 1] * 1.3  #  change here -> Saturation
    # hsvImg[:, :, 2] = hsvImg[:, :, 2] * 1.   #  change here -> Brightness
    # outImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    # ------------------------------------------------------------------------------
    # --------------  PIL method for improving output image's color  ---------------
    correct_clr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp = Image.fromarray(correct_clr)
    enhancer_clr = ImageEnhance.Color(temp)
    temp = enhancer_clr.enhance(1.3)
    enhancer_ctrst = ImageEnhance.Contrast(temp)
    temp = enhancer_ctrst.enhance(1.2)
    temp.show()
    return temp # type = PIL
    # ------------------------------------------------------------------------------

def younger(img):
    """
    Equation:
        Dest =(Src * (100 - Opacity) + (Src + n * GuassianBlur(highPass_Filter(Src) - Src + 128) - 255) * Opacity) / 100
    """

    var1 = 5    #  magnitude for young face
    var2 = 3    #  filter kernel size = magnitude for face's details

    #  Assigning params for Bilateral Filter
    dx = var1 * 5
    fc = var1 * 15

    p = 0.5     #  weight

    tmp1 = cv2.bilateralFilter(img, dx, fc, fc)
    tmp2 = cv2.subtract(tmp1, img)
    tmp2 = cv2.add(tmp2, (10, 10, 10, 128))
    tmp3 = cv2.GaussianBlur(tmp2, (2 * var2 - 1, 2 * var2 - 1), 0)
    mixin = cv2.subtract(cv2.add(cv2.add(tmp3, tmp3), img), (10, 10, 10, 255))

    magic = cv2.addWeighted(img, p, mixin, 1-p, 0)
    magic = cv2.add(magic, (10, 10, 10, 255))

    return magic # type = BGR

def init():

    start = time.time()

    img = cv2.imread('../media/img/dotted.png')

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("../media/img/dotted.png")

    # pil_image = Image.fromarray(image)
    # # Create a Pillow ImageDraw Draw instance to draw with
    # draw = ImageDraw.Draw(pil_image)

# --------------------------------------------------------------------------------------------------------
    # # default HOG-based model
    # face_locations = face_recognition.face_locations(image)
    # Looking for faces by using CNN ( uncomment for testing CNN computing *** COST time ***)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
# --------------------------------------------------------------------------------------------------------

    print("Magic recognizes {} face(s) in the photograph.".format(len(face_locations)))

    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(255,105,180))
    del draw
    pil_img.show()

    print('Used: {}s'.format(time.time() - start))

    face = younger(img)
    youngFace = enhancer(face).save('youngFace.png')


if __name__ == "__main__":
    init()