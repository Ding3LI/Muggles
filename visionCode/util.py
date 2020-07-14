import numpy as np

def im2double(I):
    Min = I.min()
    Max = I.max()
    dis = float(Max - Min)
    m, n = I.shape
    J = np.zeros((m, n), dtype="float")
    for x in range(m):
        for y in range(n):
            a = I[x, y]
            if a != 255 and a != 0:
                b = float((I[x, y] - Min) / dis)
                J[x, y] = b
            J[x, y] = float((I[x, y] - Min) / dis)
    return J

def rot90cc(I, n=1):
    '''
    counter-clockwise 90 degrees
    :param I:
    :param n:  number of 90 degrees rot
    :return:
    '''
    rI = I
    for x in range(n):
        rI = list(zip(*rI[ : : ]))
    return rI


def rot90c(I):
    '''
    clockwise 90 degrees
    cc rotate 3 times 90 degrees
    :param I:
    :return:
    '''
    rI = I
    for x in range(3):
        rI = rot90cc(rI)
    return rI