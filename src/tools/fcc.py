# Python3 code for generating 8-neighbourhood chain
# code for a 2-D line

import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

codeList = [
    0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, np.pi+np.pi/4, np.pi+np.pi/2, np.pi+3*np.pi/4, np.pi+np.pi
]

def m_between_points(x0, y0, x1, y1):

    centroids = [(x0,y0), (x1, y1)]
    x_coords, y_coords = zip(*centroids)
    
    # gradient and intercecpt of line passing through centroids
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    return m

# This function generates the chaincode 
# for transition between two neighbour points
def getChainCode(x1, y1, x2, y2):
    m = m_between_points(x1, y1, x2, y2)
    ang = np.pi/2 - np.arctan(m)

    i = min(range(len(codeList)), key=lambda i: abs(codeList[i]-ang))
    if i == 8:
        i = 0 # list wraps around
    return i


def chaincode(p, min_length=5, reduce_length=1):
    #assert min_length <= reduce_length, "reduce_length must be smaller than min_length"
    min_y = round(min(p))

    p = [round(i)-min_y for i in p]

    chaincodes = []
    for i in range(len(p)):
        if i == 0:
            continue
        
        x2 = i*2
        y2 = p[i]

        x1 = (i-1)*2
        y1 = p[(i-1)]
        
        chaincodes.append(getChainCode(x1, y1, x2, y2))

    ml = min_length if min_length else 0

    cc1 = []
    count = 1
    for i in range(1, len(chaincodes)):
        this = chaincodes[i]
        that = chaincodes[i-1]
        
        if this == that:
            count += 1
        else:
            if count >= ml:
                cc1 += chaincodes[i-count:i]
            count = 1
    
    if count >= ml:
        cc1 += chaincodes[i-count+1:i+1]

    count = 1
    cc2 = []
    if reduce_length:
        for i in range(1, len(cc1)):
            this = cc1[i]
            that = cc1[i-1]
            
            if this == that:
                count += 1
            else:
                if count > reduce_length:
                    cc2 += cc1[i-reduce_length:i]
                else:
                    cc2 += cc1[i-count+1:i+1]
                count = 1
        
        if count > reduce_length:
            cc2 += cc1[i-reduce_length:i]
        else:
            cc2 += cc1[i-count+1:i+1]

    return cc2
