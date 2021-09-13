import numpy as np
from skimage import measure
from skimage import morphology, util


def BBoxCalc(tempobject):
    """
    Given binary image of object, returns coordinates of bounding box with
    buffer zone
    input=binary image of object
    output=coordinates for cropped version with buffer
    """
    sMask = tempobject.shape
    properties = measure.regionprops(
        tempobject
    )  # ideally find a way to only measure bounding box for speed?
    BBox = properties[0].bbox
    buffer = 7
    BBox = np.array(BBox)
    BBox[0] = round(BBox[0] - buffer)
    BBox[1] = round(BBox[1] - buffer)
    BBox[2] = round(BBox[2] + buffer)
    BBox[3] = round(BBox[3] + buffer)
    BBox[BBox < 0] = 0
    if BBox[2] > sMask[0]:
        BBox[2] = sMask[0]
    if BBox[3] > sMask[1]:
        BBox[3] = sMask[1]
    return BBox


def imimposemin(I, BW, conn=2):
    if I.size == 0:
        return I
    fm = I
    fm = np.uint8(255 * util.invert(BW))
    if not np.issubdtype(I.dtype, np.integer):
        range = float(np.amax(I)) - float(np.amin(I))
        if range == 0:
            h = 0.1
        else:
            h = range * 0.001
    else:
        h = 1
    fp1 = I + h
    g = np.minimum(fp1, fm)
    if conn == 2:
        J = morphology.reconstruction(
            util.invert(fm), util.invert(g), selem=morphology.square(3)
        )
    if conn == 1:
        J = morphology.reconstruction(
            util.invert(fm), util.invert(g), selem=morphology.disk(1)
        )
    J = util.invert(np.uint8(J))
    return J
