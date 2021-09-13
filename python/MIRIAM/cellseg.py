import numpy as np
import math
from . import bwmorph
import skfmm
from .common import BBoxCalc, imimposemin
from skimage import io, morphology, exposure, filters, measure, util, segmentation
from numpy import ndarray
from scipy import ndimage, spatial, signal
from operator import itemgetter


def bwperim(mask):
    # Output perimeter of object
    se = morphology.disk(1)
    b = np.pad(mask, (1, 1), "constant")
    b_eroded = morphology.binary_erosion(b, se)
    p = b ^ b_eroded  # bitwise xor
    # shape = p.shape  # unused
    p = p[1 : p.shape[0] - 1, 1 : p.shape[1] - 1]
    return p


def connectpoints(skel):
    """
    Connects inner line segments

    Parameters
    ----------
    skel : multiple line segments

    Returns
    -------
    L3 : connected line segments

    """
    L2 = measure.label(skel)
    s = np.max(L2)
    L3 = np.zeros(L2.shape)
    for i in range(1, s + 1):
        temp = selectobjects(i, L2)
        eptemp = bwmorph.endpoints(np.array(temp, dtype=np.bool))
        if np.max(eptemp) != 0:
            L3 = L3 + temp
    L2 = measure.label(L3)
    endpoints = labelendpoints(L2)
    properties = measure.regionprops(L2)
    skelarea = [prop.area for prop in properties]
    if len(skelarea) > 0:
        minimum = min(skelarea)
    else:
        minimum = 0
    rc = len(skelarea)
    skelarea = [(index, val) for index, val in enumerate(skelarea)]

    while rc > 1 and minimum < 40 and findDistanceEndpoint(L2) < 13:
        bb = 0
        DI = sorted(skelarea, key=itemgetter(1))
        i = DI[bb][0]
        (xnew, ynew, d) = findClosestEndpoint(i, endpoints)
        while d > 13:
            bb += 1
            i = DI[bb][0]
            (xnew, ynew, d) = findClosestEndpoint(i, endpoints)
        newimg = np.zeros(L3.shape)
        index = np.unique(
            np.ravel_multi_index(
                (
                    np.array(np.round(xnew), dtype=np.int64),
                    np.array(np.round(ynew), dtype=np.int64),
                ),
                newimg.shape,
                order="F",
            )
        )
        index_coords = np.unravel_index(index, newimg.shape, order="F")
        newimg[index_coords] = 1
        newimg = np.array(newimg, dtype=bool)
        L3 = L3 > 0
        L3 = L3 | newimg
        L2 = measure.label(L3)
        endpoints = labelendpoints(L2)
        props = measure.regionprops(L2)
        skelarea = [prop.area for prop in props]
        if len(skelarea) > 0:
            minimum = min(skelarea)
        else:
            minimum = 0
        skelarea = [(index, val) for index, val in enumerate(skelarea)]
        rc = len(skelarea)
    return L3


def extend2(obj, backdrop, hlen):
    """
    Extend line segments at both endpoints

    Parameters
    ----------
    obj : input skeleton image
    backdrop : cell object
    hlen : length of extension

    Returns
    -------
    input_skeleton_image : extended lines

    """
    trimBy = 6

    # image with just edge of cell
    e = bwperim(backdrop)
    object2 = morphology.thin(obj)

    # find endpoints
    ep = bwmorph.endpoints(object2)

    # create frame
    totalcanvas = np.zeros(backdrop.shape)
    (br, bc) = totalcanvas.shape
    totalcanvas[0:2, :] = 1
    totalcanvas[br - 2 : br, :] = 1
    totalcanvas[:, 0:2] = 1
    totalcanvas[:, bc - 2 : bc] = 1

    # if endpoints on edge of frame, subtract
    ep = ep - totalcanvas
    ep = ep > 0
    interior = obj - e
    interior = interior > 0

    # find objects that contains endpoints with object > 1 pixel and their locations
    L = measure.label(interior)
    number = np.amax(L) + 1
    temparray = []
    endpointarray = []
    for i in range(1, number):
        temp = L == i
        k = ep & temp
        if np.max(k) > 0 and np.sum(temp) > 1:
            temparray.append(temp)
            endpointarray.append(k)
    col = len(temparray)
    storeimage = np.zeros(obj.shape)
    if col == 0:
        return obj
    for j in range(0, col):
        test = temparray[j]
        etest = endpointarray[j]
        test = morphology.thin(test)
        (r, _) = test.shape
        length = 1000
        etest = bwmorph.endpoints(test)
        etest = (etest - totalcanvas) > 0
        # (y, x) = np.nonzero(etest)  # unused
        trimmed = bwmorph.spur(test, trimBy)
        trimSeg = test * (test ^ trimmed)
        # new2 = trimSeg

        # locate branchpoints
        new21 = morphology.skeletonize(trimSeg)
        bp = bwmorph.branchpoints(new21)
        bp = morphology.binary_dilation(bp, morphology.disk(1))

        # get rid of branch points
        new2 = new21 * (new21 ^ bp)
        trimL = measure.label(new2)
        num = np.max(trimL)
        retrim = np.zeros(obj.shape, dtype=bool)
        for d in range(1, num + 1):
            tempT = trimL == d
            kT = etest & tempT
            (yK, _) = np.nonzero(kT)
            a = yK.size
            if a > 1:
                tempT = morphology.thin(tempT)
                trimmed2 = bwmorph.spur(tempT, trimBy + 1)
                trimSeg2 = tempT * (tempT ^ trimmed2)
            else:
                trimSeg2 = tempT
            retrim = retrim + trimSeg2
        retrimL = measure.label(retrim)
        num = np.max(retrimL)
        tempendT = np.empty(
            (0, 3)
        )  # vertical array of index, tempT, and kT (change to list for speed)
        for k in range(1, num + 1):
            tempT = retrimL == k
            kT = etest & tempT
            if np.max(kT) > 0 and np.sum(tempT) > 1:
                tempendT = np.vstack((tempendT, [k, tempT, kT]))
        for row in tempendT:
            testT = row[1]
            # etestT = row[2]  # unused
            testT = morphology.thin(testT)
            # (rT, cT) = testT.shape  # unused

            # find location of endpoints
            endpointsT = bwmorph.endpoints(testT)
            (yy, xx) = np.nonzero(endpointsT)

            # connect endpoints
            distmax = np.vstack((xx, yy))  #  x coords on first row y on second row
            distmax = ndarray.transpose(
                distmax
            )  #  is it easier to transpose twice and stack or stack and tranpose a bigger array?
            distancematrix = spatial.distance.squareform(
                spatial.distance.pdist(distmax)
            )
            (one, _) = np.argwhere(distancematrix == np.amax(distancematrix))
            i1 = one[0]
            i2 = one[1]
            new3 = np.zeros(obj.shape)
            ylin = np.linspace(yy[i1], yy[i2], length)
            xlin = np.linspace(xx[i1], xx[i2], length)
            index = np.unique(
                np.ravel_multi_index(
                    (
                        np.array(np.round(ylin), dtype=np.int64),
                        np.array(np.round(xlin), dtype=np.int64),
                    ),
                    obj.shape,
                    order="F",
                )
            )
            index_coord = np.unravel_index(index, new3.shape, order="F")
            new3[index_coord] = 1

            L = measure.label(new3)
            properties = measure.regionprops(L)
            props = [(prop.orientation) for prop in properties]
            O = np.mean(props)
            if O != math.pi / 4 and O != -math.pi / 4:
                if O < 0:
                    O = math.pi / 2 + O
                else:
                    O = O - math.pi / 2
            cosOrient = math.cos(O)
            sinOrient = math.sin(O)
            yx_ep = np.argwhere(bwmorph.endpoints(new3) == 1)
            largey = np.max(yx_ep[:, 0])
            smally = np.min(yx_ep[:, 0])
            largex = np.max(yx_ep[:, 1])
            smallx = np.min(yx_ep[:, 1])
            # y = yx[:,0]
            # x = yx[:,1]
            # largey = np.min(np.argwhere(y == np.max(y))) #the coordinates seem to be sorted by size...
            # smally = np.min(np.argwhere(y == np.min(y)))
            # largex = np.min(np.argwhere(x == np.max(x)))
            # smallx = np.min(np.argwhere(x == np.min(x)))

            if largex - smallx == 0:
                xt = [largex, largex]
                yt = [largey + hlen, smally - hlen]
                largey = np.min(np.argwhere(yt == max(yt)))
                smally = np.min(np.argwhere(yt == min(yt)))
                if yt[largey] > r:
                    yt[largey] = r
                if yt[smally] < 1:
                    yt[smally] = 1
            else:
                slope = (yx_ep[1, 0] - yx_ep[0, 0]) / (yx_ep[1, 1] - yx_ep[0, 1])
                if slope >= 0:
                    xcoords = largex + hlen * np.array([0, cosOrient])
                    ycoords = largey - hlen * np.array([0, sinOrient])
                    xcoords2 = smallx - hlen * np.array([0, cosOrient])
                    ycoords2 = smally + hlen * np.array([0, sinOrient])

                    yt = (ycoords[1], ycoords2[1])  # 1 out of bounds
                    xt = (xcoords[1], xcoords2[1])

                    # largey = np.argwhere(yt == max(yt))
                    # smally = np.argwhere(yt == min(yt))
                    # largex = np.argwhere(yt == max(xt))
                    # smallx = np.argwhere(yt == min(xt))
                if slope < 0:
                    xcoords = largex + hlen * np.array([0, cosOrient])
                    ycoords = smally - hlen * np.array([0, sinOrient])

                    xcoords2 = smallx - hlen * np.array([0, cosOrient])
                    ycoords2 = largey + hlen * np.array([0, sinOrient])

                    yt = (ycoords[1], ycoords2[1])
                    xt = (xcoords[1], xcoords2[1])

                    # largey = np.argwhere(yt == max(yt))
                    # smally = np.argwhere(yt == min(yt))
                    # largex = np.argwhere(yt == max(xt))
                    # smallx = np.argwhere(yt == min(xt))
            ynew = np.linspace(yt[0], yt[1], length)
            xnew = np.linspace(xt[0], xt[1], length)
            ynew = np.round(ynew)
            xnew = np.round(xnew)
            (ccc, rrr) = obj.shape
            greatx = np.nonzero(xnew >= 0)
            xnew = xnew[greatx]
            ynew = ynew[greatx]
            greatx = np.nonzero(xnew < rrr)
            xnew = xnew[greatx]
            ynew = ynew[greatx]
            greatx = np.nonzero(ynew >= 0)
            xnew = xnew[greatx]
            ynew = ynew[greatx]
            greatx = np.nonzero(ynew < ccc)
            xnew = xnew[greatx]
            ynew = ynew[greatx]

            newimg = np.zeros(obj.shape)
            index = np.unique(
                np.ravel_multi_index(
                    (
                        np.array(np.round(ynew), dtype=np.int64),
                        np.array(np.round(xnew), dtype=np.int64),
                    ),
                    obj.shape,
                    order="F",
                )
            )
            index_coord = np.unravel_index(index, newimg.shape, order="F")
            newimg[index_coord] = 1

            newimg = morphology.thin(newimg)
            ep2 = bwmorph.endpoints(newimg)
            epx = ep2
            ep2 = ep2 - totalcanvas
            ep2 = ep2 > 0
            epObject = testT & etest
            xy = np.argwhere(epObject == 1)

            ind = np.argwhere(ep2 == 1)
            if ind.size == 0:
                ep2 = epx
            x1y1 = ind

            th_ends = np.vstack((xy, x1y1))
            distancematrix = spatial.distance.squareform(
                spatial.distance.pdist(th_ends)
            )
            minimum = min([elem for elem in distancematrix[0, :] if elem != 0])
            endpoint = np.argwhere(distancematrix[0, :] == minimum)
            endpoint = endpoint[0][0]

            newimg = np.zeros(obj.shape)
            ynew = np.linspace(x1y1[endpoint - 1, 1], xy[0][1], length)
            xnew = np.linspace(x1y1[endpoint - 1, 0], xy[0][0], length)
            index = np.unique(
                np.ravel_multi_index(
                    (
                        np.array(np.round(xnew), dtype=np.int64),
                        np.array(np.round(ynew), dtype=np.int64),
                    ),
                    obj.shape,
                    order="F",
                )
            )
            index_coord = np.unravel_index(index, newimg.shape, order="F")
            newimg[index_coord] = 1

            before = morphology.thin(interior)
            intersection_pts_b = find_skel_intersection(before)
            # intersection_pts_b = np.argwhere(bwmorph.branchpoints(before) == 1)
            sizeb = intersection_pts_b.shape[0]

            interior_temp = interior | np.array(newimg, dtype=np.int16)

            intersecttest = morphology.thin(interior_temp)
            intersection_pts = find_skel_intersection(intersecttest)
            # intersection_pts = np.argwhere(bwmorph.branchpoints(intersecttest) == 1)
            sizef = intersection_pts.shape[0]
            if sizef > sizeb:
                if sizeb == 0:
                    intersect_point = intersection_pts
                else:
                    seta = {tuple(rowa) for rowa in intersection_pts}
                    setb = {tuple(rowb) for rowb in intersection_pts_b}
                    intersect_point = seta.symmetric_difference(setb)
                    intersect_point = np.asarray(list(intersect_point))
                intsize = intersect_point.size
                if intsize > 2:
                    dmat = np.asarray(
                        [
                            (intersect_point[1][0], intersect_point[0][0]),
                            (xy[0][0], xy[0][1]),
                            x1y1[endpoint - 1],
                        ]
                    )
                else:
                    dmat = np.asarray(
                        [
                            (intersect_point[0][1], intersect_point[0][0]),
                            (xy[0][0], xy[0][1]),
                            x1y1[endpoint - 1],
                        ]
                    )
                # dmat very strange in matlab: if matrix is 2 dimensional dmat
                # takes first and second x value for a point instead of first
                # x,y
                distance_s = spatial.distance.pdist(dmat, "euclidean")
                if np.min(distance_s) > 2:
                    newimg = np.zeros(interior.shape)
                    epObject = testT & etest
                    xy = np.argwhere(epObject == 1)  # only 1 point should be found
                    if intsize > 2:
                        addon = np.flip(intersect_point)
                        th_ends = np.vstack((xy[0], addon))
                        # th_ends = np.asarray([xy[0], (intersect_point[:,1],intersect_point[:,0])])
                    else:
                        th_ends = np.asarray(
                            [xy[0], (intersect_point[0][1], intersect_point[0][0])]
                        )
                    distance_matrix = spatial.distance.squareform(
                        spatial.distance.pdist(th_ends)
                    )
                    minimum = min([elem for elem in distance_matrix[0, :] if elem != 0])
                    endpoint = np.argwhere(distance_matrix[0, :] == minimum)
                    endpoint = endpoint[0][0]
                    ynew = np.linspace(
                        intersect_point[endpoint - 1, 1], xy[0][1], length
                    )
                    xnew = np.linspace(
                        intersect_point[endpoint - 1, 0], xy[0][0], length
                    )
                    index = np.unique(
                        np.ravel_multi_index(
                            (
                                np.array(np.round(xnew), dtype=np.int64),
                                np.array(np.round(ynew), dtype=np.int64),
                            ),
                            interior.shape,
                            order="F",
                        )
                    )
                    index_coord = np.unravel_index(index, newimg.shape, order="F")
                    newimg[index_coord] = 1
                    interior = interior | np.array(newimg, dtype=np.int16)
                else:
                    interior = interior_temp
            else:
                interior = interior_temp
            storeimage = np.array(storeimage, dtype=np.int16) | np.array(
                newimg, dtype=np.int16
            )
    # outline = storeimage | obj
    outline = storeimage + obj
    outline = outline > 0
    final = np.multiply(outline, backdrop)
    selem = morphology.disk(1)
    e = morphology.binary_dilation(e, selem)
    final = final - e
    input_skeleton_image = morphology.thin(final > 0)
    return input_skeleton_image


def finalconnect(lines):
    """
    Connect any outstanding endpoint pairs.

    Parameters
    ----------
    lines : membrane lines
    Returns
    -------
    out : connected membrane

    """
    lines = morphology.thin(lines)
    eptemp = bwmorph.endpoints(lines)
    final = lines
    out = final
    if np.sum(eptemp) != 0:
        xy = np.argwhere(eptemp == 1)
        x = xy[:, 0]
        y = xy[:, 1]
        rr = x.size
        while rr > 1:
            distance = spatial.distance.pdist(xy)
            distance[distance == 0] = 999
            dsearch = np.min(distance)
            distance = spatial.distance.squareform(distance)
            distance[distance == 0] = 999
            x1x2 = np.argwhere(distance == dsearch)
            x1 = x1x2[:, 0]
            y1 = x1x2[:, 1]
            x11 = x1[0]
            x22 = y1[0]
            newimg = np.zeros(eptemp.shape)
            x1 = x[x11]
            y1 = y[x11]

            x2 = x[x22]
            y2 = y[x22]

            xnew = np.linspace(x1, x2, 1000)
            ynew = np.linspace(y1, y2, 1000)
            index = np.ravel_multi_index(
                (
                    np.array(np.round(xnew), dtype=np.int64),
                    np.array(np.round(ynew), dtype=np.int64),
                ),
                newimg.shape,
                order="F",
            )
            index = np.unravel_index(index, newimg.shape, order="F")
            newimg[index] = 1
            before = morphology.thin(final)
            intersection_pts_b = find_skel_intersection(before)
            # intersection_pts_b = bwmorph.branchpoints(before)
            sizeb = intersection_pts_b.shape[0]
            final = (newimg + final) > 0
            intersecttest = morphology.thin(final)
            intersection_pts = find_skel_intersection(intersecttest)
            # intersection_pts = bwmorph.branchpoints(intersecttest)
            sizef = intersection_pts.shape[0]
            if sizef > sizeb:
                if sizeb == 0:
                    intersect_point = intersection_pts
                else:
                    seta = {tuple(rowa) for rowa in intersection_pts}
                    setb = {tuple(rowb) for rowb in intersection_pts_b}
                    intersect_point = seta.symmetric_difference(setb)
                    intersect_point = np.asarray(list(intersect_point))
                intsize = intersect_point.size
                if intsize > 2:
                    dmat = np.asarray(
                        [
                            (intersect_point[1][0], intersect_point[0][0]),
                            (x1, y1),
                            (x2, y2),
                        ]
                    )
                else:
                    dmat = np.asarray(
                        [
                            (intersect_point[0][1], intersect_point[0][0]),
                            (x1, y1),
                            (x2, y2),
                        ]
                    )
                # dmat = np.asarray([(intersect_point[0][1], intersect_point[0][0]), (x1,y1), (x2,y2)])
                distance_s = spatial.distance.pdist(dmat, "euclidean")
                if np.min(distance_s) > 2:
                    final = before
            x = np.delete(x, x22)
            y = np.delete(y, x22)
            x = np.delete(x, x11)
            y = np.delete(y, x11)
            xy = np.transpose(np.array((x, y)))
            rr = x.size
        out = morphology.thin(final)
        return out
    else:
        return out


def finalconnect_2(obj, backdrop, hlen):
    """
    Extend from spurs and connect line segments

    Parameters
    ----------
    obj : membrane lines
    backdrop : cell object
    hlen : length of extension

    Returns
    -------
    final : extended and connected lines

    """
    obj = morphology.thin(obj)
    ep = bwmorph.endpoints(obj)
    totalcanvas = np.zeros(backdrop.shape)
    (br, bc) = totalcanvas.shape
    totalcanvas[0:2, :] = 1
    totalcanvas[br - 2 : br, :] = 1
    totalcanvas[:, 0:2] = 1
    totalcanvas[:, bc - 2 : bc] = 1
    ep = ep - totalcanvas
    ep = ep > 0
    new21 = morphology.skeletonize(obj)
    bp = bwmorph.branchpoints(new21)
    bp = morphology.binary_dilation(bp, morphology.disk(1))
    new2 = obj * (obj ^ bp)
    e = bwperim(backdrop)
    ed = morphology.binary_dilation(e, morphology.disk(4))
    new2 = new2 * (new2 ^ ed)
    new2 = new2 > 0
    interior = new2[:, :]
    L = measure.label(new2)
    number = np.max(L) + 1
    temparray = []
    endpointarray = []
    for i in range(1, number):
        temp = L == i
        temp = morphology.thin(temp)
        remove = bwmorph.spur(temp, 5)
        temp = temp * (temp ^ remove)
        k = ep & temp
        if np.amax(k) > 0 and np.sum(temp) > 1:
            temparray.append(temp)
            endpointarray.append(k)
    col = len(temparray)
    storeimage = np.zeros(obj.shape)
    length = 1000
    for j in range(0, col):
        test = temparray[j]
        etest = endpointarray[j]
        test = morphology.thin(test)
        (r, _) = test.shape
        new3 = morphology.thin(test)
        endPoints = bwmorph.endpoints(new3)
        yyxx = np.argwhere(endPoints == 1)
        distancematrix = spatial.distance.squareform(spatial.distance.pdist(yyxx))
        ab = np.argwhere(distancematrix == np.max(distancematrix))
        i1 = ab[0][0]
        i2 = ab[0][1]
        new3 = np.zeros(obj.shape)
        ylin = np.linspace(yyxx[i1, 0], yyxx[i2, 0], length)
        xlin = np.linspace(yyxx[i1, 1], yyxx[i2, 1], length)
        index = np.ravel_multi_index(
            (
                np.array(np.round(ylin), dtype=np.int64),
                np.array(np.round(xlin), dtype=np.int64),
            ),
            obj.shape,
            order="F",
        )
        index_coord = np.unravel_index(index, new3.shape, order="F")
        new3[index_coord] = 1
        L = measure.label(new3)
        # thinner = morphology.thin(L)  # unused
        # endPoints2 = bwmorph.endpoints(thinner)  # unused
        properties = measure.regionprops(L)
        props = [(prop.orientation) for prop in properties]
        O = np.mean(props)
        if O != math.pi / 4 and O != -math.pi / 4:
            if O < 0:
                O = math.pi / 2 + O
            else:
                O = O - math.pi / 2
        cosOrient = math.cos(O)
        sinOrient = math.sin(O)
        yx_ep = np.argwhere(bwmorph.endpoints(new3) == 1)
        largey = np.max(yx_ep[:, 0])
        smally = np.min(yx_ep[:, 0])
        largex = np.max(yx_ep[:, 1])
        smallx = np.min(yx_ep[:, 1])
        if largex - smallx == 0:
            xt = [largex, largex]
            yt = [largey + hlen, smally - hlen]
            largey = np.min(np.argwhere(yt == max(yt)))
            smally = np.min(np.argwhere(yt == min(yt)))
            if yt[largey] > r:
                yt[largey] = r
            if yt[smally] < 1:
                yt[smally] = 1
        else:
            slope = (yx_ep[1, 0] - yx_ep[0, 0]) / (yx_ep[1, 1] - yx_ep[0, 1])
            if slope >= 0:
                xcoords = largex + hlen * np.array([0, cosOrient])
                ycoords = largey - hlen * np.array([0, sinOrient])
                xcoords2 = smallx - hlen * np.array([0, cosOrient])
                ycoords2 = smally + hlen * np.array([0, sinOrient])

                yt = (ycoords[1], ycoords2[1])  # 1 out of bounds
                xt = (xcoords[1], xcoords2[1])

            if slope < 0:
                xcoords = largex + hlen * np.array([0, cosOrient])
                ycoords = smally - hlen * np.array([0, sinOrient])

                xcoords2 = smallx - hlen * np.array([0, cosOrient])
                ycoords2 = largey + hlen * np.array([0, sinOrient])

                yt = (ycoords[1], ycoords2[1])
                xt = (xcoords[1], xcoords2[1])

        ynew = np.linspace(yt[0], yt[1], length)
        xnew = np.linspace(xt[0], xt[1], length)
        ynew = np.round(ynew)
        xnew = np.round(xnew)
        (ccc, rrr) = obj.shape
        greatx = np.nonzero(xnew >= 0)
        xnew = xnew[greatx]
        ynew = ynew[greatx]
        greatx = np.nonzero(xnew < rrr)
        xnew = xnew[greatx]
        ynew = ynew[greatx]
        greatx = np.nonzero(ynew >= 0)
        xnew = xnew[greatx]
        ynew = ynew[greatx]
        greatx = np.nonzero(ynew < ccc)
        xnew = xnew[greatx]
        ynew = ynew[greatx]

        newimg = np.zeros(obj.shape)
        index = np.ravel_multi_index(
            (
                np.array(np.round(ynew), dtype=np.int64),
                np.array(np.round(xnew), dtype=np.int64),
            ),
            obj.shape,
            order="F",
        )
        index_coord = np.unravel_index(index, newimg.shape, order="F")
        newimg[index_coord] = 1

        newimg = morphology.thin(newimg)
        ep2 = bwmorph.endpoints(newimg)
        epx = ep2
        ep2 = ep2 - totalcanvas
        ep2 = ep2 > 0
        xy = np.argwhere(etest == 1)
        ind = np.argwhere(ep2 == 1)
        if ind.size == 0:
            ep2 = epx
        x1y1 = np.argwhere(ep2 == 1)
        th_ends = np.vstack((xy, x1y1))
        distancematrix = spatial.distance.squareform(spatial.distance.pdist(th_ends))
        # (one, two) = np.argwhere(distancematrix == np.max(distancematrix))  # unused?
        ind = np.argwhere(ep2 == 1)
        if ind.size == 0:
            ep2 = epx

        x1y1 = np.argwhere(ep2 == 1)
        th_ends = np.vstack((xy, x1y1))
        distancematrix = spatial.distance.squareform(spatial.distance.pdist(th_ends))
        minimum = min([elem for elem in distancematrix[0, :] if elem != 0])
        endpoint = np.argwhere(distancematrix[0, :] == minimum)
        endpoint = endpoint[0][0]
        newimg = np.zeros(obj.shape)
        ynew = np.linspace(x1y1[endpoint - 1, 1], xy[0][1], length)
        xnew = np.linspace(x1y1[endpoint - 1, 0], xy[0][0], length)
        index = np.unique(
            np.ravel_multi_index(
                (
                    np.array(np.round(xnew), dtype=np.int64),
                    np.array(np.round(ynew), dtype=np.int64),
                ),
                obj.shape,
                order="F",
            )
        )
        index_coord = np.unravel_index(index, newimg.shape, order="F")
        newimg[index_coord] = 1

        before = morphology.thin(interior)
        intersection_pts_b = find_skel_intersection(before)
        sizeb = intersection_pts_b.shape[0]

        interior_temp = interior | np.array(newimg, dtype="int64")

        intersecttest = morphology.thin(interior_temp)
        intersection_pts = find_skel_intersection(intersecttest)
        sizef = intersection_pts.shape[0]
        if sizef > sizeb:
            if sizeb == 0:
                intersect_point = intersection_pts
            else:
                seta = {tuple(rowa) for rowa in intersection_pts}
                setb = {tuple(rowb) for rowb in intersection_pts_b}
                intersect_point = seta.symmetric_difference(setb)
                intersect_point = np.asarray(list(intersect_point))
            intsize = intersect_point.size
            if intsize > 2:
                dmat = np.asarray(
                    [
                        (intersect_point[1][0], intersect_point[0][0]),
                        (xy[0][0], xy[0][1]),
                        x1y1[endpoint - 1],
                    ]
                )
            else:
                dmat = np.asarray(
                    [
                        (intersect_point[0][1], intersect_point[0][0]),
                        (xy[0][0], xy[0][1]),
                        x1y1[endpoint - 1],
                    ]
                )

            distance_s = spatial.distance.pdist(dmat, "euclidean")
            if np.min(distance_s) > 2:
                newimg = np.zeros(new2.shape)
                xy = np.argwhere(etest == 1)
                if intsize > 2:
                    addon = np.flip(intersect_point)
                    th_ends = np.vstack((xy[0], addon))
                    # th_ends = np.asarray([xy[0], (intersect_point[:,1],intersect_point[:,0])])
                else:
                    th_ends = np.asarray(
                        [xy[0], (intersect_point[0][1], intersect_point[0][0])]
                    )
                distance_matrix = spatial.distance.squareform(
                    spatial.distance.pdist(th_ends)
                )
                minimum = min([elem for elem in distance_matrix[0, :] if elem != 0])
                endpoint = np.argwhere(distance_matrix[0, :] == minimum)
                endpoint = endpoint[0][0]
                ynew = np.linspace(intersect_point[endpoint - 1, 1], xy[0][1], length)
                xnew = np.linspace(intersect_point[endpoint - 1, 0], xy[0][0], length)
                index = np.unique(
                    np.ravel_multi_index(
                        (
                            np.array(np.round(xnew), dtype=np.int64),
                            np.array(np.round(ynew), dtype=np.int64),
                        ),
                        interior.shape,
                        order="F",
                    )
                )
                index_coord = np.unravel_index(index, interior.shape, order="F")
                newimg[index_coord] = 1
                interior = interior | np.array(newimg, dtype="int64")
            else:
                interior = interior_temp
        else:
            interior = interior_temp
        storeimage = np.array(storeimage, dtype="int64") | np.array(
            newimg, dtype="int64"
        )
    final = storeimage + obj
    return final > 0


def findClosestEndpoint(index, endpoints):
    """
    Find and connect endpoints closest to ones in index segment

    Parameters
    ----------
    index : segment of interest
    endpoints : list of endpoints

    Returns
    -------
    xnew : x-coords of line connecting endpoints
    ynew : y-coords of line conencting endpoints
    mindi : distance of closest endpoint

    """
    reference = endpoints[index]
    rc = reference.shape
    # r1c1 = len(endpoints)  # unused?
    distance2 = np.zeros((rc[0], len(endpoints)))
    index2 = np.zeros((rc[0], len(endpoints)))
    for i in range(rc[0]):
        for j in range(len(endpoints)):
            distance = []
            test = endpoints[j]
            r2c2 = test.shape
            for k in range(r2c2[0]):
                temp = np.array([reference[i], test[k]])
                distance.append(spatial.distance.pdist(temp)[0])
            distance2[i, j] = min(distance)
            minindex = distance.index(min(distance))
            index2[i, j] = minindex
    nonzeromin = np.min(distance2[np.nonzero(distance2)])
    mn1 = np.argwhere(distance2 == nonzeromin)[0]
    m1 = mn1[0]
    m2 = mn1[1]
    linepoint2 = index2[m1, m2]
    line2 = endpoints[m2]
    point = reference[m1, :]
    point2 = line2[int(linepoint2), :]
    xnew = np.linspace(point[0], point2[0], 1000)
    ynew = np.linspace(point[1], point2[1], 1000)
    mindi = distance2[mn1[0], mn1[1]]
    return (xnew, ynew, mindi)


def findDistanceEndpoint(mask):
    """
    Find the min distance between all endpoints and line segments

    Parameters
    ----------
    mask : mask of line segments

    Returns
    -------
    output : min distance

    """
    mask = measure.label(mask)
    distance = []
    number = np.max(mask)
    endpoints = labelendpoints(mask)
    for i in range(number):
        (_, _, d) = findClosestEndpoint(i, endpoints)
        distance.append(d)
    distance = sorted(distance)
    output = distance[0]
    return output


def find_skel_intersection(input_skeleton_image):
    """
    Find intersection points in a given skeleton image

    Parameters
    ----------
    input_skeleton_image : binary skeleton image

    Returns
    -------
    intersecting_pts : (x,y) coordinates of intersection points

    """
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    conv_img = signal.convolve2d(input_skeleton_image, kernel, mode="same")
    conv_img = np.multiply(conv_img, input_skeleton_image)
    yx = np.argwhere(conv_img > 3)
    intersecting_pts = []
    if yx.size > 0:
        classes = sortclasses(yx, 1, 8)
        for group in classes:
            if group.ndim == 1:
                intersecting_pts.append(group)
            else:
                X = np.mean(group[:, 0])
                Y = np.mean(group[:, 1])
                mean_point = np.array([X, Y])
                minimum = float(0)
                point = np.empty((0, 0))
                for row in group:
                    temp = spatial.distance.euclidean(mean_point, row)
                    if temp > minimum:
                        minimum = temp
                        point = row
                if point.size != 0:
                    intersecting_pts.append(point)
    intersecting_pts = np.asarray(intersecting_pts)
    return intersecting_pts


def imextendedmin(I, h, conn=2):
    # Computes extended minima transform
    BW = morphology.local_minima(imhmin(I, h), connectivity=conn)
    return BW


def imhmax(I, H, conn=2):
    # h-maxima transform
    if conn == 1:
        J = morphology.reconstruction((I - H), I, selem=np.array(morphology.disk(1)))
    if conn == 2:
        J = morphology.reconstruction((I - H), I, selem=morphology.square(3))
    return J


def imhmin(I, H, conn=2):
    # H-minima transform
    I = util.invert(I)
    I = imhmax(I, H, conn)
    I = util.invert(I)
    return I


def intersect_k(line, e):
    """
    Intersect inner lines to edge

    Parameters
    ----------
    line : line segments
    e : edge

    Returns
    -------
    out : line segments conencted to edge

    """
    line2 = line
    L3 = measure.label(line)
    for i in range(1, np.max(L3) + 1):
        line = selectobjects(i, L3)
        inpt = morphology.thin(line)
        endpoints = labelendpoints(inpt)
        endpoints = endpoints[0]
        rc = endpoints.shape
        (x, y) = np.nonzero(e)
        for j in range(rc[0]):
            newimg = np.zeros(line.shape)
            distance = []
            for k in range(x.size):
                distance.append(
                    spatial.distance.pdist(np.asarray([endpoints[j, :], (x[k], y[k])]))[
                        0
                    ]
                )
            minimum = min(distance)
            index2 = distance.index(minimum)
            x1 = endpoints[j, 0]
            y1 = endpoints[j, 1]
            x2 = x[index2]
            y2 = y[index2]
            xnew = np.linspace(x1, x2, 1000)
            ynew = np.linspace(y1, y2, 1000)
            index = np.ravel_multi_index(
                (
                    np.array(np.round(xnew), dtype=np.int64),
                    np.array(np.round(ynew), dtype=np.int64),
                ),
                newimg.shape,
                order="F",
            )
            index_coord = np.unravel_index(index, newimg.shape, order="F")
            newimg[index_coord] = 1
            line2 = line2 + newimg
    out = e + line2
    return out


def intersectfirst(line, mask):
    """
    Connect line segments to edge

    Parameters
    ----------
    line : line segments
    mask : mask of object

    Returns
    -------
    remainder : remaining segments not connected
    edge : new edge

    """
    if np.max(line) == 0:
        remainder = np.zeros(line.shape)
        edge = remainder
        return (remainder, edge)
    line2 = np.zeros(line.shape)
    remainder = morphology.thin(line)
    L3 = measure.label(line)
    e = bwperim(mask)  # originally sobel filter, but sobel filter is non binary
    properties = measure.regionprops(mask)
    BBox = [prop.bbox for prop in properties]
    BBox = [value for value in BBox[0]]
    sCanvas = line.shape
    buffer = 0
    BBox[0] = round(BBox[0] - buffer)
    BBox[1] = round(BBox[1] - buffer)
    BBox[2] = round(BBox[0] + BBox[2] + 2 * buffer)
    BBox[3] = round(BBox[1] + BBox[3] + 2 * buffer)
    for val in BBox:
        if val < 0:
            val = 0
    if BBox[2] > sCanvas[0]:
        BBox[2] = sCanvas[0]
    if BBox[3] > sCanvas[1]:
        BBox[3] = sCanvas[1]
    totalcanvas = np.zeros(sCanvas)

    totalcanvas[BBox[0] : BBox[2] - 1, BBox[1]] = 1
    totalcanvas[BBox[0] : BBox[2] - 1, BBox[3] - 1] = 1
    totalcanvas[BBox[0], BBox[1] : BBox[3] - 1] = 1
    totalcanvas[BBox[2] - 1, BBox[1] : BBox[3] - 1] = 1

    for i in range(1, np.max(L3) + 1):
        line = L3 == i
        inpt = morphology.thin(line)
        endpoints = labelendpoints(inpt)
        points = endpoints[0]
        row = points.shape[0]
        xy = np.nonzero(e)
        inmask = mask
        if np.max(inmask) > 0:
            x2y2 = np.nonzero(totalcanvas)
        else:
            x2y2 = ([], [])
        x = np.append(xy[0], x2y2[0])
        y = np.append(xy[1], x2y2[1])
        distance = []
        smallesti = []
        smallestd = []
        xyvect = np.transpose((x, y))
        for j in range(row):  # extremely slow, not sure what can be done
            vy2 = np.vstack((points[j, :], xyvect))
            t_xyvect = spatial.distance.squareform(spatial.distance.pdist(vy2))
            distance = t_xyvect[0, 1 : x.size + 1]
            index2 = np.argwhere(distance == np.min(distance))
            smallesti.append(index2[0][0])
            smallestd.append(distance[index2[0][0]])
            del t_xyvect
            del distance
        ep = np.argwhere(np.array(smallestd) < 8)
        if ep.size > 0:
            for row in ep:
                newimg = np.zeros(line2.shape)
                x1 = points[row[0], 0]
                y1 = points[row[0], 1]
                x2 = x[smallesti[row[0]]]
                y2 = y[smallesti[row[0]]]

                xnew = np.linspace(x1, x2, 1000)
                ynew = np.linspace(y1, y2, 1000)
                index = np.unique(
                    np.ravel_multi_index(
                        (
                            np.array(np.round(xnew), dtype=np.int64),
                            np.array(np.round(ynew), dtype=np.int64),
                        ),
                        newimg.shape,
                        order="F",
                    )
                )
                index_coord = np.unravel_index(index, newimg.shape, order="F")

                newimg[index_coord] = 1
                newimg = newimg > 0.5
                line2 = line2 + newimg
                line2 = np.array(line2, dtype=bool) | inpt
                remainder = remainder * (remainder ^ line)

    edge = e + line2 > 0
    return (remainder, edge)


def labelendpoints(L2):
    """
    Find coords of endpoints

    Parameters
    ----------
    L2 : mask of object

    Returns
    -------
    endpoints : array of endpoint coords

    """
    endpoints = []
    c = np.max(L2)
    for i in range(1, c + 1):
        # eptemp = ndimage.filters.generic_filter(L2 == i,endFilt, (3,3))
        eptemp = bwmorph.endpoints(L2 == i)
        rowcol = np.argwhere(eptemp)
        endpoints.append(rowcol)
    return endpoints


def longestConstrainedPath(bwin):
    """
    Calculates longest continous path in a thinned binary image

    Parameters
    ----------
    bwin : 2D binary image input

    Returns
    -------
    bwOut : 2D binary image showing the longest calculated path

    """
    M = bwin.shape[0]
    neighborOffsets = [-1, M, 1, -M, M + 1, M - 1, -M + 1, -M - 1]
    thinnedImg = morphology.thin(bwin)
    endpoints = np.argwhere(bwmorph.endpoints(thinnedImg) > 0)
    if endpoints.shape[0] == 2:
        return thinnedImg
    # mask = bwmorph.endpoints(thinnedImg)>0  # unused
    bwdg = skfmm.distance(thinnedImg)
    bwOut = np.zeros(thinnedImg.shape)
    startPoint = np.argwhere(bwdg == np.max(bwdg))
    startPoint = startPoint[0]
    bwdg[startPoint[0], startPoint[1]] = 0
    bwOut[startPoint[0], startPoint[1]] = 1
    startPoint = np.ravel_multi_index((startPoint[0], startPoint[1]), bwdg.shape)
    neighbors = startPoint + np.array(neighborOffsets)
    inds = np.argsort(bwdg[np.unravel_index(neighbors, bwdg.shape)])
    # bothNeighbors = np.transpose(np.unravel_index(neighbors[inds[0:2]], bwdg.shape))
    bothn = neighbors[inds[0:2]]
    for i in range(1, -1, -1):
        activePixel = bothn[i]
        while activePixel != 0:
            active = np.unravel_index(activePixel, bwdg.shape)
            bwOut[active[0], active[1]] = 1
            bwdg[active[0], active[1]] = 0
            neighbors = neighborOffsets + activePixel
            use = np.unravel_index(neighbors, bwdg.shape)
            if (np.argwhere(bwdg[use] == np.max(bwdg[use]))[0][0]) != 0:
                activePixel = neighbors[
                    np.argwhere(bwdg[use] == np.max(bwdg[use]))[0][0]
                ]
            else:
                activePixel = 0
    return bwOut


def multiNuc(mem, nuc):
    """
    Find cells with multiple nuclei

    Parameters
    ----------
    mem : cell mask
    nuc : nucleus mask

    Returns
    -------
    k2 : mask of objects with greater than 1 nucleus

    """
    mem = np.array(mem > 0, dtype=np.int16)
    memL = measure.label(mem, connectivity=1)
    nucL = nuc * memL
    properties = measure.regionprops(
        np.array(nucL, dtype=np.int16), np.array(nuc, dtype=bool)
    )
    props = np.asarray(
        [
            (prop.label, prop.area, prop.max_intensity)
            for prop in properties
            if prop.area >= 18 and prop.max_intensity > 0
        ],
        order="F",
    )
    if props.size == 0:
        return np.zeros(mem.shape)

    uniqueX = np.array(np.unique(props[:, 0], return_counts=True))
    indexToRepeatedValue = np.argwhere(uniqueX[1, :] != 1)
    if indexToRepeatedValue.size == 2:
        repeatedValues = uniqueX[indexToRepeatedValue[1]]
    else:
        repeatedValues = uniqueX[indexToRepeatedValue]
    k2 = np.isin(memL, repeatedValues) * memL
    return k2


def NucCountBatch(memx, nuc, tube, total, mucLoc, dapi, sup):
    """
    Generates final nuclear and cell masks

    Parameters
    ----------
    memx : cell segmentatation from before
    nuc : nuclear mask
    tube : epithelial mask
    total : membrane mask
    mucLoc : Muc 2 tif file
    dapi : dapi tif file
    sup : supermembrane grayscale from main

    Returns
    -------
    mem5 : nuclear mask
    out2 : cell mask

    """
    totalorig = total
    memx = measure.label(memx, connectivity=1)
    mem = ridSmall(memx, nuc)
    mem = np.array(mem > 0, dtype=np.int16)

    # nuclei in epithelial
    nucBW = np.array(tube & nuc, dtype=np.int16)
    mem = nucBW + mem

    # find a muc2 mask
    if len(mucLoc) == 0:
        mask2 = np.zeros(total.shape, np.int16)
    else:
        # adaptiveimages: mucLoc always empty, placeholder
        return

    nuc = nuc * (nuc ^ mask2)
    nuc = morphology.remove_small_objects(
        nuc, 13
    )  # UserWarning: Only one label was provided to `remove_small_objects`. Did you mean to use a boolean array?

    # find cell_mask objects with multiple nuclei
    k2 = multiNuc(mem, nuc)

    # find all indices in k2
    indices = np.unique(k2)
    indices = indices[indices != 0]
    # rc = np.size(indices)  # unused
    k3 = k2 > 0
    sMask = np.shape(k3)
    for index in indices:
        tempobject = np.isin(k2, index)
        k3 = k3 * (k3 ^ tempobject)
        mem = mem * (mem ^ tempobject)
        se = morphology.disk(1)
        tempobject = morphology.closing(tempobject, se)

        BBox = BBoxCalc(tempobject)
        subObject = tempobject[BBox[0] : BBox[2], BBox[1] : BBox[3]]
        subTotal = total[BBox[0] : BBox[2], BBox[1] : BBox[3]]

        if subObject.size > 160000:
            continue

        lines = reseg(singleobject=subObject, total=subTotal, numx=-1)
        linesSub = np.zeros(sMask)
        linesSub[BBox[0] : BBox[2], BBox[1] : BBox[3]] = lines
        linesSub = np.array(linesSub, dtype=np.int16)

        output = tempobject * (tempobject ^ linesSub)

        k3 = k3 | output
        mem = mem | output

    k4 = ridSmall(k3, nuc)
    r1 = multiNuc(k4, nuc)

    # redo dapi, first by looking only at dapi inside the objects remaining
    dapi2 = dapi * np.array(r1, dtype=np.int16)

    # simple otsu size has to be > 50, not noise from otsu
    if np.max(dapi2) == 0:
        level = 0
    else:
        level = filters.threshold_otsu(dapi2)
    dapi3 = np.where(dapi2 > level + 0.01, 1, 0)
    dapi4 = morphology.remove_small_objects(dapi3, 50, connectivity=2)

    # new out by subtracting previous nuc, and then adding back the new mask
    # only inside objects
    out2 = (nuc - r1) > 0
    out2 = (out2 + dapi4) > 0

    # get remainder
    r2 = multiNuc(r1, out2) > 0

    # use supermembrane mask to do a simple watershed
    super2 = sup * r2
    mem2 = (mem * (mem ^ r2)) > 0
    bw3 = segmore(super2, 1)
    bw3 = bw3 > 0
    mem3 = mem2 | bw3

    r3 = multiNuc(bw3, out2)

    total = morphology.remove_small_objects(totalorig, 2, connectivity=2)

    indices = np.unique(r3)
    indices = indices[indices != 0]
    # rc = np.shape(indices)  # unused
    k4 = r3 > 0
    for i in indices:
        tempobject = np.isin(r3, i)
        k4 = k4 * (k4 ^ tempobject)
        mem3 = mem3 * (mem3 ^ tempobject)
        se = morphology.disk(1)
        tempobject = morphology.closing(tempobject, se)
        BBox = BBoxCalc(tempobject)
        subObject = tempobject[BBox[0] : BBox[2], BBox[1] : BBox[3]]
        subTotal = total[BBox[0] : BBox[2], BBox[1] : BBox[3]]

        if subObject.size > 160000:
            continue

        lines = reseg(singleobject=subObject, total=subTotal, numx=-1)
        linesSub = np.zeros(sMask)
        linesSub[BBox[0] : BBox[2], BBox[1] : BBox[3]] = lines
        linesSub = np.array(linesSub, dtype=np.int16)

        output = tempobject * (tempobject ^ linesSub)

        k4 = k4 | output
        mem3 = mem3 | output
    k5 = ridSmall(k4, out2)

    r4 = multiNuc(k5, out2)

    # filter small bits of nuc out in 2 nuclei
    temp = out2 & (r4 > 0)
    out2 = out2 * (out2 ^ temp)
    L = measure.label(temp, connectivity=1)
    if not np.max(L) == 0:
        properties = measure.regionprops(L)
        props = np.asarray([(prop.label, prop.area) for prop in properties], order="F")
        Areas = props[:, 1]
        keep = props[:, 0][Areas > 75]
        temp2 = np.isin(L, keep)
        out2 = out2 | temp2
    mem4 = ridSmall(mem3, out2)

    L = measure.label(mem4, connectivity=1)
    if not np.max(L) == 0:
        properties = measure.regionprops(L)
        props = np.asarray([(prop.label, prop.area) for prop in properties], order="F")
        Areas = props[:, 1]
        large_ones = props[:, 0][Areas > 2500]
        large = np.isin(L, large_ones) * L

    # circle metric
    bw = measure.label(large, connectivity=1)
    if not np.max(bw) == 0:
        properties = measure.regionprops(bw)
        props = np.asarray(
            [(prop.label, prop.area, prop.perimeter) for prop in properties], order="F"
        )
        Areas = props[:, 1]
        Perims = props[:, 2]
        Metrics = 4 * math.pi * Areas / Perims ** 2
        erase = props[:, 0][Metrics < 0.15]
        remain = props[:, 0][Metrics >= 0.15]
        erase1 = np.isin(bw, erase) * bw
        remain1 = np.isin(bw, remain) * bw
        # keep long cells
        L = measure.label(remain1, connectivity=1)
        properties = measure.regionprops(L)
        props = np.asarray(
            [(prop.label, prop.eccentricity) for prop in properties], order="F"
        )
        Eccen = props[:, 1]
        erasex = props[:, 0][Eccen < 0.5]
        erase2 = np.isin(L, erasex) * L
        erase1 = erase1 | erase2
    else:
        erase1 = np.zeros(mem4.shape, np.uint8)
    mem5 = mem4 * (mem4 ^ erase1)
    mem5 = measure.label(mem5, connectivity=1)
    return (mem5, out2)


def reseg(singleobject, total, numx, index=None):
    """
    Resegment a single cell with greater than 10% internal membrane

    Parameters
    ----------
    singleobject : initial segmentation for single cell
    total : membrane mask of cell
    numx : TYPE
        DESCRIPTION.
    index : TYPE
        DESCRIPTION.

    Returns
    -------
    bbb : resegmented cell

    """
    s = total.shape
    pixadj = 1
    if s[0] != 2048 or s[1] != 2048:
        pixadj = 3
    mw5 = singleobject - total
    se = morphology.disk(2)
    mw5 = morphology.binary_opening(mw5, se)
    e = bwperim(singleobject)
    added = mw5 + e
    del mw5
    e2 = morphology.binary_dilation(e, se)
    added = added + e2
    inv = util.invert(
        morphology.remove_small_holes(added, 15 * pixadj)
    )  # UserWarning: Any labeled images will be returned as a boolean array. Did you mean to use a boolean array?
    L = measure.label(inv, connectivity=1)
    properties = measure.regionprops(L)

    areas = [prop.area for prop in properties]
    maximum = max(areas)
    areas.remove(maximum)
    okay = np.asarray([(count + 1) for count, area in enumerate(areas) if area > 1])
    line = np.isin(L, okay)
    input_skeleton_image = morphology.thin(line)
    input_skeleton_image = np.multiply(input_skeleton_image, singleobject)
    if np.amax(input_skeleton_image) > 0:
        input_skeleton_image = extend2(input_skeleton_image, singleobject, 7)
    (remainder7, edge7) = intersectfirst(input_skeleton_image, singleobject)
    del input_skeleton_image
    L2 = measure.label(remainder7)
    number = np.max(L2)
    input_skeleton_image2 = np.zeros(L2.shape)
    for kk in range(1, number + 1):
        temp = L2 == kk
        skelD = trimtree(temp)
        if np.sum(skelD) > 1:
            input_skeleton_image2 = input_skeleton_image2 + skelD
    del L2
    inpt = morphology.thin(input_skeleton_image2)
    out = connectpoints(inpt)
    del inpt
    a = intersect_k(out, edge7)
    del edge7
    del out
    a = morphology.thin(a)
    segobject = bwmorph.spur(a, 3)
    segobject = finalconnect_2(segobject, singleobject, 7)
    segobject = segobject * singleobject
    segobject = np.where(segobject > 0.5, 1, 0) + np.where(e, 1, 0)
    segobject2 = finalconnect(segobject)
    del segobject
    if numx > 0:
        segobject3 = finalconnect_2(segobject2, singleobject, 100)
    else:
        segobject3 = segobject2
    del segobject2
    segobject3 = segobject3 * singleobject
    segobject3 = np.where(segobject3 > 0.5, 1, 0) + np.where(e, 1, 0)
    e2 = bwperim(singleobject)
    e2 = morphology.binary_dilation(e2, morphology.disk(2))
    segobject3 = np.where(segobject3 > 0, True, False)
    segobject3 = segobject3 * (segobject3 ^ e2)
    segobject3 = morphology.thin(segobject3)
    temp2 = morphology.binary_dilation(singleobject, morphology.disk(1))
    finale = bwperim(temp2)
    bbb = intersect_k(segobject3, finale)
    bbb = np.array(bbb, bool)
    bbb = bbb * (bbb ^ finale)
    return bbb


def ReSegCells(cellmask, total):
    """
    Resegment cells with internal membrane in an initial segmentation image

    Parameters
    ----------
    cellmask : initial watershed segmentation image
    total : membrane mask

    Returns
    -------
    maskout : resegmented cell mask

    """
    total = morphology.remove_small_objects(total, 30)
    mask2 = np.array(cellmask, dtype=bool)
    mask2 = util.invert(mask2)
    se = morphology.disk(4)
    """
    matlab structural element 
    se = np.array([[0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0]])
    """
    mask2 = morphology.binary_dilation(mask2, se)
    mask3 = cellmask - 65535 * mask2
    mask3 = np.where(mask3 > 0, mask3, 0)
    region_nums = np.unique(mask3)
    region_nums = np.delete(region_nums, 0)
    del mask2
    properties = measure.regionprops(
        np.array(mask3, dtype=np.int16), np.array(total, dtype=bool)
    )
    props = np.asarray(
        [
            (prop.label, prop.area, prop.mean_intensity)
            if prop.area > 600
            else (prop.label, prop.area, 0)
            for prop in properties
        ],
        order="F",
    )
    # Areas = props[:,1]  # unused
    Intensities = props[:, 2]
    keep = []
    keep = props[:, 0][Intensities > 0.1]
    # 90 cells kept, 108 in matlab, 108 if same selem is used
    print("Cell ReSeg " + str(len(keep)) + " objects; ")
    maskout = np.array(cellmask, dtype=np.int16)
    sMask = maskout.shape
    for obj in keep:
        tempobject = np.array(np.isin(cellmask, obj), dtype=np.int16)
        BBox = BBoxCalc(tempobject)
        subObject = tempobject[BBox[0] : BBox[2], BBox[1] : BBox[3]]
        subTotal = total[BBox[0] : BBox[2], BBox[1] : BBox[3]]
        if subObject.size > 160000:
            continue
        lines = reseg(subObject, subTotal, -1, obj)
        linesSub = np.zeros(sMask)
        linesSub[BBox[0] : BBox[2], BBox[1] : BBox[3]] = lines
        linesSub = np.array(linesSub, dtype=np.int16)
        maskout = maskout - 65535 * linesSub
    maskout[maskout < 0] = 0
    maskout = measure.label(maskout, connectivity=1)
    return maskout


def ridSmall(k3, nuc):
    """
    Filter out cells with areas less than 50

    Parameters
    ----------
    k3 : cell mask
    nuc : nuclear mask

    Returns
    -------
    k4 : cells with area > 50 and included in nuc mask

    """
    L = measure.label(k3, connectivity=1)
    if np.max(L) == 0:
        return k3
    else:
        properties = measure.regionprops(L, np.array(nuc, dtype=bool))
        props = np.asarray(
            [
                (prop.label, prop.area, prop.mean_intensity)
                if prop.area > 50
                else (prop.label, prop.area, 0)
                for prop in properties
            ],
            order="F",
        )
        # Areas = props[:,1]  # unused
        Intensities = props[:, 2]
        keep = []
        keep = props[:, 0][Intensities > 0]
        k4 = np.isin(L, keep) * L
        return k4


def segmore(name, s):
    """
    Segment with watershed

    Parameters
    ----------
    name : image or filename
    s : level 1 or 2

    Returns
    -------
    bw3 : segmented image

    """
    if type(name) == str:
        I = io.imread(name)
    else:
        I = name
    bw = I
    D = -ndimage.morphology.distance_transform_edt(np.logical_not(bw) == 0)
    # Ld = segmentation.watershed(D, watershed_line = True)  # unused
    # bw2 = np.where(Ld == 0, 0, bw)  # unused
    mask = imextendedmin(D, s)
    D2 = imimposemin(D, mask)
    Ld2 = segmentation.watershed(D2, watershed_line=True)
    bw3 = np.where(Ld2 == 0, 0, bw)
    return bw3


def selectobjects(keep, mask):
    """
    Select a list of objects to keep in mask

    Parameters
    ----------
    keep : list of indices to keep
    mask : image mask

    Returns
    -------
    storemask : mask of indices/objects kept

    """
    storemask = np.zeros(mask.shape)
    mask2 = mask == keep
    indices = np.nonzero(mask2)
    storemask[indices] = keep
    return storemask


def sortclasses(given_inputs, separation_distance, connectivity):
    """
    Sort coordinates into the same class if categorized as
    neighboring pixels based on parameters

    Parameters
    ----------
    given_inputs : input coords
    separation_distance : separation distance of neighbouring pixels from
    current coordinate
    connectivity : 4 or 8, 4 covers pixels horizontal and vertical to current,
    8 covers all surrounding pixels to current

    Returns
    -------
    output : Mx2 matrices, sorted matrices of input

    """
    output = []
    pixel_class = np.empty((0, 0))
    if given_inputs.size > 0:
        if separation_distance < 1:
            scaling_factor = 1 / separation_distance
            given_inputs = np.multiply(scaling_factor, given_inputs)
            separation_distance = 1
        pixel_class = given_inputs[0, :]
        given_inputs = given_inputs[1:, :]
    else:
        print("Warning: There are no inputs to sort")  # replace with error throw
    Nshape = pixel_class.shape
    while given_inputs.size > 0:
        n = 0
        while n < Nshape[0]:
            pos_in_given_inputs = []
            if pixel_class.ndim == 1:
                x = pixel_class[0]
                y = pixel_class[1]
            else:
                x = pixel_class[n, 0]
                y = pixel_class[n, 1]
            for m in range(1, separation_distance + 1):
                if connectivity == 8:
                    coord = np.array(
                        [
                            [x - m, y - m],
                            [x - m, y],
                            [x - m, y + m],
                            [x, y + m],
                            [x, y - m],
                            [x + m, y - m],
                            [x + m, y],
                            [x + m, y + m],
                        ]
                    )
                elif connectivity == 4:
                    coord = np.array([[x - m, y], [x, y + m], [x, y - m], [x + m, y]])
                else:
                    print("Wrong connectivitiy")
                for row in coord:
                    xmatch = np.argwhere(given_inputs[:, 0] == row[0])
                    ymatch = np.argwhere(given_inputs[:, 1] == row[1])
                    match = np.intersect1d(xmatch, ymatch)
                    if match.size > 0:
                        pos_in_given_inputs.append(match)

            pos_in_given_inputs = np.asarray(pos_in_given_inputs, dtype=int)
            matched_coord = given_inputs[pos_in_given_inputs, :]

            if np.size(pos_in_given_inputs) > 0:
                pixel_class = np.append(pixel_class, matched_coord)
                given_inputs = np.delete(given_inputs, pos_in_given_inputs, 0)
            n = n + 1
            pixel_class = np.reshape(pixel_class, (-1, 2))
            Nshape = pixel_class.shape
        output.append(pixel_class)
        if given_inputs.size > 0:
            pixel_class = given_inputs[0, :]
            given_inputs = given_inputs[1:, :]
        else:
            pixel_class = np.empty((0, 0))

    if pixel_class.size > 0:
        output.append(pixel_class)

    if len(output) > 0:
        for cell in output:
            cell = cell / 2
    return output


def stromal_nuclei_segmentation(image):
    """
    Segments stromal nuclei using nuclear probability file

    Parameters
    ----------
    image : stromal probability image

    Returns
    -------
    B : Segmented stromal nuclei

    """
    image = image[:, :, 0]
    threshold = 1.0  # threshold for binarization
    total = image.size

    # apply top hat and bottom hat filter
    se = morphology.disk(30)  # se different from matlab
    tophat = image - morphology.opening(image, se)  # extremely slow
    bottomhat = morphology.closing(image, se) - image
    diff = np.array(tophat, np.int16) - np.array(bottomhat, np.int16)
    diff[diff < 0] = 0
    filterImage = image + diff
    se = morphology.disk(15)
    tophat = image - morphology.opening(image, se)  # extremely slow
    bottomhat = morphology.closing(image, se) - image
    diff = np.array(tophat, np.int16) - np.array(bottomhat, np.int16)
    diff[diff < 0] = 0
    filterImage = filterImage + diff

    counts = exposure.histogram(filterImage)[0]
    ssum = np.cumsum(counts)
    bg = 0.215 * total
    fg = 0.99 * total
    low = np.searchsorted(ssum, bg)
    high = np.searchsorted(ssum, fg)
    highin = high / 255
    if highin > 1:
        highin = 1

    adjustedImage = exposure.rescale_intensity(
        filterImage, (low, highin * 255), (0, 255)
    )
    adjustedImage = exposure.adjust_gamma(adjustedImage, 1.8)

    # image binarization, threshold chosen based on experience
    binarization = adjustedImage > threshold
    # se2 = morphology.disk(5)
    se2 = np.array(
        [
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    afterOpening = morphology.binary_opening(binarization, se2)
    se3 = np.array(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ],
        dtype=bool,
    )
    final = morphology.binary_opening(afterOpening, se3)
    D = -ndimage.morphology.distance_transform_edt(final)
    D[~final] = float("-inf")
    D = imhmin(D, 1, 1)

    L = segmentation.watershed(D, connectivity=2, watershed_line=True)

    B = L > 1
    # B[B == 1] = 0
    # B[B > 0] = 1
    # B = np.array(B, bool)
    return B


def trimtree(mask):
    """
    Return unbranched segment

    Parameters
    ----------
    mask : line segment

    Returns
    -------
    skelD : longest branch

    """
    skel = morphology.skeletonize(mask)
    B = bwmorph.branchpoints(skel)
    B = measure.label(B)
    if np.max(B) == 0:
        skelD = mask
    else:
        skelD = longestConstrainedPath(mask)
    return skelD
