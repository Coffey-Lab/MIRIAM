import os
import skimage
import numpy as np
import platform
import cv2
import time
import csv
from skimage import io, morphology, exposure, filters, measure, util, segmentation
from scipy import signal, ndimage
from itertools import zip_longest
from .common import imimposemin
from .cellseg import ReSegCells, NucCountBatch, stromal_nuclei_segmentation
from .cellshape import cell_shape_images, CellShapeAutoencoder
from .quant import MxIF_quantify, MxIF_quantify_stroma


def bwareafilt(image, range):
    # filter objects by area
    out = image.copy()
    img_label = measure.label(image, connectivity=1)
    component_sizes = np.bincount(img_label.ravel())
    too_large = component_sizes > range[1]
    too_small = component_sizes < range[0]
    too_large_mask = too_large[img_label]
    too_small_mask = too_small[img_label]
    out[too_large_mask] = 0
    out[too_small_mask] = 0
    return out


def blurimg2_batch(nuc):
    # find and exclude blurred regions from dapi image
    nuc = np.uint8(
        (nuc.astype(float) - float(np.amin(nuc[:])))
        / (float(np.amax(nuc[:])) - float(np.amin(nuc[:])))
        * 255
    )
    nuc = exposure.equalize_adapthist(nuc)
    edge = filters.sobel(nuc)  # matlab uses threshold of 0.04
    se = morphology.disk(20)
    closed = morphology.binary_closing(edge, se)
    return closed


def MaskFiltration(mask, low):
    # filter mask
    label_img = measure.label(mask)
    properties = measure.regionprops(label_img)
    areas = [prop.area for prop in properties]
    filtMask = bwareafilt(
        mask,
        [(min(areas) + low * (max([area - min(areas) for area in areas]))), max(areas)],
    )
    return filtMask


def ML_probability(probs, LowAreaLim, thresh):
    # process probability masks
    tm = probs[:, :, 0]  # get red or epithelial channel from epithelium
    out = np.zeros(tm.shape, np.double)
    tm = cv2.normalize(
        tm, out, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F
    )  # normalize red channel to grayscale
    tm = tm > thresh  # thresholding
    if not np.amax(tm):
        return tm

    # morphological filtering to smooth edges
    new = morphology.remove_small_objects(tm, 500)
    if not np.amax(new):
        return
    new = morphology.binary_opening(new, morphology.disk(5))
    new = morphology.binary_erosion(new, morphology.disk(3))
    new = morphology.binary_closing(new, morphology.disk(7))

    # more filtering
    new = MaskFiltration(new, LowAreaLim)

    new = morphology.binary_erosion(new, morphology.disk(3))
    new = morphology.remove_small_holes(new, 1000)
    new = morphology.binary_opening(new, morphology.disk(3))
    new = morphology.binary_closing(new, morphology.disk(3))
    new = morphology.remove_small_objects(new, 500)
    new = morphology.dilation(new, morphology.disk(4))
    return new


def SegDirFormatting(Directory):
    """
    Create and format data structure of single cell segmentation

    Parameters
    ----------
    Directory : directory for slide containing AFRemoved images folder and
    Registered images folder - assumes Round 001 is baseline and all fies are
    tif format
    Returns
    -------
    Tuple containing:
        AFRemoved = string for location of AFRemoved images
        DAPI =string for location of DAPI images
        OutDir = cell array of strings for the output files
    """

    OutDir = []
    if platform.system() == "Windows":
        AFRemoved = Directory + r"\AFRemoved"
        DAPI = Directory + r"\RegisteredImages\S001"
        Seg = Directory + r"\SegQuants"
        if not os.path.exists(Seg):
            os.makedirs(Seg)
        if not os.path.exists(Seg + r"\Stacks"):
            os.makedirs(Seg + r"\Stacks")
        if not os.path.exists(Seg + r"\CellSeg"):
            os.makedirs(Seg + r"\CellSeg")
        if not os.path.exists(Seg + r"\CellSegFinal"):
            os.makedirs(Seg + r"\CellSegFinal")
        if not os.path.exists(Seg + r"\EpiMask"):
            os.makedirs(Seg + r"\EpiMask")
        if not os.path.exists(Seg + r"\Novlp"):
            os.makedirs(Seg + r"\Novlp")
        if not os.path.exists(Seg + r"\NucMask"):
            os.makedirs(Seg + r"\NucMask")
        if not os.path.exists(Seg + r"\SuperMem"):
            os.makedirs(Seg + r"\SuperMem")
        if not os.path.exists(Seg + r"\MemMask"):
            os.makedirs(Seg + r"\MemMask")
        if not os.path.exists(Seg + r"\NucMaskFinal"):
            os.makedirs(Seg + r"\NucMaskFinal")
        if not os.path.exists(Seg + r"\PosStats"):
            os.makedirs(Seg + r"\PosStats")
        if not os.path.exists(Seg + r"\ML"):
            os.makedirs(Seg + r"\ML")
        if not os.path.exists(Seg + r"\TumorMask"):
            os.makedirs(Seg + r"\TumorMask")
        if not os.path.exists(Seg + r"\CellShape"):
            os.makedirs(Seg + r"\CellShape")
        OutDir = [
            (Seg + "\\" + file + "\\")
            for file in os.listdir(Seg)
            if not file.startswith(".")
        ]
    else:
        AFRemoved = Directory + r"/AFRemoved"
        DAPI = Directory + r"/RegisteredImages/S001"
        Seg = Directory + r"/SegQuants"
        if not os.path.exists(Seg):
            os.makedirs(Seg)
        if not os.path.exists(Seg + r"/Stacks"):
            os.makedirs(Seg + r"/Stacks")
        if not os.path.exists(Seg + r"/CellSeg"):
            os.makedirs(Seg + r"/CellSeg")
        if not os.path.exists(Seg + r"/CellSegFinal"):
            os.makedirs(Seg + r"/CellSegFinal")
        if not os.path.exists(Seg + r"/EpiMask"):
            os.makedirs(Seg + r"/EpiMask")
        if not os.path.exists(Seg + r"/Novlp"):
            os.makedirs(Seg + r"/Novlp")
        if not os.path.exists(Seg + r"/NucMask"):
            os.makedirs(Seg + r"/NucMask")
        if not os.path.exists(Seg + r"/SuperMem"):
            os.makedirs(Seg + r"/SuperMem")
        if not os.path.exists(Seg + r"/MemMask"):
            os.makedirs(Seg + r"/MemMask")
        if not os.path.exists(Seg + r"/NucMaskFinal"):
            os.makedirs(Seg + r"/NucMaskFinal")
        if not os.path.exists(Seg + r"/PosStats"):
            os.makedirs(Seg + r"/PosStats")
        if not os.path.exists(Seg + r"/ML"):
            os.makedirs(Seg + r"/ML")
        if not os.path.exists(Seg + r"/TumorMask"):
            os.makedirs(Seg + r"/TumorMask")
        if not os.path.exists(Seg + r"/CellShape"):
            os.makedirs(Seg + r"/CellShape")
        OutDir = [
            (Seg + r"/" + file + r"/")
            for file in os.listdir(Seg)
            if not file.startswith(".")
        ]
    return (AFRemoved, DAPI, OutDir)


def CellSeg(SlideDir, quantify, shape, stroma, tumor, start):
    """
    Wrapper for cell segmentation

    Parameters
    ----------
    SlideDir : directory for slide containing AFRemoved folder and Registered
    images folder - assumes round 001 is baseline and all files are .tif
    quantify : whether or not to quantify, 1=yes, 0=no
    shape : whether or not to characterize shape, 1=yes, 0=no
    stroma : whether or not to segment stroma, 1=yes, 0=no
    tumor : whether or not to include tumors, 1=yes, 0=no
    start : what image to start processing

    Returns
    -------
    None - function saves images and quantifications.

    """

    # Parse Direcotry supplied for cell segmentation
    (AFRemoved, DAPI, OutDir) = SegDirFormatting(SlideDir)
    # get formatting for AfRemoved, DAPI, and output directories
    AFFiles = os.listdir(AFRemoved)
    AFList = []
    for file in AFFiles:
        AFList.append(file.split("_AFRemoved_"))
    AFList = np.asarray(AFList)
    AFList = np.resize(AFList, (95, 1, 2))
    PosList = np.unique(AFList[:, :, 1])
    AFList = np.unique(AFList[:, :, 0])  # list of markers
    PosList = np.char.replace(PosList, ".tif", "")  # list of positions
    OutPos = PosList

    # Format DAPI images for Cytell based imaging
    DapiList = sorted(DAPI + "/" + element for element in os.listdir(DAPI))

    # make sure the number of DAPI images equals the number of positions
    if len(DapiList) != len(PosList):
        print("Error: Dapi Image Mismatch")
        return
    OutDir = sorted(OutDir)

    # status updates
    print("Segmentation of:", SlideDir, ";", str(len(PosList)), " Positions;\n")

    # Segmentation and Quantification for each position
    for i in range(start, len(PosList)):
        print(f"{OutPos[i]}:")
        # make Stacks of AFRemoved images and Dapi if they don't exist
        if not os.path.exists(f"{OutDir[10]}{OutPos[i]}_stack.tif"):
            print(f"Stack: {OutPos[i]}")
            # form tif image stack for each position with images from each marker
            # io.imsave(f"{OutDir[10]}{OutPos[i]}_stack.tif", io.imread(DapiList[i]))
            stack = []
            stack.append(io.imread(DapiList[i]))
            for j in range(
                len(AFList)
            ):  # loop through AFRemoved images and append to tiff stack
                stack.append(
                    io.imread(f"{AFRemoved}/{AFList[j]}_AFRemoved_{OutPos[i]}.tif")
                )
            stack = np.asarray(stack)
            io.imsave(f"{OutDir[10]}{OutPos[i]}_stack.tif", stack)
        # Check for probability files
        if not os.path.exists(f"{OutDir[4]}epi_{OutPos[i]}_stack_Probabilities.png"):
            print("No Epithelial Probability File")
            continue
        if not os.path.exists(f"{OutDir[4]}mem_{OutPos[i]}_stack_Probabilities.png"):
            print("No Membrane/Nucleus Probabilty File")
            continue

        # nuclear segmentation and generate supermembrane and binary membrane mask
        if not (
            os.path.exists(f"{OutDir[7]}NucMask_{OutPos[i]}.png")
            or os.path.exists(f"{OutDir[11]}SuperMem_{OutPos[i]}.tif")
            or os.path.exists(f"{OutDir[5]}MemMask_{OutPos[i]}.png")
        ):
            # read in membrane probability file
            Probs = io.imread(f"{OutDir[4]}mem_{OutPos[i]}_stack_Probabilities.png")

            # threshold with nuclear probability >0.6 for nuclear mask
            mask = np.where(Probs[:, :, 1] > 255 * 0.6, np.uint8(255), np.uint8(0))
            io.imsave(f"{OutDir[7]}NucMask_{OutPos[i]}.png", mask)
            io.imsave(f"{OutDir[11]}SuperMem_{OutPos[i]}.tif", Probs[:, :, 0])
            # thresholding for membrane mask
            MemMask = np.where(Probs[:, :, 0] > 255 * 0.6, np.uint8(255), np.uint8(0))
            io.imsave(f"{OutDir[5]}MemMask_{OutPos[i]}.png", MemMask)
        else:
            # read files if previously generated
            mask = io.imread(f"{OutDir[7]}NucMask_{OutPos[i]}.png")
            SuperMem = io.imread(f"{OutDir[11]}SuperMem_{OutPos[i]}.tif")
            MemMask = io.imread(f"{OutDir[5]}MemMask_{OutPos[i]}.png")

        mask = np.where(mask > 0, np.uint8(1), np.uint8(0))  # make nuclear mask binary

        # fill in small holes and smooth
        mask = morphology.remove_small_holes(mask, 20 ** 3)
        selem = morphology.disk(3)
        mask = morphology.binary_opening(mask, selem)

        # remove blurred nuclear regions
        mask = np.multiply(mask, blurimg2_batch(io.imread(DapiList[i])))

        s = mask.shape
        pixadj = 1
        if s[0] != 2048 or s[1] != 2048:
            pixadj = 3

        # generate epithelial mask from machine learning

        if not (os.path.exists(OutDir[3] + "EpiMask_" + OutPos[i] + ".png")):
            print("EpiMask Processing: ")
            epiMask = io.imread(
                OutDir[4] + "epi_" + OutPos[i] + "_stack_Probabilities.png"
            )
            epiMask = ML_probability(
                epiMask, pixadj * 0.01, 0.45
            )  # create epi mask from probability map
            io.imsave(
                OutDir[3] + "EpiMask_" + OutPos[i] + ".png",
                255 * np.array(epiMask, dtype=np.uint8),
            )
        else:
            epiMask = np.array(
                io.imread(OutDir[3] + "EpiMask_" + OutPos[i] + ".png"), dtype=bool
            )

        # thin membrane borders prior to initial watershed
        MemMask = morphology.thin(MemMask)

        # generate cell (re)segmentation and nuclear segmentation images
        if (not os.path.exists(f"{OutDir[8]}NucMaskFinal_{OutPos[i]}.png")) or (
            not os.path.exists(f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif")
        ):

            print("CellSeg;")

            if not (os.path.exists(f"{OutDir[0]}L2_{OutPos[i]}.tif")):
                L2 = np.array(np.add(util.invert(epiMask), MemMask), dtype=np.uint8)

                # watershed segmentation with nuclei as basins
                L2 = segmentation.watershed(imimposemin(L2, mask), watershed_line=True)
                L2 = np.array(L2, dtype=np.float_)

                # return cells only in epithelial mask
                L2 = np.multiply(L2, epiMask)
                io.imsave(f"{OutDir[0]}L2_{OutPos[i]}.tif", np.int16(L2))

            else:
                L2 = io.imread(f"{OutDir[0]}L2_{OutPos[i]}.tif")
            if not (os.path.exists(f"{OutDir[0]}CellSeg_{OutPos[i]}.tif")):
                MemMask = np.array(
                    io.imread(f"{OutDir[5]}MemMask_{OutPos[i]}.png"), dtype=bool
                )
                start = time.time()
                CellSeg = ReSegCells(L2, MemMask)
                end = time.time()
                print(end - start)
                io.imsave(
                    f"{OutDir[0]}CellSeg_{OutPos[i]}.tif",
                    np.array(CellSeg, dtype=np.int16),
                )
            else:
                CellSeg = io.imread(f"{OutDir[0]}CellSeg_{OutPos[i]}.tif")

            if not (os.path.exists(f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif")):
                CellSeg = io.imread(f"{OutDir[0]}CellSeg_{OutPos[i]}.tif")
                SuperMem = io.imread(f"{OutDir[11]}SuperMem_{OutPos[i]}.tif")
                Probs = io.imread(f"{OutDir[4]}mem_{OutPos[i]}_stack_Probabilities.png")
                # check for cells with multiple nuclei and re-segment if they exist
                (watcellseg, mask) = NucCountBatch(
                    CellSeg, mask, epiMask, MemMask, [], Probs[:, :, 1], SuperMem
                )
                watcellseg = watcellseg > 0

                filt = np.array(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    dtype=bool,
                )

                # fill small crosses
                hitmiss = ndimage.morphology.binary_hit_or_miss(
                    watcellseg, ~filt, np.zeros((5, 5))
                )
                spots = morphology.remove_small_objects(hitmiss, 2)
                diff = hitmiss * (hitmiss ^ spots)

                diff = signal.convolve2d(np.array(diff, np.uint8), filt, mode="same")
                watcellseg = watcellseg + diff
                # set non-epithelial pixels to zero
                watcellseg[epiMask == 0] = 0
                watcellseg = morphology.remove_small_objects(watcellseg, 15)
                watcellseg = watcellseg > 0
                watcellseg = measure.label(watcellseg, connectivity=1)

                io.imsave(
                    f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif",
                    np.array(watcellseg, dtype=np.uint16),
                )
                io.imsave(
                    f"{OutDir[8]}NucMaskFinal_{OutPos[i]}.png",
                    np.array(255 * (mask > 0), dtype=np.uint8),
                )

            else:
                watcellseg = io.imread(f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif")
                mask = io.imread(f"{OutDir[8]}NucMaskFinal_{OutPos[i]}.png") > 0
        else:
            watcellseg = io.imread(f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif")
            mask = io.imread(f"{OutDir[8]}NucMaskFinal_{OutPos[i]}.png") > 0

        # Quantification if specified
        if quantify == 1:

            if os.path.exists(
                f"{OutDir[9]}PosStats_{OutPos[i]}.csv"
            ) and os.path.exists(f"{OutDir[6]}Novlp_{OutPos[i]}.png"):
                continue
            else:
                if np.max(watcellseg) == 0:
                    print("\\n")
                print("Quant; ")

                if tumor == 0:
                    (Stats, NoOvlp) = MxIF_quantify(
                        i, watcellseg, AFRemoved, AFList, PosList, mask, MemMask, OutPos
                    )

                if tumor == 1:
                    if not os.path.exists(f"{OutDir[4]}TumorMask_{OutPos[i]}.png"):
                        tumorMask = io.imread(
                            f"{OutDir[4]}tum_{OutPos[i]}_stack_Probabilities.png"
                        )
                        tumorMask = ML_probability(tumorMask, pixadj * 0.01, 0.5)
                        io.imsave(
                            f"{OutDir[4]}TumorMask_{OutPos[i]}.png",
                            np.array(255 * (tumorMask > 0), np.uint8),
                        )
                        (Stats, NoOvlp) = MxIF_quantify(
                            i,
                            watcellseg,
                            AFRemoved,
                            AFList,
                            PosList,
                            mask,
                            MemMask,
                            OutPos,
                            tumorMask,
                        )

                # format data table and write
                transposed_data = list(zip_longest(*Stats.values()))
                with open(r"{OutDir[9]}PosStats_{OutPos[i]}.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(Stats.keys())
                    writer.writerows(transposed_data)

                io.imsave(r"{OutDir[6]}Novlp_{OutPos[i]}.png", NoOvlp)

        # Stromal Quantification
        if stroma == 1:
            print("Stromal quant:")
            if not os.path.exists(
                f"{OutDir[4]}str_{OutPos[i]}_stack_Probabilities.png"
            ):
                ("No epithelial probability file")
            elif not os.path.exists(
                f"{OutDir[9]}StrPosStats_{OutPos[i]}.csv"
            ) or not os.path.exists(f"{OutDir[6]}StrNovlp{OutPos[i]}.png"):
                stromal_nuclei = stromal_nuclei_segmentation(
                    io.imread(f"{OutDir[4]}str_{OutPos[i]}_stack_Probabilities.png")
                )
                stromal_nuclei[epiMask == 1] = 0
                stromal_grow = morphology.binary_dilation(
                    stromal_nuclei, morphology.square(5)
                )  # dilate nuclei
                # watershed on dilated cells with nuclei as seed points
                stromal_label = segmentation.watershed(
                    imimposemin(np.array(stromal_grow, np.uint8), stromal_nuclei),
                    watershed_line=True,
                )
                stromal_label[stromal_grow == 0] = 0
                io.imsave(
                    f"{OutDir[1]}StrCellSegFinal_{OutPos[i]}.tif",
                    np.array(stromal_label, np.uint16),
                )
                io.imsave(
                    f"{OutDir[8]}StrNucMaskFinal_{OutPos[i]}.png",
                    255 * np.array(stromal_nuclei, np.uint8),
                )

                # quantify markers in cells and write out data
                (strStats, strNoOvlp) = MxIF_quantify_stroma(
                    i,
                    stromal_label,
                    AFRemoved,
                    AFList,
                    PosList,
                    stromal_nuclei,
                    pixadj,
                    epiMask,
                    OutPos,
                )
                transposed_data = list(zip_longest(*strStats.values()))
                with open(
                    f"{OutDir[9]}strPosStats_{OutPos[i]}.csv", "w", newline=""
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(strStats.keys())
                    writer.writerows(transposed_data)

                io.imsave(f"{OutDir[6]}strNovlp_{OutPos[i]}.png", strNoOvlp)
            else:
                continue

        # Shape Pre-processing
        if shape == 1:
            # load final segmentation image, extract cells, save as npz files
            if os.path.exists(
                f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif"
            ) and not os.path.exists(f"{OutDir[2]}CellShape_{OutPos[i]}.npz"):
                print("Cell Shape Pre-Processing; ")
                CellImages = io.imread(f"{OutDir[1]}CellSegFinal_{OutPos[i]}.tif")
                CellImages = cell_shape_images(CellImages)
                np.savez_compressed(f"{OutDir[2]}CellShape_{OutPos[i]}", CellImages)
                # np.savez_compressed(OutDir[2] + 'CellShape_' + OutPos[i] + '.npz', CellImages)
            if i == (len(PosList) - 1):
                print("Training Autoencoder; ")
                # Run autoencoder with extracted cell images
                trainList = CellShapeAutoencoder(OutDir[2], 0.2)
                header = ["ID", "Pos", "Selec"]
                for k in range(1, 257):
                    header.append("Enc" + str(k))
                with open(f"{OutDir[2]}EncodedCells.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(trainList)
    return None
