import numpy as np
from skimage import measure, morphology, io


def MxIF_quantify(
    i, watcellseg, AFRemoved, AFList, PosList, mask, MemMask, OutPos, tumorMask=None
):
    """
    Quantify AFRemoved images

    Parameters
    ----------
    i : index
    watcellseg : watershed seg
    AFRemoved : AFRemoved directory
    AFList : mxif marker array
    PosList : position array
    mask : mask
    MemMask : membrane mask
    OutPos : outpos array
    tumorMask : tumor mask. The default is None.

    Returns
    -------
    Stats : quanitification results
    NoOvlp : No overlap image

    """
    properties = measure.regionprops(watcellseg)
    props = np.asarray(
        [
            (prop.label, prop.centroid[0], prop.centroid[1], prop.area)
            for prop in properties
        ],
        order="F",
    )
    Stats = {}
    # Stats = measure.regionprops_table(watcellseg, properties = ('label', 'centroid', 'area'))
    Stats.update(
        {
            "Label": props[:, 0],
            "Cell_Centroid_X": props[:, 1],
            "Cell_Centroid_Y": props[:, 2],
            "Cell_Area": props[:, 3],
        }
    )
    s = watcellseg.shape
    NoOvlp = np.zeros((s[0], s[1], 3))
    for j in range(len(AFList)):
        print(AFList[j] + " ")
        AFim = io.imread(
            AFRemoved + "/" + AFList[j] + "_AFRemoved_" + OutPos[i] + ".tif"
        )
        AForig = np.copy(AFim)
        properties = measure.regionprops(watcellseg, AFim)
        AFQuantCell = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Cell_" + AFList[j]: AFQuantCell})
        AFim[mask == 0] = 0

        if j == 0:
            properties = measure.regionprops(watcellseg, mask)
            Area = [prop.area for prop in properties]
            Stats.update({"Nuc_Area": Area})

        properties = measure.regionprops(watcellseg, AFim)
        AFQuantNuc = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Nuc_" + AFList[j]: AFQuantNuc})

        # quantify cell edge (mem) stats
        AFim = np.copy(AForig)
        MemMask = watcellseg == 0
        disksize = 5

        MemMask = morphology.dilation(MemMask, morphology.square(disksize))
        MemMask[watcellseg == 0] = 0
        MemMask[mask == 1] = 0
        AFim[MemMask == 0] = 0

        if j == 0:
            properties = measure.regionprops(watcellseg, MemMask)
            Area = [prop.area for prop in properties]
            Stats.update({"Mem_Area": Area})

        properties = measure.regionprops(watcellseg, AFim)
        AFQuantMem = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Mem_" + AFList[j]: AFQuantMem})

        # quantify non nuclear and non mem (cyt) stats
        AFim = np.copy(AForig)
        CytMask = (watcellseg > 0) & (mask == 0) & (MemMask == 0)
        AFim[CytMask == 0] = 0

        if j == 0:
            properties = measure.regionprops(watcellseg, CytMask)
            Area = [prop.area for prop in properties]
            Stats.update({"Cyt_Area": Area})

        properties = measure.regionprops(watcellseg, AFim)
        AFQuantCyt = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Cyt_" + AFList[j]: AFQuantCyt})

    if tumorMask != None:
        properties = measure.regionprops(watcellseg, tumorMask)
        tumQuantCell = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Tumor": tumQuantCell})

    CellBorders = morphology.dilation(watcellseg > 0, np.ones((3, 3))) & (
        watcellseg == 0
    )

    NoOvlp[:, :, 0] = MemMask + CellBorders
    NoOvlp[:, :, 1] = CytMask + CellBorders
    NoOvlp[:, :, 2] = mask + CellBorders
    return (Stats, NoOvlp)


def MxIF_quantify_stroma(
    i, watcellseg, AFRemoved, AFList, PosList, mask, pixadj, epiMask, OutPos
):
    """
    Quantify stroma

    Parameters
    ----------
    i : index
    watcellseg : watershed seg
    AFRemoved : AFRemoved directory
    AFList : mxif marker array
    PosList : position array
    mask : mask
    MemMask : membrane mask
    OutPos : outpos array

    Returns
    -------
    Stats : quanitification results
    NoOvlp : No overlap image

    """
    properties = measure.regionprops(watcellseg)
    props = np.asarray(
        [
            (prop.label, prop.centroid[0], prop.centroid[1], prop.area)
            for prop in properties
        ],
        order="F",
    )
    Stats = {}
    # Stats = measure.regionprops_table(watcellseg, properties = ('label', 'centroid', 'area'))
    Stats.update(
        {
            "Label": props[:, 0],
            "Cell_Centroid_X": props[:, 1],
            "Cell_Centroid_Y": props[:, 2],
            "Cell_Area": props[:, 3],
        }
    )
    s = watcellseg.shape
    NoOvlp = np.zeros((s[0], s[1], 3))
    for j in range(len(AFList)):
        print(AFList[j] + " ")
        AFim = io.imread(
            AFRemoved + "/" + AFList[j] + "_AFRemoved_" + OutPos[i] + ".tif"
        )
        AForig = np.copy(AFim)
        properties = measure.regionprops(watcellseg, AFim)
        AFQuantCell = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Cell_" + AFList[j]: AFQuantCell})
        AFim[mask == 0] = 0

        if i == 0:
            properties = measure.regionprops(watcellseg, mask)
            Area = [prop.area for prop in properties]
            Stats.update({"Nuc_Area": Area})

        properties = measure.regionprops(watcellseg, AFim)
        AFQuantNuc = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Nuc_" + AFList[j]: AFQuantNuc})

        # quantify non nuclear and non mem (cyt) stats
        AFim = np.copy(AForig)
        CytMask = (watcellseg > 0) & (mask == 0)
        AFim[CytMask == 0] = 0

        if i == 0:
            properties = measure.regionprops(watcellseg, CytMask)
            Area = [prop.area for prop in properties]
            Stats.update({"Cyt_Area": Area})

        properties = measure.regionprops(watcellseg, AFim)
        AFQuantCyt = [
            np.median(prop.intensity_image[prop.intensity_image != 0])
            if np.any(prop.intensity_image)
            else 0
            for prop in properties
        ]
        Stats.update({"Median_Cyt_" + AFList[j]: AFQuantCyt})

    CellBorders = morphology.dilation(watcellseg > 0, np.ones((3, 3))) & (
        watcellseg == 0
    )

    NoOvlp[:, :, 0] = CellBorders
    NoOvlp[:, :, 1] = CytMask + CellBorders
    NoOvlp[:, :, 2] = mask + CellBorders
    return (Stats, NoOvlp)
