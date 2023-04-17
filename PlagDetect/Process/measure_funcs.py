import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
import cv2

def measure_bbox(img, sqrt_area = True):
    """Function to measure crystal sizes using best fit bounding box; this is the method akin to Higgins' and perhaps similar to how Marian
    thinks of crystal sizes as well.

    Args:
        img (ndarray): labelled object image - result of the segmentation algorithm.
        sqrt_area (bool, optional): Set to True if size measure is area^0.5, if False then length will be used. Defaults to True.

    Returns:
        size (ndarray): Array with all segmented crystal sizes
        aspect ratio (ndarray): Array with all segmented crystal aspect ratios
    """
    #calculate contours
    cnt, _ = cv2.findContours(img.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #calculate minimum area bounding box (not axis aligned, but best fit)
    bb = cv2.minAreaRect(cnt[0])
    l1, l2 = bb[1]
    #work out right way round
    if l1 > l2:
        l = l1
        w = l2
    else:
        l = l2
        w = l1
    #calculate aspect ratio
    aspect_ratio = l/w
    #get the right size measure
    if sqrt_area == True:
        size = np.sqrt(l*w)
    else:
        size = l

    return size, aspect_ratio

def measure_ellipse(img, sqrt_area=True):
    """ Use best fit ellipse to calculate the crystal dimensions using the scikit-image toolbox. This is the default for
    many/most computer vision tasks and is also used in my slicing model. It is crucial to use this in conjunction with the slicing model
    otherwise additional errors will have to be taken into account.

    Args:
        img (ndarray): labelled object image - result of the segmentation algorithm.
        sqrt_area (bool, optional): Set to True if size measure is area^0.5, if False then length will be used. Defaults to True.

    Returns:
        size (ndarray): Array with all segmented crystal sizes
        aspect ratio (ndarray): Array with all segmented crystal aspect ratios
    """
    #generate properties objects
    props = measure.regionprops(img)

    area = []
    major_axis = []
    minor_axis = []
    #cycle through all objects to get data below
    for item in props:
        if item.minor_axis_length > 0:
            area.append(item.area)
            major_axis.append(item.major_axis_length)
            minor_axis.append(item.minor_axis_length)
        else:
            pass
    
    #calculate aspect ratio
    aspect_ratio = np.divide(major_axis,minor_axis)
    #get the chosen size measure right
    if sqrt_area == True:
        size = np.sqrt(np.asarray(area))
    else:
        size = np.asarray(major_axis)

    return size, aspect_ratio

def gen_texture_data(img, sqrt_area = True, BBOX = False):
    """Simple wrapper function to be used for generating textural data from labelled images. User has the option to 
    choose between the different size measure and the method used to measure them as well between best fit (minimum area)
    bounding box and best fit ellipse.

    Args:
        img (ndarray): labelled object image - result of the segmentation algorithm.
        sqrt_area (bool, optional): Set to True if size measure is area^0.5, if False then length will be used. Defaults to True.
        BBOX (bool, optional): Set to True if crystal measurements are to be done using best fit bounding box - otherwise best fit
                             ellipse is used. Defaults to False.

    Returns:
        size (ndarray): Array with all segmented crystal sizes
        aspect ratio (ndarray): Array with all segmented crystal aspect ratios
    """

    #make sure image is labelled - we'll assume connectivity = 2
    img = measure.label(img, connectivity=2)
    if BBOX == True:
        #use BBOX function
        size, aspect_ratio = measure_bbox(img, sqrt_area)
    else:
        #use ellipse function
        size, aspect_ratio = measure_ellipse(img, sqrt_area)

    return size, aspect_ratio
