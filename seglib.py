import numpy as np
from skimage.measure import label
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#from skimage.segmentation import mark_boundaries, find_boundaries, label
from skimage.exposure import adjust_gamma, rescale_intensity, equalize_hist, equalize_adapthist
from skimage.util import img_as_float
from skimage.segmentation import quickshift, felzenszwalb, slic
import random
from skimage.color import rgb2lab, rgb2grey
from skimage import filter
from scipy import ndimage as ndi
import cv2
from skimage.morphology import disk, opening, binary_opening


def seg10(img, threshold_func, disk_size, pre_denoise_background):
    grey_img = rgb2grey(img)
    # remove noise from the background
    if pre_denoise_background:
        grey_img = opening(grey_img, disk(disk_size))
    t = threshold_func(grey_img)
    mask = grey_img > t
    #invert if required
    if (np.sum(mask == 1) > np.sum(mask == 0)):
        mask = np.invert(mask)
    #remove noise
    mask = opening(mask, disk(disk_size))
    return label(mask)

def seg11(img):
    return seg10(img, filter.threshold_yen, 7, False)

def seg12(img):
    return seg10(img, filter.threshold_otsu, 7, True)



def seg13(img):
    return seg10(img, filter.threshold_yen, 9, False)









def seg1(img):
    grey_img = rgb2grey(img)
    t = filter.threshold_yen(grey_img)

    mask = label(grey_img > t)

    return mask

def seg2(img):
    grey_img = rgb2grey(img)
    t = filter.threshold_otsu(grey_img)

    mask = label(grey_img > t)

    return mask

#blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)




def seg4(img, circle_size):



    #img_gray = rgb2grey(img)
    #img_gray = img[:, :, 1]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = cv2.GaussianBlur(image_gray, (7, 7), 1)
    # morphological opening (size tuned on training data)
    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (circle_size, circle_size))
    img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle)
    # Otsu thresholding
    img_th = cv2.threshold(img_open, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert the image in case the objects of interest are in the dark side
    if (np.sum(img_th == 255) > np.sum(img_th == 0)):
        img_th = cv2.bitwise_not(img_th)
    # second morphological opening (on binary image this time)
    bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle)
    # connected components
    return cv2.connectedComponents(bin_open)[1]

def seg5(img):
    return(seg4(img, 7))

def seg6(img):
    return(seg4(img, 5))

def seg3(img):
    grey_img = rgb2grey(img)
    edges = filter.sobel(grey_img)

    #t = filter.threshold_otsu(grey_img)
    #binary_mask = grey_img > t

    mask = label(ndi.binary_fill_holes(edges))
    return mask

def seg7(img):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (7, 7), 1)
    ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU)

    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    max_cnt_area = cv2.contourArea(cnts[0])

    if max_cnt_area > 50000:
        ret, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


    mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.connectedComponents(mask)[1]


SEG_ALGS = {0:seg11}
#SEG_ALGS = {0:seg5, 1:seg11, 2:seg13}
#SEG_ALGS = {0:seg1, 1:seg2, 2:seg3}
