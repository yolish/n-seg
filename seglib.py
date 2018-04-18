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
import cv2



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


    #img = mark_boundaries(img, quickshift(img_as_float(img), kernel_size =5, max_dist = 10, ratio = 1.0))

    #img = mark_boundaries(img, slic(img_as_float(img), n_segments=10))
    #fimg = rgb2grey(img)
    #t = filters.threshold_otsu(fimg)
    #img = mark_boundaries(img, (fimg > t).astype(np.uint8), color=(1,0,0))
    #img  = mark_boundaries(img, (fimg - filters.threshold_niblack(fimg)< 0).astype(np.uint8), color=(1,0,0))

    #img_gray = rgb2grey(img)
    #img_gray = img[:, :, 1]
    # morphological opening (size tuned on training data)
    #circle7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    #img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
    # Otsu thresholding
    #img_th = cv2.threshold(img_open, 0, 255, cv2.THRESH_OTSU)[1]
    # Invert the image in case the objects of interest are in the dark side
    #if (np.sum(img_th == 255) > np.sum(img_th == 0)):
    #    img_th = cv2.bitwise_not(img_th)
    # second morphological opening (on binary image this time)
    #bin_open = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    # connected components
    #img = mark_boundaries(img,cv2.connectedComponents(bin_open)[1], color=(1,0,0))


SEG_ALGS = {0:seg1, 1:seg2}
