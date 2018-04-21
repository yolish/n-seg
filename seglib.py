import numpy as np
from skimage.measure import regionprops
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
#from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.exposure import adjust_gamma, rescale_intensity, equalize_hist, equalize_adapthist
from skimage.util import img_as_float
from skimage.segmentation import quickshift, felzenszwalb, slic
import random
from skimage.color import rgb2lab, rgb2grey
from skimage import filters
from scipy import ndimage as ndi
import cv2
from skimage.morphology import label, convex_hull_object
from skimage.feature import canny


class Segmentation(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class ThreshSegmentation(Segmentation):

    def __init__(self, name, threshold_func, block_size=None):
        super(ThreshSegmentation, self).__init__(name)
        self.threshold = threshold_func
        self.block_size = block_size

    def __call__(self, img):
        grey_img = rgb2grey(img)
        if self.block_size is not None:
            t = self.threshold(grey_img, self.block_size)
        else:
            t = self.threshold(grey_img)
        mask = (grey_img > t).astype(np.uint8)
        if (np.sum(mask == 1) > np.sum(mask == 0)):
            mask = np.invert(mask)

        return label(mask)

class EdgeSegmentation(Segmentation):

    def __init__(self):
        super(EdgeSegmentation, self).__init__("edge")


    def __call__(self, img):
        grey_img = rgb2grey(img)
        mask = canny(grey_img)
        mask = ndi.binary_fill_holes(mask)
        #t = filters.threshold_minimum(mask)
        #mask = mask > t
        return mask

class MorphSegmentation(Segmentation):
    def __init__(self, radius):
        name = "morph-" + str(radius)
        super(MorphSegmentation, self).__init__(name)
        self.radius = radius

    def __call__(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.radius, self.radius))
        # remove background noise with morphological opening
        grey_img = cv2.morphologyEx(grey_img, cv2.MORPH_OPEN, ellipse)
        grey_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_OTSU)[1]
        if (np.sum(grey_img == 255) > np.sum(grey_img == 0)):
            grey_img = cv2.bitwise_not(grey_img)
        mask = cv2.morphologyEx(grey_img, cv2.MORPH_OPEN, ellipse)
        # label connected components
        return cv2.connectedComponents(mask)[1]




'''
SEG_ALGS = {0:EdgeSegmentation()}
SEG_ALGS = {0:ThreshSegmentation("minimum",filters.threshold_minimum, False),
            1:ThreshSegmentation("mean", filters.threshold_mean, False),
            2: ThreshSegmentation("triangle", filters.threshold_triangle, False)}
'''

SEG_ALGS = {0:ThreshSegmentation("isodata",filters.threshold_isodata),
            1:ThreshSegmentation("adaptive", filters.threshold_local, block_size=15),
            2:ThreshSegmentation("yen", filters.threshold_yen),
            3:MorphSegmentation(5),
            4:MorphSegmentation(7),
            5:ThreshSegmentation("minimum", filters.threshold_minimum)}

