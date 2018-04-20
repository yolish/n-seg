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


class Segmentation(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class ThreshSegmentation(Segmentation):

    def __init__(self, name, threshold_func, separate_objects):
        super(ThreshSegmentation, self).__init__(name)
        self.threshold = threshold_func
        self.separate_objects = separate_objects

    def __call__(self, img):
        grey_img = rgb2grey(img)
        t = self.threshold(grey_img)
        mask = label(grey_img > t)
        if self.separate_objects:
            pass
        return mask

class MorphSegmentation(Segmentation):
    def __init__(self, name, radius):
        super(ThreshSegmentation, self).__init__(name)
        self.radius = radius

    def __call__(self, img):
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.radius, self.radius))
        # remove background noise with morphological opening
        grey_img = cv2.morphologyEx(grey_img, cv2.MORPH_OPEN, ellipse)
        grey_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_OTSU)[1]
        if (np.sum(grey_img == 255) > np.sum(grey_img == 0)):
            grey_img = cv2.bitwise_not(grey_img)
        mask = cv2.morphologyEx(grey_img, cv2.MORPH_OPEN, circle)
        # label connected components
        return cv2.connectedComponents(mask)[1]




#SEG_ALGS = {0:seg3}
SEG_ALGS = {0:ThreshSegmentation("yen",filter.threshold_yen, False)}
#SEG_ALGS = {0:seg1, 1:seg2, 2:seg3}
