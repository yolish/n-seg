import numpy as np
import pandas as pd
from skimage.measure import label
import os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage.io import imread
import time



# region dataset preparation
def read_imgs_labels(labels_file):
    # based on: https://www.kaggle.com/kmader/nuclei-overview-to-submission, with modifications
    imgs_labels = pd.read_csv(labels_file)
    imgs_labels['EncodedPixels'] = imgs_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])
    return imgs_labels

def collect_imgs_details(data_dir):
    all_imgs_paths = glob(os.path.join(data_dir, '*', '*', '*', '*'))
    imgs_df = pd.DataFrame({'path': all_imgs_paths})
    img_id = lambda in_path: in_path.split('/')[-3]
    img_type = lambda in_path: in_path.split('/')[-2]
    img_group = lambda in_path: in_path.split('/')[-4]
    imgs_df['ImageId'] = imgs_df['path'].map(img_id)
    imgs_df['ImageType'] = imgs_df['path'].map(img_type)
    imgs_df['DatasetType'] = imgs_df['path'].map(img_group)
    return imgs_df

def collect_dataset(imgs_df, dataset_type, labels_file = None):
    labels_df = None
    if labels_file is not None:
        labels_df = read_imgs_labels(labels_file)
    query = 'DatasetType==\"' + dataset_type + '\"'
    dataset_details = imgs_df.query(query)
    dataset_rows = []
    group_cols = ['ImageId']

    for n_group, n_rows in dataset_details.groupby(group_cols):
        img_record = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}

        img_paths = n_rows.query('ImageType == "images"')['path'].values.tolist()
        assert(len(img_paths) == 1)
        img_record['ImagePath'] = n_rows.query('ImageType == "images"')['path'].values.tolist()[0]
        if labels_df is not None:
            img_record['MaskPaths'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()
            img_record['EncodedPixels'] = labels_df[labels_df['ImageId'] == img_record.get('ImageId')]['EncodedPixels'].values.tolist()
        dataset_rows += [img_record]
    dataset = pd.DataFrame(dataset_rows)
    return dataset
# endregion


#region visualization
def plot_predicted_masks(examples, fig_size, plot_true_mask=True):
    n_cols = 3
    total_n_imgs = len(examples)
    if not plot_true_mask:
        n_cols = 2
    mpl.rcParams.update({'font.size': 6})

    n_imgs = int(total_n_imgs/5)

    fig, axes = plt.subplots(n_imgs, n_cols, figsize=fig_size)

    i = 0
    for img_id, prediction in examples.items():

        img = prediction.get('img')
        predicted_mask = prediction.get('predicted_mask')
        true_mask = prediction.get('true_mask')
        iou = prediction.get('iou')

        subplot = axes[i][0]
        subplot.imshow(img)
        subplot.axis('off')
        subplot.set_title('Image')

        subplot = axes[i][2]
        subplot.imshow(predicted_mask, cmap='nipy_spectral')
        subplot.axis('off')
        if iou is not None:
            subplot.set_title('Mask (IoU: {:.3f}'.format(iou))
        else:
            subplot.set_title('Mask')

        if plot_true_mask and true_mask is not None:
            subplot = axes[i][3]
            subplot.imshow(true_mask, cmap='nipy_spectral')
            subplot.axis('off')
            subplot.set_title('True mask')
        i = i + 1
        if i == n_imgs:
            i = 0
            plt.show()
            fig, axes = plt.subplots(n_imgs, n_cols, figsize=fig_size)




#endregion

#region segmentation encoding/decoding (RLE)
def calc_rle(arr):
    # original code from: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
    # minor changes (comments, renaming, etc.) done
    '''
    run-length encoding (rle) assumes that pixels are one-indexed and numbered from top to bottom, then left to right
    so, for example 1 is pixel (1,1), 2 is pixel (2,1), etc.
    full definition: https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    the input is a labelled blob
    :param arr: numpy array of shape (height, width), 1 - mask, 0 - background
    :return: run length encoding for the given array
    '''
    # take the transpose to get down-then-right ordering, then flatten
    mask = np.where(arr.T.flatten()==1)[0]
    rle = []
    prev = -2
    for index in mask:
        if index > prev+1: # end of "stretch" - need to start a new run length
            rle.extend((index+1, 0)) # add the first pixel (1-indexed) and the run-length
        rle[-1] += 1 # increase the run-length by one until we get to the next "stretch"
        prev = index
    return rle

def combine_rles(rles, h, w):
    mask = np.zeros(h * w, dtype=np.uint8)
    for i, rle in enumerate(rles):
        rld = decode_rle(rle)
        mask[rld] = i+1
    return mask.reshape((h, w)).T

def combine_masks(masks_paths):
    # the combined mask has a different label for each mask
    combined_mask = imread(masks_paths[0])
    combined_mask[combined_mask > 0] = 1
    for i in xrange(1, len(masks_paths)):
        mask = imread(masks_paths[i])
        combined_mask[mask > 0] = i + 1
    return combined_mask

def decode_rle(rle):
    indices = []
    for idx, cnt in zip(rle[0::2], rle[1::2]):
        indices.extend(list(range(idx - 1, idx + cnt - 1)))  # RLE is 1-based index
    return indices

def get_rles_from_df(imgs_df, img_id):
    rles = (imgs_df[imgs_df['ImageId'] == img_id]['EncodedPixels']).values[0]
    return sorted(rles, key = lambda x: x[0])

def get_rles_from_mask(labelled_img, label_img = False):
    '''
    :param labelled_img: the image with the segmented objects; integer np array of shape (hight_, width)
           if not labelled 1= foreground, 0= background
    :param thresh: the threshold to apply to convert class probabilities to mask vs background
    :param label_img: a boolean indicating whether to label the mask to get the different objects
    :return: a list of RLEs for the segmented objects in the given image
    '''
    rles = []
    if label_img:
        labelled_img = label(labelled_img) # Label connected regions of an integer array.

    n_labels = labelled_img.max()
    if n_labels<1:
        labelled_img[0,0] = 1 # ensure at least one mask per image
    for l in range(1, n_labels+1): # for each mask calculate its run-length encoding
        mask = labelled_img == l
        rle = calc_rle(mask)
        rles.append(rle)
    return sorted(rles, key = lambda x: x[0])
#endregion


#region post-processing for submission #
def format_rle(rle):
    return " ".join([str(i) for i in rle])

def to_submission_df(predictions):
    df = pd.DataFrame()
    for img_id, pred_rles in predictions.items():
        for rle in pred_rles:
            s = pd.Series({'ImageId': img_id, 'EncodedPixels': format_rle(rle)})
            df = df.append(s, ignore_index=True)
    return df
#endregion

#region testing and logging
# check rle function
# from: https://www.kaggle.com/kmader/nuclei-overview-to-submission with modifications
def test_rle(rles_from_mask, rles_from_df):
    match, mismatch = 0, 0
    # assume they are both sorted
    for mask_rle, df_rle in zip(rles_from_mask,rles_from_df):
        for i_x, i_y in zip(mask_rle, df_rle):
            if i_x == i_y:
                match += 1
            else:
                mismatch += 1
    print('Matches: %d, Mismatches: %d'% (match, mismatch))

def start_action(msg):
    start_time = time.time()
    print("start " + msg)
    return start_time

def complete_action(msg, start_time):
    print("completed " + msg)
    print("elapsed time: {}".format(time.time()-start_time))

# check mean IoU vs visualization
# test iou - make a fake prediction
#endregion
