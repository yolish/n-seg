import pandas as pd
import random
import numpy as np
from skimage.io import imread
from segml import calc_expected_iou
from segutils import collect_dataset, combine_rles, combine_masks


class NucleiDataset(object):
    """Nuclei dataset."""

    def __init__(self, type, imgs_df=None, dataset=None,
                 labels_file=None):
        '''

        :param imgs_df: a Pandas dataframe with details about the images
        :param labels_file: a csv file with the labels, optional
        '''
        assert(imgs_df is not None or dataset is not None)
        if type != 'test':
            assert(labels_file is not None)

        if imgs_df is not None:
            self.dataset = collect_dataset(imgs_df, type, labels_file)
        else:
            self.dataset = dataset
        self.type = type


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_record = self.dataset.iloc[idx,]
        img_path = img_record['ImagePath']
        img_id = img_record['ImageId']
        img = imread(img_path)[:,:,:3]
        sample = {'id':img_id, 'img':img, 'size':img.shape}
        if self.type != 'test':
            mask_paths = img_record['MaskPaths']
            if len(mask_paths) > 0:
                mask = combine_masks(mask_paths)
            else:
                h = img.shape[0]
                w = img.shape[1]
                rles = img_record['EncodedPixels']
                mask = combine_rles(rles, h, w)
            sample['labelled_mask'] = mask # only used for evaluation and plotting
            sample['binary_mask'] = (mask > 0).astype(np.uint8)
            if self.type == 'train':
                sample['expected_iou'] = calc_expected_iou(mask)
        return sample

    def split(self, frac, type, transform = None, filename=None):
        # split to 2 new datasets
        if filename is not None:
            split_out_df = pd.read_csv(filename)
            img_ids = split_out_df['ImageId'].values
            split_out_dataset = self.dataset.loc[self.dataset['ImageId'].isin(img_ids)]
            self.dataset = self.dataset.loc[~self.dataset['ImageId'].isin(img_ids)]
        else:
            sample_size = int(self.__len__()*frac)
            sampled_idx = random.sample(xrange(self.__len__()), sample_size)
            remaining_idx = [idx for idx in xrange(self.__len__()) if idx not in sampled_idx]
            split_out_dataset = self.dataset.iloc[sampled_idx]
            self.dataset = self.dataset.iloc[remaining_idx]

        return NucleiDataset(type, dataset=split_out_dataset, transform = transform, add_borders_to_mask=self.add_borders_to_mask)








