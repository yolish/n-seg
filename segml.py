
import numpy as np
from segutils import get_rles_from_mask, decode_rle, get_rles_from_df
import time


#region evaluation metrics
# see alternative impl. on: https://www.kaggle.com/wcukierski/example-metric-implementation/notebook
def calc_avg_precision_iou(pred_rles, true_rles, thrs = np.arange(0.5, 1.0, 0.05)):
    '''
    # given true rles and predicted rles for a given image
    # decode them
    # calculate the intersection and union for each rle pair
    # for each threshold t, calculate TP, FP, FN:
    # A true positive is counted when a single predicted object matches a ground truth object with an IoU > t.
    # A false positive indicates a predicted object had no associated ground truth object.
    # A false negative indicates a ground truth object had no associated predicted object.
    # so all the pred RLE with at least one IoU > t are TP
    # all other pred RLEs are FP
    # all true RLEs with all IoU <= t are FN
    # return the mean IoU over all threshold
'''
    pred_rlds = [decode_rle(rle) for rle in pred_rles]
    true_rlds = [decode_rle(rle) for rle in true_rles]

    total_pred = len(pred_rlds)
    total_true = len(true_rlds)
    iou_scores = np.zeros(shape=(total_pred, total_true))

    for i in xrange(total_pred):
        for j in xrange(total_true):
            pred_rld = set(pred_rlds[i])
            true_rld = set(true_rlds[j])
            intersection = len(pred_rld.intersection(true_rld))
            union = len(pred_rld) + len(true_rld) - intersection
            iou_scores[i,j] = float(intersection)/union

    avg_precision_iou = 0.0
    for t in thrs:
        pred_with_match = np.sum(iou_scores > t, 1)
        true_with_match = np.sum(iou_scores > t, 0)
        pred_with_match[pred_with_match > 0] = 1
        true_with_match[true_with_match > 0] = 1
        tps = np.sum(pred_with_match)
        fps = total_pred - tps
        fns = total_true - np.sum(true_with_match)
        precision_iou = float(tps) / (tps + fps + fns)
        avg_precision_iou = avg_precision_iou + precision_iou
    return avg_precision_iou/len(thrs)

def calc_expected_iou(labelled_mask):
    true_rles = get_rles_from_mask(labelled_mask, label_img=False)
    pred_rles = get_rles_from_mask(labelled_mask > 0, label_img=True)
    expected_iou = calc_avg_precision_iou(pred_rles, true_rles)
    return expected_iou

#endregion


#region learning and testing
def assign_best_seg(train_dataset, seg_map):

    for sample in train_dataset:
        img_id = sample.get('id')
        img = sample.get('img')
        seg_ious = np.zeros(len(seg_map))
        for cls, seg_alg in seg_map.items():
            mask = seg_alg(img)
            pred_rles = get_rles_from_mask(mask)
            true_rles = get_rles_from_df(train_dataset.dataset, img_id)
            avg_precision_iou = calc_avg_precision_iou(pred_rles, true_rles)
            seg_ious[cls] = avg_precision_iou
        sample['seg_ious'] = seg_ious
        sample['best_seg_cls'] = np.argmax(seg_ious)



def train_seg_classifier(train_dataset):
    model = None
    return model


def predict_best_seg_cls(model, dataset):
    predicted_seg_cls = {}
    for sample in dataset:
        img_id = sample.get('id')
        img = sample.get('img')
        cls = model.predict(img)
        predicted_seg_cls[img_id] = cls
    return predicted_seg_cls

def predict_masks(predicted_seg_cls, dataset, seg_map):
    predicted_masks = {}
    for sample in dataset:
        img_id = sample.get('id')
        img = sample.get('img')
        seg_class = predicted_seg_cls.get(img_id)
        seg_alg = seg_map.get(seg_class)
        predicted_masks[img_id] = seg_alg(img)
    return predict_masks

def evaluate(predictions, dataset, examples=None):
    sum_avg_precision_iou = 0.0
    for img_id, pred_rles in predictions.items():
        true_rles = get_rles_from_df(dataset.dataset, img_id)
        avg_precision_iou = calc_avg_precision_iou(pred_rles, true_rles)
        sum_avg_precision_iou = sum_avg_precision_iou + avg_precision_iou
        if examples is not None:
            example = examples.get(img_id)
            if example is not None:
                example['iou'] = avg_precision_iou
    mean_avg_precision_iou = sum_avg_precision_iou / len(predictions.keys())
    return mean_avg_precision_iou

def test(model, dataset, seg_map, n_masks_to_collect=15):
    predicted_seg_cls = predict_best_seg_cls(model, dataset)
    predicted_masks = predict_masks(predicted_seg_cls, dataset, seg_map)

    i = 0
    predictions = {}
    examples = {}
    for img_id, predicted_mask in predicted_masks.items():
        pred_rles = get_rles_from_mask(predicted_mask)
        predictions[img_id] = pred_rles
        # save a few examples for plotting
        if i < n_masks_to_collect:
            sample = dataset[i]
            img = sample.get('img')
            examples[img_id] = {'img': img, 'seg': seg_map.get(predicted_seg_cls.get(img_id)),
                                'predicted_mask': predicted_mask,
                                'true_mask': sample.get('labelled_mask')  # can be None for the test set
                                }
        i = i + 1
    return predictions, examples
#endregion
