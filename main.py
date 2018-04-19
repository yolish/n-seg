import os
import json
import sys
import datetime
import time
from NucleiDataset import NucleiDataset
from seglib import SEG_ALGS
import segml
import segutils
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")

    config_filename = sys.argv[1]
    with open(config_filename) as json_config_file:
        config = json.load(json_config_file)

    paths_config = config.get("paths")
    data_path = paths_config.get("input_path")
    output_path = paths_config.get("output_path")

    actions_config = config.get("actions")
    explore = actions_config.get("explore")
    evaluate = actions_config.get("evaluate")
    test = actions_config.get("test")
    visualize = actions_config.get("visualize")

    action = "collecting images details"
    start_time = segutils.start_action(action)
    imgs_details = segutils.collect_imgs_details(data_path)
    segutils.complete_action(action, start_time)

    action = "creating train dataset"
    start_time = segutils.start_action(action)
    train_labels_file = os.path.join(data_path, 'train_labels.csv')
    train_dataset = NucleiDataset('train', imgs_df=imgs_details,
                                  labels_file=train_labels_file)
    print("set size: {}".format(len(train_dataset)))
    segutils.complete_action(action, start_time)

    if explore:
        for i in xrange(5):
            sample = train_dataset[i]
            for cls in xrange(len(SEG_ALGS)):
                segutils.plot_sample_segmentation(sample, cls, SEG_ALGS)
        sys.exit(0)

    action = "assigning best segmentation class"
    start_time = segutils.start_action(action)
    best_segs = segml.assign_best_seg(train_dataset, SEG_ALGS)
    segutils.complete_action(action, start_time)


    n_imgs_to_plot = 5
    i = 0
    n_seg_candidates = len(SEG_ALGS)
    sum_seg_ious = np.zeros(n_seg_candidates)
    n_seg_chosen = np.zeros(n_seg_candidates)
    for img_id, best_seg in best_segs.items():
        cls = best_seg.get('cls')
        iou = best_seg.get('iou')
        sum_seg_ious[cls] = sum_seg_ious[cls] + iou
        n_seg_chosen[cls] = n_seg_chosen[cls] + 1

        if visualize and i < n_imgs_to_plot:
            sample = train_dataset[i]
            segutils.plot_sample_segmentation(sample, cls, SEG_ALGS, iou)
        i = i+1

    print("IOUs for segmentation algorithms (by class): {}".format(sum_seg_ious/n_seg_chosen))
    print("mean IoU for train set: {}".format(sum_seg_ious.sum()/(i-1)))
    action = "training classifier"
    start_time = segutils.start_action(action)
    model = segml.train_seg_classifier(train_dataset, best_segs)
    segutils.complete_action(action, start_time)

    if evaluate:
        action = "creating validation  dataset"
        start_time = segutils.start_action(action)
        validation_labels_file = os.path.join(data_path, 'validation_labels.csv')
        validation_dataset = NucleiDataset('validation', imgs_df=imgs_details,
                                           labels_file=validation_labels_file)
        print("set size: {}".format(len(validation_dataset)))
        segutils.complete_action(action, start_time)


        action = "making predictions for the validation set"
        start_time = segutils.start_action(action)
        predictions, examples = segml.test(model, validation_dataset, SEG_ALGS)
        segutils.complete_action(action, start_time)

        action = "evaluating predictions"
        start_time = segutils.start_action(action)
        mean_avg_precision_iou = segml.evaluate(predictions, validation_dataset, examples=examples)
        print("IoU for validation dataset: {}".format(mean_avg_precision_iou))
        segutils.complete_action(action, start_time)
        if visualize:
            # visually evaluate a few images by comparing images and masks
            segutils.plot_predicted_masks(examples, (22, 27))

    if test:

        test_dataset = NucleiDataset('test', imgs_df=imgs_details)

        action = "making predictions for the test set"
        start_time = segutils.start_action(action)
        predictions, examples = segml.test(model, test_dataset, SEG_ALGS)
        segutils.complete_action(action, start_time)
        # visually evaluate a few images by comparing images and masks
        if visualize:
            segutils.plot_predicted_masks(examples, (7, 12), plot_true_mask=False)

        action = "writing the predictions to submission format"
        start_time = segutils.start_action(action)
        submission_df = segutils.to_submission_df(predictions)
        submission_filename = output_path + "model_predictions_" + timestamp + ".csv"
        submission_df.to_csv(submission_filename, columns=('ImageId','EncodedPixels'), index=False)
        print("predictions on tess set written to: {}".format(submission_filename))
        segutils.complete_action(action, start_time)


