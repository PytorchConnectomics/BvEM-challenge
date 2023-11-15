import numpy as np
import cc3d
import scipy


# define a function to evaluate the performance for an instance
def single_instance_metric(pred, gt):
    # pred and gt are binary mask
    
    # compute the tp, fp, fn
    tp = ((pred > 0) & (gt > 0)).sum()
    fp = ((pred > 0) & (gt <= 0)).sum()
    fn = ((pred <= 0) & (gt > 0)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = tp / (tp + fp + fn)
    return precision, recall, acc


# define a function to evaluate the instance level performance
def multi_instance_metric(pred, gt):
    # pred and gt are instance level labels
    
    pred_cc3d = cc3d.connected_components(pred > 0)

    best_precision = 0
    best_recall = 0
    best_acc = 0
    # for each instance in pred
    for i in range(1, pred_cc3d.max()+1):
        precision, recall, acc = single_instance_metric(pred_cc3d == i, gt)
        if precision > best_precision:
            best_precision = precision
            best_recall = recall
            best_acc = acc

    return best_precision, best_recall, best_acc