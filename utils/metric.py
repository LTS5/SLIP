
import numpy as np


def accuracy(preds, targets):
    """Computes the accuracy given a target tensor and a prediction one
    Returns:
       The accuracy
    """

    acc = (preds==targets).sum() / len(preds)
    return acc*100


def classwise_accuracy(preds, targets):
        
    classes = np.arange(targets.max()+1)
    class_acc = []
    for i in classes:
        mask_class = targets==i
        accuracy_i = accuracy(preds[mask_class], targets[mask_class])
        class_acc.append(accuracy_i)

    avg_acc = np.array(class_acc).mean()

    return avg_acc, class_acc
