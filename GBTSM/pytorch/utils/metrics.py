import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

class Segmentationmetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.classes = list(range(num_classes))
        # Calculating metrics only to claases, excluding background class
        self.correct_classes = [1, 2, 3] 

    def calculate_f1_scores(self, pred, true):
        return f1_score(true, pred, labels=self.correct_classes, average=None)

    def calculate_confusion_metrics(self, pred, true):
        cm = confusion_matrix(true, pred, labels=self.correct_classes)
        TP = np.diag(cm).astype(np.float64)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)
        
        epsilon = 1e-6
        IoU = TP / (TP + FP + FN + epsilon)
        Sensitivity = TP / (TP + FN + epsilon)
        Specificity = TN / (TN + FP + epsilon)
        
        return {
            'IoU': IoU,
            'Sensitivity': Sensitivity,
            'Specificity': Specificity
        }
        
class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg