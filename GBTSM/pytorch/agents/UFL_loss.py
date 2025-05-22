import torch
import torch.nn as nn

# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn
def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [2,3,4]

    # Two dimensional
    elif len(shape) == 4 : return [2,3]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, delta, gamma, epsilon=1e-07, num_classes=4):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        cross_entropy = -y_true_one_hot * torch.log_softmax(y_pred, dim=1)

        losses = []
        for c in range(self.num_classes):
            if c == 0:  # Background - apply focal modulation
                ce = torch.pow(1 - torch.softmax(y_pred, dim=1)[:, c, :, :, :], self.gamma) * cross_entropy[:, c, :, :, :]
                ce = (1 - self.delta) * ce
            else:  # Foreground classes - no focal modulation
                ce = cross_entropy[:, c, :, :, :]
                ce = self.delta * ce / (self.num_classes - 1)
            losses.append(ce)

        loss = torch.stack(losses, dim=1)  
        loss = torch.mean(loss, dim=[2, 3, 4])
        return loss

class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta, gamma, epsilon=1e-07, num_classes=4):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        axis = identify_axis(y_true_one_hot.size())

        tp = torch.sum(y_true_one_hot * y_pred, axis=axis)
        fn = torch.sum(y_true_one_hot * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true_one_hot) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        losses = []
        for c in range(self.num_classes):
            if c == 0:  # Background - no focal modulation
                dice_loss = 1 - dice_class[:, c]
            else:  # Foreground classes - apply focal modulation
                dice_loss = (1 - dice_class[:, c]) * torch.pow(1 - dice_class[:, c], self.gamma)
            losses.append(dice_loss)

        loss = torch.stack(losses, dim=-1) 
        return loss

class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight, delta, gamma, num_classes=4, class_weights=None):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.num_classes = num_classes
        
        # Inicializuojame class weights if not provided then set to 1
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)
        if self.class_weights.device != torch.device("cuda"):
            self.class_weights = self.class_weights.cuda()
            
        self.asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma, num_classes=self.num_classes)
        self.asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma, num_classes=self.num_classes)

    def forward(self, y_pred, y_true):
        asymmetric_ftl = self.asymmetric_ftl(y_pred, y_true)  
        asymmetric_fl = self.asymmetric_fl(y_pred, y_true)    
        
        loss = self.weight * asymmetric_ftl + (1 - self.weight) * asymmetric_fl  
        weighted_loss = loss * self.class_weights.view(1, -1) 
        return torch.mean(weighted_loss)