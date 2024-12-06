from torch import nn
import torch
import math
import torch.nn.functional as F


class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):

        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss

class incftContrastive(nn.Module):
    def __init__(self, lamda=2.0):
        super(incftContrastive, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda

    def forward(self, x, y):

        """
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of augmentation inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """

        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).cuda())

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()

        return loss_ce + self.lamda * loss_ne

class ClassContrastive(nn.Module):
    def __init__(self, lamda=2.0):
        super(ClassContrastive, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = lamda

    def forward(self, x, y):

        """
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of augmentation inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """

        assert x.shape == y.shape, ('Inputs are required to have same shape')

        # partition uncertainty index
        pui = torch.mm(F.normalize(x.t(), p=2, dim=1), F.normalize(y, p=2, dim=0))
        loss_ce = self.xentropy(pui, torch.arange(pui.size(0)).cuda())

        # balance regularisation
        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = math.log(p.size(0)) + (p * p.log()).sum()

        return loss_ce + self.lamda * loss_ne

class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()
        self.kl_loss_function = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x, y):
        """
        Arguments:
            x {Tensor} -- [assignment probabilities of pre inputs (N x K)]
            y {Tensor} -- [assignment probabilities of current inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        # kl_loss = kl_loss_function(mask_preds.log_softmax(dim=-1), joint_preds.softmax(dim=-1))
        sum_dist = (x+y)/2
        js_d1 = self.kl_loss_function(sum_dist.log_softmax(dim=-1), x.softmax(dim=-1))
        js_d2 = self.kl_loss_function(sum_dist.log_softmax(dim=-1), y.softmax(dim=-1))
        return 0.5*js_d1 + 0.5*js_d2


class PriorDist(nn.Module):
    def __init__(self):
        super(PriorDist, self).__init__()

    def forward(self, x, old_class, new_class, m):

        """
        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
        Returns:
            [Tensor] -- [Loss value]
        """

        x_old_class = x[:, :old_class * m]
        x_new_class = x[:, old_class * m: new_class * m]
        sum_x_old_class = torch.sum(x_old_class, dim=1)
        sum_x_new_class = torch.sum(x_new_class, dim=1)
        prior_pred_loss = sum_x_old_class.sum() / sum_x_new_class.sum()

        return prior_pred_loss

