import utils
import torch
import torch.nn as nn

def hamming_loss(y_pred, y):
  if y_pred.shape != y.shape:
    raise Exception('The tensors don\'t have the same shape!')
  return (y_pred != y).sum().item()/y.numel()

def iou(y_, y):
    mask_y_ = (y_ == 1)
    mask_y = (y == 1)
    intersection = (mask_y_ & mask_y).sum((1, 2))
    union = (mask_y_ | mask_y).sum((1, 2))
    iou = torch.true_divide(intersection, union).mean()
    return iou.item()

def dice_coefficient(y_, y):
    mask_y_ = (y_ == 1)
    mask_y = (y == 1)
    numerator = 2*(mask_y_ & mask_y).sum((1, 2))
    denominator = (mask_y_ == 1).sum((1, 2)) + (mask_y == 1).sum((1, 2))
    dice_coeff = torch.true_divide(numerator, denominator).mean()
    return dice_coeff.item()

def init_running_metrics(metrics_names):
    metrics = {}
    for name in metrics_names:
        metrics[name] = 0.
    return metrics

def update_running_metrics(metrics, logits_missing, y_missing, letter_masks, 
                           total_n_missing_letters, loss=None, custom_loss=None, criterion=None):
    with torch.no_grad():
        batch_size_ = letter_masks.shape[0]
        n_missing_letters = (~letter_masks).sum().item()
        y_pred = utils.discretize(torch.sigmoid(logits_missing))

        if 'bce_all' in metrics:
            assert loss is not None
            metrics['bce_all'] += loss.item()*batch_size_
        if 'bce' in metrics:
            metrics['bce'] += criterion(logits_missing, y_missing).item()*n_missing_letters
        if 'custom_bce' in metrics:
            assert custom_loss is not None
            metrics['custom_bce'] += custom_loss.item()*n_missing_letters
        if 'hamming' in metrics:
            metrics['hamming'] += hamming_loss(y_pred, y_missing)*n_missing_letters
        if 'iou' in metrics:
            # print(iou(y_pred, y_missing), n_missing_letters, iou(y_pred, y_missing)*n_missing_letters)
            metrics['iou'] += iou(y_pred, y_missing)*n_missing_letters
        if 'dice' in metrics:
            metrics['dice'] += dice_coefficient(y_pred, y_missing)*n_missing_letters
        if 'euclidian' in metrics:
            metrics['euclidian'] += torch.cdist(y_pred, y_missing)*n_missing_letters
        
        total_n_missing_letters += n_missing_letters

    return metrics, total_n_missing_letters


def compute_final_metrics(running_metrics, dataset_size, total_n_missing_letters):
    
    final = {}

    if 'bce_all' in running_metrics:
        final['bce_all'] = running_metrics['bce_all'] / dataset_size
    if 'bce' in running_metrics:
        final['bce'] = running_metrics['bce'] / total_n_missing_letters
    if 'custom_bce' in running_metrics:
        final['custom_bce'] = running_metrics['custom_bce'] / total_n_missing_letters
    if 'hamming' in running_metrics:
        final['hamming'] = running_metrics['hamming'] / total_n_missing_letters
    if 'iou' in running_metrics:
        final['iou'] = running_metrics['iou'] / total_n_missing_letters
    if 'dice' in running_metrics:
        final['dice'] = running_metrics['dice'] / total_n_missing_letters

    return final

class boostedLoss(nn.Module):
    '''   
    if c > 1 increases the value of optimizing for the hidden masks'''

    def __init__(self, loss, c):
        super(boostedLoss, self).__init__()
        self.loss = loss
        self.c = c

    def forward(self, y, y_, letter_masks):
        # print(y.device, y_.device, letter_masks.device, n_masked.device, n_not_masked.device)~
        n_not_masked = (letter_masks).sum().detach()
        n_masked = (~letter_masks).sum().detach()
        loss_not_masked = self.loss(y[letter_masks], y_[letter_masks])
        loss_masked = self.loss(y[~letter_masks], y_[~letter_masks])
        output = n_not_masked*loss_not_masked + self.c*n_masked*loss_masked
        output = output / (n_not_masked + self.c*n_masked)
        return output
