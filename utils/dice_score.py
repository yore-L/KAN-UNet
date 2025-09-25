import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def intersection_over_union(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate IoU
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    intersection = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - intersection
    union = torch.where(union == 0, intersection, union)

    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean()


def recall(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate Recall (Sensitivity)
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    true_positives = (input * target).sum(dim=sum_dim)
    false_negatives = ((1 - input) * target).sum(dim=sum_dim)

    recall = (true_positives + epsilon) / (true_positives + false_negatives + epsilon)
    return recall.mean()


def precision(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate Precision (Positive Predictive Value)
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    true_positives = (input * target).sum(dim=sum_dim)
    false_positives = (input * (1 - target)).sum(dim=sum_dim)

    precision = (true_positives + epsilon) / (true_positives + false_positives + epsilon)
    return precision.mean()


def f1_score(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate F1 Score
    p = precision(input, target, reduce_batch_first, epsilon)
    r = recall(input, target, reduce_batch_first, epsilon)
    return 2 * (p * r) / (p + r + epsilon)


def multiclass_metrics(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate metrics for multiclass output (F1, Recall, IoU)
    dice = multiclass_dice_coeff(input, target, reduce_batch_first, epsilon)
    iou = intersection_over_union(input, target, reduce_batch_first, epsilon)
    f1 = f1_score(input, target, reduce_batch_first, epsilon)
    recall_value = recall(input, target, reduce_batch_first, epsilon)
    precision_value = precision(input, target, reduce_batch_first, epsilon)

    return {
        'dice_coeff': dice,
        'iou': iou,
        'f1_score': f1,
        'recall': recall_value,
        'precision': precision_value
    }


def dice_metrics(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Calculate metrics for dice loss (F1, Recall, IoU)
    dice = dice_coeff(input, target, reduce_batch_first, epsilon)
    iou = intersection_over_union(input, target, reduce_batch_first, epsilon)
    f1 = f1_score(input, target, reduce_batch_first, epsilon)
    recall_value = recall(input, target, reduce_batch_first, epsilon)
    precision_value = precision(input, target, reduce_batch_first, epsilon)

    return {
        'dice_coeff': dice,
        'iou': iou,
        'f1_score': f1,
        'recall': recall_value,
        'precision': precision_value
    }


# # Example usage
# if __name__ == '__main__':
#     input_tensor = torch.randn(4, 1024, 1024)  # Example input tensor (batch_size, height, width)
#     target_tensor = torch.randint(0, 2, (4, 1024, 1024))  # Example target tensor (binary mask)
#
#     # Compute the metrics
#     metrics = dice_metrics(input_tensor, target_tensor)
#
#     print("Metrics: ", metrics)
