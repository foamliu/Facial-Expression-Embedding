import argparse
import logging
import math

import numpy as np
import torch

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import im_size


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, acc, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'acc': acc,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeterBag(object):

    def __init__(self, name_list):
        self.meter_dict = dict()
        self.name_list = name_list
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()

    def update(self, val_list):
        for i, name in enumerate(self.name_list):
            val = val_list[i]
            self.meter_dict[name].update(val)

    def __str__(self):
        ret = ''
        for name in self.name_list:
            ret += '{0}:\t {1:.4f}({2:.4f})\t'.format(name, self.meter_dict[name].val, self.meter_dict[name].avg)

        return ret


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def parse_args():
    parser = argparse.ArgumentParser(description='Facial Expression Embedding')
    # general
    parser.add_argument('--end-epoch', type=int, default=1000, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def align_face(raw, facial5points):
    # raw = cv.imread(img_fn, True)  # BGR
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (im_size, im_size)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (im_size, im_size)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def select_significant_face(bounding_boxes):
    best_index = -1
    best_rank = float('-inf')
    for i, b in enumerate(bounding_boxes):
        bbox_w, bbox_h = b[2] - b[0], b[3] - b[1]
        area = bbox_w * bbox_h
        score = b[4]
        rank = score * area
        if rank > best_rank:
            best_rank = rank
            best_index = i

    return best_index


def triplet_margin_loss(anchor_emb, positive_emb, negative_emb, margin=0.0):
    dist_12 = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    dist_13 = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    dist_23 = torch.sum((positive_emb - negative_emb) ** 2, dim=1)
    loss = torch.abs(dist_12 - dist_13 + margin) + torch.abs(dist_12 - dist_23 + margin)
    # print('loss.size(): ' + str(loss.size()))
    return loss.mean()


def triplet_prediction_accuracy(anchor_emb, positive_emb, negative_emb):
    print('anchor_emb: ' + str(anchor_emb))
    dist_12 = torch.sum((anchor_emb - positive_emb) ** 2, dim=1)
    print('dist_12: ' + str(dist_12))
    dist_13 = torch.sum((anchor_emb - negative_emb) ** 2, dim=1)
    dist_23 = torch.sum((positive_emb - negative_emb) ** 2, dim=1)
    print('dist_12.lt(dist_13): ' + str(dist_12.lt(dist_13)))
    batch_size = anchor_emb.size(0)
    correct = dist_12.lt(dist_13) * dist_12.lt(dist_23)
    print('correct: ' + str(correct))
    correct_total = correct.view(-1).float().sum()
    print('correct_total: ' + str(correct_total))
    return correct_total * (100.0 / batch_size)


def accuracy(pred, target):
    batch_size = pred.size(0)
    correct = []
    for i in range(batch_size):
        if math.fabs(pred[i].item() - target[i].item()) < 0.5:
            correct += [1.0]
    # correct = torch.abs(pred - target).lt(0.5)
    # correct_total = correct.view(-1).float().sum()  # 0D tensor
    correct_total = sum(correct)
    # return correct_total.item() * (100.0 / batch_size)
    return correct_total * (100.0 / batch_size)
