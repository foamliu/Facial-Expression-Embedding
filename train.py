import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
# from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from config import device, grad_clip, print_freq, num_workers
from data_gen import FECDataset
from models import FECNet
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, \
    triplet_prediction_accuracy, triplet_margin_loss


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_acc = 0
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = FECNet()
        model = nn.DataParallel(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = FECDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=num_workers)
    test_dataset = FECDataset('test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=num_workers)

    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_accuracy', train_acc, epoch)

        lr = get_learning_rate(optimizer)
        writer.add_scalar('model/learning_rate', lr, epoch)
        print('\nCurrent effective learning rate: {}\n'.format(lr))

        # One epoch's validation
        valid_loss, valid_acc = valid(valid_loader=test_loader,
                                      model=model,
                                      logger=logger)

        writer.add_scalar('model/valid_loss', valid_loss, epoch)
        writer.add_scalar('model/valid_accuracy', valid_acc, epoch)

        # Check if there was an improvement
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_acc, is_best)
        # scheduler.step(epoch)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for i, (img_0, img_1, img_2, margin) in enumerate(train_loader):
        # Move to GPU, if available
        img_0 = img_0.to(device)
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        margin = margin.float().to(device)

        # Forward prop.
        emb0 = model(img_0)
        emb1 = model(img_1)
        emb2 = model(img_2)
        # print(x.size())
        # print('x: ' + str(x))

        # Calculate loss
        loss = triplet_margin_loss(emb0, emb1, emb2, margin)
        # loss = F.triplet_margin_loss(anchor=emb0, positive=emb1, negative=emb2, margin=margin, swap=True)
        acc = triplet_prediction_accuracy(emb0, emb1, emb2)
        # print('x.size(): ' + str(x.size()))
        # print('y.size(): ' + str(y.size()))
        # loss = -y * torch.log(x) - (1 - y) * torch.log(1 - x)
        # print('loss.size(): ' + str(loss.size()))
        # loss = loss.mean()
        # print('loss.size(): ' + str(loss.size()))

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

        # Print status
        if i % print_freq == 0:
            if i % print_freq == 0:
                status = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                         'Accuracy {acc.val:.5f} ({acc.avg:.5f})\t'.format(epoch, i,
                                                                           len(train_loader),
                                                                           loss=losses,
                                                                           acc=accs,
                                                                           )
                logger.info(status)

    return losses.avg, accs.avg


def valid(valid_loader, model, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for i, (img_0, img_1, img_2, margin) in enumerate(valid_loader):
        # Move to GPU, if available
        img_0 = img_0.to(device)
        img_1 = img_1.to(device)
        img_2 = img_2.to(device)
        margin = margin.float().to(device)

        # Forward prop.
        with torch.no_grad():
            emb0 = model(img_0)
            emb1 = model(img_1)
            emb2 = model(img_2)
            # print(x.size())
            # print('x: ' + str(x))

        # Calculate loss
        loss = triplet_margin_loss(emb0, emb1, emb2, margin)
        # loss = F.triplet_margin_loss(anchor=emb0, positive=emb1, negative=emb2, margin=margin, swap=False)
        acc = triplet_prediction_accuracy(emb0, emb1, emb2)
        # loss = -y * torch.log(x) - (1 - y) * torch.log(1 - x)
        # loss = loss.mean()

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

    # Print status
    status = 'Validation\t Loss {loss.avg:.5f}\t Accuracy {acc.avg:.5f}\n'.format(loss=losses, acc=accs)
    logger.info(status)

    return losses.avg, accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
