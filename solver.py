from collections import OrderedDict
import torch
from src.metrics import iou_score
from src.utils import AverageMeter


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    model.train()

    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        iou, dice, _, _, _, _, _ = iou_score(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)
                        ])


def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'SE':AverageMeter(),
                   'PC':AverageMeter(),
                   'F1':AverageMeter(),
                   'SP':AverageMeter(),
                   'ACC':AverageMeter()
                   }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            iou, dice, SE, PC, F1, SP, ACC = iou_score(output, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['SE'].update(SE, input.size(0))
            avg_meters['PC'].update(PC, input.size(0))
            avg_meters['F1'].update(F1, input.size(0))
            avg_meters['SP'].update(SP, input.size(0))
            avg_meters['ACC'].update(ACC, input.size(0))

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('SE', avg_meters['SE'].avg),
                        ('PC', avg_meters['PC'].avg),
                        ('F1', avg_meters['F1'].avg),
                        ('SP', avg_meters['SP'].avg),
                        ('ACC', avg_meters['ACC'].avg)
                        ])
