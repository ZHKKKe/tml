import os
import time
import math
import logging
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src import dataset, models, utils


logger = logging.getLogger('main')


args = {
    'exp_id': 'p2a_bn',
    'model': 'bl_p2a',

    'test': False,
    'resume': None,

    'out_path': 'results',

    'trainset': './canvas/social-checkin-prediction/train.csv',
    'valset': './canvas/social-checkin-prediction/validation.csv',
    'testset': './canvas/social-checkin-prediction/test.csv',
    'data_workers': 1, 
    'map_size': (2915, 1982),
    'epochs': 40,
    'batch_size': 64,

    'lr':0.01,
    'lr_decay': 0.1,
    'lr_steps': 0.25,

    'loc_scale': 10,
    'time_scale': 1, 

    'val_freq': 1, 
    'log_freq': 100,
    'checkpoint_freq': 10,
}


def main():
    logger.info('--- experiment: {0} ---\n'.format(args['exp_id']))

    # prepare path
    args['out_path'] = "{root}/{exp_id}/{date:%Y-%m-%d_%H:%M:%S}/".format(
        root=args['out_path'], exp_id=args['exp_id'], date=datetime.datetime.now())
    logger.info('experiment folder: \n  {0} \n'.format(args['out_path']))

    checkpoint_path = os.path.join(args['out_path'], 'ckpt')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # handle multiply GPUs
    gpu_num = torch.cuda.device_count()
    logger.info('GPU: \n  total GPU(s): {0}'.format(gpu_num))
    if gpu_num < 1:
        logger.error('  no GPU be detected')

    args['lr'] *= gpu_num
    args['batch_size'] *= gpu_num
    args['data_workers'] *= gpu_num
    logger.info('  total learn rate: {0}\n'
                '  total batch size: {1}\n'
                '  total data workers: {2}\n'
                .format(args['lr'], args['batch_size'], args['data_workers']))

   # create dataloader
    trainset = dataset.CheckInDataset(args['trainset'], map_size=args['map_size'])
    valset = dataset.CheckInDataset(args['valset'], map_size=args['map_size'])
    testset = dataset.CheckInDataset(args['testset'], map_size=args['map_size'])

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True,
                                               num_workers=args['data_workers'], pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args['batch_size'], shuffle=False,
                                               num_workers=args['data_workers'], pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False,
                                               num_workers=args['data_workers'], pin_memory=True)

    # create network
    model = models.__dict__[args['model']]()
    model = nn.DataParallel(model)
    model = model.cuda()
    logger.info('model: \n  {0}\n'.format(args['model']))
    logger.info(models.model_str(model))

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int(args['lr_steps'] * args['epochs']), gamma=args['lr_decay'])

    # load state from checkpoint
    start_epoch = 0
    if args['resume'] is not None:        
        logger.info('load checkpoint: ' + args['resume'])
        checkpoint = torch.load(args['resume'])
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args['test']:
        logger.info('--- start to test model ---')
        test(model, test_loader, 0)
        return

    # main training loop
    for epoch in range(start_epoch, args['epochs']):
        logger.info('--- start to train epoch: {0} ---'.format(epoch))
        timer = time.time()

        train(model, train_loader, optimizer, epoch)

        if epoch % args['val_freq'] == 0:
            logger.info('--- start to test epoch: {0} ---'.format(epoch))
            test(model, test_loader, epoch)

        if epoch % args['checkpoint_freq'] == 0:
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }

            state_file = os.path.join(checkpoint_path, 'checkpoint.{0}.ckpt'.format(epoch))
            logger.info("--- checkpoint saved to %s ---" % state_file)
            torch.save(state, state_file)

        lr_scheduler.step()
        logger.info('--- epoch in {} seconds ---\n'.format(time.time() - timer))


def train(model, train_loader, optimizer, epoch):
    meters = utils.AverageMeterSet()

    mse_loss = nn.MSELoss()

    model.train()
    for idx, (user_id, in_data, gt_data) in enumerate(train_loader):
        timer = time.time()
        optimizer.zero_grad()
        
        user_id = Variable(user_id).cuda()
        in_data = Variable(in_data).cuda()
        gt_data = Variable(gt_data).cuda()
        
        pred_loc, pred_time = model.forward(in_data)

        gt_loc = gt_data[:, 0:2]
        gt_time = gt_data[:, 2:]

        loc_loss = args['loc_scale'] * mse_loss(pred_loc, gt_loc)
        time_loss = args['time_scale'] * mse_loss(pred_time, gt_time)
        meters.update('loc_loss', loc_loss.data)
        meters.update('time_loss', time_loss.data)

        loss = loc_loss + time_loss
        loss.backward()
        optimizer.step()

        meters.update('batch_time', time.time() - timer)
        if idx % args['log_freq'] == 0:
            logger.info('step: [{0}][{1}/{2}]\t'
                        'loc_loss: {meters[loc_loss]:.4f}\t'
                        'time_loss: {meters[time_loss]:.4f}\t'
                        .format(epoch, idx, len(train_loader), meters=meters))


def test(model, val_loader, epoch):

    """
    Takes entire batch of results and compute the SAD
    """
    def getSAD(vec1, vec2):
        return torch.mean(torch.abs(vec1.data - vec2.data))

    meters = utils.AverageMeterSet()

    model.eval()
    for idx, (user_id, in_data, gt_data) in enumerate(val_loader):
        timer = time.time()

        user_id = Variable(user_id).cuda()
        in_data = Variable(in_data).cuda()
        gt_data = Variable(gt_data).cuda()

        pred_loc, pred_time = model.forward(in_data)

        gt_loc = gt_data[:, 0:2]
        gt_time = gt_data[:, 2:]

        loc_l1 = getSAD(pred_loc, gt_loc)
        time_l1 = getSAD(pred_time, gt_time)
        
        meters.update('loc_l1', loc_l1)
        meters.update('time_l1', time_l1)

        meters.update('batch_time', time.time() - timer)
        if idx % args['log_freq'] == 0:
            logger.info('step: [{0}][{1}/{2}]\t'
                        'loc_l1: {meters[loc_l1]:.4f}\t'
                        'time_l1: {meters[time_l1]:.4f}\t'
                        .format(epoch, idx, len(val_loader), meters=meters))

    logger.info('--- finish test ---\n'
                'loc_l1: {meters[loc_l1]:.4f}\t'
                'time_l1: {meters[time_l1]:.4f}\t'
                .format(epoch, idx, len(val_loader), meters=meters))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()