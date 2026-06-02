import argparse
import math
import os
import random
import time

import numpy as np
import torch
from ignite.metrics import Accuracy

import model.model as Models
from config_file.configs import Config
from dataset.dataloader import Data_Loader
from utils.utils import Logger, Timer, Loss_Accumulator, EarlyStopping
from worker.worker import Model_Train, Model_Finetune, Model_Test

parser = argparse.ArgumentParser()
#Training Setup
parser.add_argument('--mode', default='train', type=str,
                    help='worker mode: train, finetune, test')
parser.add_argument('--signal-length', default=1024, type=int,
                    help='signal length')
parser.add_argument('--data-path', default='/home/pdz/pdz_HDD/code/data', type=str,
                    help='dataset root path')
parser.add_argument('--source-dataset', default='PU_JS/PU_0', type=str,
                    help='source datasets: pretrain')
parser.add_argument('--target-dataset', default='PU_JS/PU_0', type=str,
                    help='target datasets: finetune_test')
parser.add_argument('--percent', default=1, type=float,
                    help='finetune dataset percentage : 1(100%) 0.5(50%) 0.25 0.1')
#pretrain
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs')
parser.add_argument('--lr', '--learning-rate', default=3e-3, type=float,
                    help='learning rate 3e-3')
parser.add_argument('--batchsize', default=128, type=int,
                    help='mini-batch size')
#finetune
parser.add_argument('--ft_epochs', default=100, type=int,
                    help='number of finetune epochs')
parser.add_argument('--ft_batchsize', default=32, type=int,
                    help='mini-batch size')
parser.add_argument('--ft_lr', default=1e-3, type=float,
                    help='learning rate')
#optimizer
parser.add_argument('--weight-decay', default=1e-3, type=float,
                    help='weight decay')
parser.add_argument('--warmup-epochs', default=5, type=int,
                    help='number of warmup epochs')
#logging
parser.add_argument('--logs-save-dir', default='./exp', type=str,
                    help='saving experiments logs')
parser.add_argument('--run', default='PU_0', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=12, type=int, help='seed value')

#Hyperparameters Settings
parser.add_argument('--classes', default=13, type=int, help='class value')

parser.add_argument('--lamda', default=1, type=float, help='loss weight value 1')
parser.add_argument('--lamda2', default=2, type=float, help='loss weight value 2')

parser.add_argument('--Align_T', default=0.5, type=float, help='')
parser.add_argument('--Scaling', default=10, type=float, help='')
parser.add_argument('--DPC_T', default=0.1, type=float, help='')
parser.add_argument('--APD_T', default=1.0, type=float, help='')

parser.add_argument('--snr_db', default=-4, type=int, help='add_noises value 0, -2, -4')

parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")

def main():
    args = parser.parse_args()
    configs = Config()
    configs.num_class = args.classes
    configs.batch_size = args.batchsize
    configs.lamda = args.lamda
    configs.lamda2 = args.lamda2


    configs.Align_T = args.Align_T
    configs.Scaling = args.Scaling
    configs.DPC_T = args.DPC_T
    configs.APD_T = args.APD_T


    # # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    #####################################################

    '''save dir'''
    experiment_log_dir = os.path.join(args.logs_save_dir, args.run)
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)

    '''logger'''
    log_name = os.path.join(
        experiment_log_dir,
        '{}_{}_seed{}_p_logs_{}.log'.format(
            args.mode,
            args.percent,
            args.seed,
            time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        )
    )
    logger = Logger(log_name)

    '''logging'''
    logger.debug('*' * 50)
    logger.debug('Mode: {}'.format(args.mode))
    logger.debug('Dataset: {}'.format(args.data_path))
    logger.debug('Signal length: {}'.format(args.signal_length))
    logger.debug('Source Dataset: {}'.format(args.source_dataset))
    logger.debug('Target Dataset: {}'.format(args.target_dataset))
    logger.debug('Percent: {}'.format(args.percent))
    logger.debug('Epochs: {}'.format(args.epochs))
    logger.debug('Batchsize: {}'.format(args.batchsize))
    logger.debug('lr: {}'.format(args.lr))
    logger.debug('ft_epochs: {}'.format(args.ft_epochs))
    logger.debug('ft_batchsize: {}'.format(args.ft_batchsize))
    logger.debug('ft_lr: {}'.format(args.ft_lr))
    logger.debug('weight_decay: {}'.format(args.weight_decay))
    logger.debug('warmup_epochs: {}'.format(args.warmup_epochs))
    logger.debug('feature_length: {}'.format(configs.length))
    logger.debug('lamda: {}'.format(configs.lamda))
    logger.debug('lamda2: {}'.format(configs.lamda2))
    logger.debug('percent: {}'.format(args.percent))

    logger.debug('Align_T: {}'.format(configs.Align_T))
    logger.debug('Scaling: {}'.format(configs.Scaling))
    logger.debug('DPC_T: {}'.format(configs.DPC_T))
    logger.debug('APD_T: {}'.format(configs.APD_T))

    logger.debug('PDC-CFN')
    logger.debug('*' * 50)

    '''GPU or CPU'''
    if torch.cuda.is_available():
        device = configs.device
    else:
        device = torch.device('cpu')
    logger.debug('Device: {}'.format(device))


    '''data-path'''
    source_path = f"{args.data_path}/{args.source_dataset}"
    target_path = f"{args.data_path}/{args.target_dataset}"

    '''load model & data'''
    if args.mode == 'train':
        model = Models.train_model(configs)
        data_iter = Data_Loader(source_path, args, args.mode)
    elif args.mode == 'finetune':
        model = Models.finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    else:
        model = Models.finetune_model(configs)
        data_iter = Data_Loader(target_path, args, args.mode)
    model.to(device)
    logger.debug('Model Loaded')
    logger.debug('Data Loaded')
    ######################################################
    '''optimizer & scheduler'''
    if args.mode == 'train':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)
    warm_up_with_cosine_lr = lambda epoch: ((epoch + 1) / args.warmup_epochs) if epoch < args.warmup_epochs \
        else 0.5 * (math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi) + 1) \
        if 0.5 * (math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi) + 1) > 0.1 else 0.1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    logger.debug('*' * 50)

    '''work'''
    main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs)


    logger.debug('*' * 50)


def main_worker(model, optimizer, scheduler, data_iter, device, logger, experiment_log_dir, args, configs):
    logger.debug('Worker Started')
    '''timer'''
    timer = Timer()

    '''training mode'''
    if args.mode == 'train':
        logger.debug('Training Started')

        '''record loss & acc'''
        loss = Loss_Accumulator()

        '''early_stopping'''
        early_stopping = EarlyStopping(patience=args.patience)

        '''epochs'''
        for epoch in range(args.epochs):
            timer.start()
            train_loss = Model_Train(model, optimizer, scheduler, data_iter, device,loss, configs)
            timer.stop()

            logger.debug(
                '\nTraining Epoch: {}. Loss: {}.'.format(
                    epoch, train_loss))

            early_stopping(train_loss)

            if early_stopping.early_stop:
                logger.debug("Early stopping triggered")
                break

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'train_state_dict.pt'))

        logger.debug('\nTrained model is saved')

    '''fientune mode'''
    if args.mode == 'finetune':
        logger.debug('Finetuning Stated')

        '''load pre-train parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'train_state_dict.pt')
        saved_model = torch.load(load_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in saved_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        '''record loss & acc'''
        loss = Loss_Accumulator()
        acc = Accuracy(device)

        '''epochs'''
        for epoch in range(args.ft_epochs):
            timer.start()
            finetune_loss, finetune_acc = Model_Finetune(model, optimizer, scheduler, data_iter, device, loss, acc)
            timer.stop()
            logger.debug('\nFinetuning Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, finetune_loss, finetune_acc))

        '''save model'''
        save_path = os.path.join(experiment_log_dir, 'saved_model')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'finetune_{}_state_dict.pt'.format(args.percent)))

        logger.debug('\nFinetuned model is saved')

    '''test mode'''
    if args.mode == 'test':
        logger.debug('Testing Stated')

        '''load finetune parameters'''
        load_path = os.path.join(experiment_log_dir, 'saved_model', 'finetune_{}_state_dict.pt'.format(args.percent))
        model.load_state_dict(torch.load(load_path))

        '''record acc'''
        acc = Accuracy(device)

        '''testing'''
        timer.start()
        test_acc = Model_Test(model, data_iter, device, acc,logger)
        timer.stop()
        logger.debug('\nTest Accuracy: {}'.format(test_acc))

    logger.debug('\nWork Time is: {} sec'.format(timer.sum()))
    logger.debug('\nWoker Finished')


if __name__ == '__main__':
    main()
