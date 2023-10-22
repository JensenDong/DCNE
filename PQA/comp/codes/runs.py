import os
import json
import logging
import argparse
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import KGEModel, DCNE
from data import TrainDataset, BatchType, ModeType, DataReader
from data import BidirectionalOneShotIterator

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='runs.py [<args>] [-h | --help]'
    )

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    parser.add_argument('--adv_vanish', action='store_true')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='Rotate3D', type=str)

    parser.add_argument('--length', default=1, type=int)

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-reg', '--regularization', default=0, type=float)
    parser.add_argument('-reg_l', '--reg_level', default=2, type=int)
    parser.add_argument('--p', default=2, type=int)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=8, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=80000, type=int)

    parser.add_argument('--save_checkpoint_steps', default=5000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, f'config_{args.length}hop.json'), 'r') as f:
        args_dict = json.load(f)

    args.model = args_dict['model']
    args.data_path = args_dict['data_path']
    args.hidden_dim = args_dict['hidden_dim']
    args.test_batch_size = args_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    args_dict = vars(args)
    with open(os.path.join(args.save_path, f'config_{args.length}hop.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, f'checkpoint_{args.length}hop')
    )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, f'train_{args.length}hop.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, f'test_{args.length}hop.log')

    logger = logging.getLogger(f"{args.length}")
    file = logging.FileHandler(filename=log_file, mode="w")
    logger.setLevel(logging.INFO)
    file.setLevel(logging.INFO)
    file.setFormatter(formatter)
    logger.addHandler(file)

    return logger


def log_metrics(mode, step, metrics, logger):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logger.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def log_metrics_hop(mode, step, metrics_hop, logger):
    for hop in metrics_hop:
        logger.info(f'{hop}hop: ')
        for metric in metrics_hop[hop]:
            logger.info('%s %s at step %d: %f' % (mode, metric, step, metrics_hop[hop][metric]))


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    logger = set_logger(args)

    data_reader = DataReader(args.data_path, logger, length=args.length)
    num_entity = len(data_reader.entity_dict)
    num_relation = len(data_reader.relation_dict)

    logger.info('Model: {}'.format(args.model))
    logger.info('Data Path: {}'.format(args.data_path))
    logger.info('length: {}'.format(args.length))
    logger.info('Num Entity: {}'.format(num_entity))
    logger.info('Num Relation: {}'.format(num_relation))

    logger.info('Num Train: {}'.format(len(data_reader.train_data)))
    logger.info('Num Valid: {}'.format(len(data_reader.valid_data)))
    logger.info('Num Test: {}'.format(len(data_reader.test_data)))

    if args.model == 'DCNE':
        kge_model = DCNE(num_entity, num_relation, args.hidden_dim, args.gamma, args.p)
    else:
        raise RuntimeError(f"Model {args.model} is not supported!")

    logger.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logger.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    kge_model = torch.nn.DataParallel(kge_model)
    kge_model = kge_model.cuda()

    if args.length > 1:
        logger.info('Loading %s...' % f'checkpoint_{args.length-1}hop')
        checkpoint = torch.load(os.path.join(args.save_path, f'checkpoint_{args.length-1}hop'))
        kge_model.load_state_dict(checkpoint['model_state_dict'])

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.HEAD_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(data_reader, args.negative_sample_size, BatchType.TAIL_BATCH),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logger.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, f'checkpoint_{args.length}hop'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        if args.length == 1:
            logger.info('Randomly Initializing %s Model...' % args.model)
        init_step = 1

    step = init_step

    if args.do_train:
        logger.info('Start Training...')
        logger.info('learning_rate = %s' % current_learning_rate)
    logger.info('init_step = %d' % init_step)
    logger.info('batch_size = %d' % args.batch_size)
    logger.info('hidden_dim = %d' % args.hidden_dim)
    logger.info('gamma = %f' % args.gamma)
    logger.info('adversarial_temperature = %f' % args.adversarial_temperature)
    logger.info('negative_sample_size = %d' % args.negative_sample_size)
    logger.info('p = %d' % args.p)
    logger.info('adv_vanish = %s' % args.adv_vanish)

    if args.do_train:
        training_logs = []

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.module.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logger.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
                if args.adv_vanish:
                    args.adversarial_temperature = 0

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics, logger)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logger.info('Evaluating on Valid Dataset...')
                metrics, metrics_hop = kge_model.module.test_step(kge_model, data_reader, ModeType.VALID, args)
                log_metrics('Valid', step, metrics, logger)

            if args.do_test and step % args.valid_steps == 0:
                logger.info('Evaluating on Test Dataset...')
                metrics, metrics_hop = kge_model.module.test_step(kge_model, data_reader, ModeType.TEST, args)
                log_metrics('Test', step, metrics, logger)
                log_metrics_hop('Test', step, metrics_hop, logger)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logger.info('Evaluating on Valid Dataset...')
        metrics, metrics_hop = kge_model.module.test_step(kge_model, data_reader, ModeType.VALID, args)
        log_metrics('Valid', step, metrics, logger)

    if args.do_test:
        logger.info('Evaluating on Test Dataset...')
        metrics, metrics_hop = kge_model.module.test_step(kge_model, data_reader, ModeType.TEST, args)
        log_metrics('Test', step, metrics, logger)
        log_metrics_hop('Test', step, metrics_hop, logger)

    if args.evaluate_train:
        logger.info('Evaluating on Training Dataset...')
        metrics, metrics_hop = kge_model.test_step(kge_model, data_reader, ModeType.TRAIN, args)
        log_metrics('Train', step, metrics, logger)
        log_metrics_hop('Train', step, metrics_hop, logger)


if __name__ == '__main__':
    args = parse_args()
    for length in range(1, 6):
        args.length = length
        main(args)

