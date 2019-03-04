"""Train refineDet"""
import argparse
import os
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import sys
sys.path.append("../../../")

import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.model_zoo.refineDet.dataloader import RefineDetDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from gluoncv.model_zoo.refineDet import loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train refineDet networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=320,
                        help="Input data shape, use 320, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./refinedet_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017')
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        anchors, _, _, _, _, _, _ = net(mx.nd.zeros((1, 3, height, width)))

    # stack image, anchor_cls_targets, anchor_box_targets
    # pad real_targets(xmin, ymin, xmax, ymax, label). will return length
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Pad(axis=0, pad_val=-1, ret_length=True))
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(RefineDetDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    # return: img (B, H, W, C); anchor_cls_targets (B, N); anchor_box_targets(B, N, 4);
    # targets(B, P, 5), target_len (B, ). m_i is the num of objects in each img, P is the length after pad.

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    warmup_epochs = float(args.warmup_epochs)
    niters = len(train_dataset) // args.batch_size

    # mbox_loss = gcv.loss.SSDMultiBoxLoss()
    arm_loss = gcv.loss.SSDMultiBoxLoss()   # Anchor Refine Module Loss
    odm_loss = loss.RefineDetMultiBoxLoss()   # Object Detection Module Loss
    arm_ce_metric = mx.metric.Loss('ARM_CrossEntropy')
    arm_smoothl1_metric = mx.metric.Loss('ARM_SmoothL1')
    odm_ce_metric = mx.metric.Loss("ODM_CrossEntropy")
    odm_smoothl1_metric = mx.metric.Loss("ODM_SmoothL1")

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info(20 * "**" + net_name + 20 * "**")
    logger.info(20 * "**" + net_name + 20 * "**")
    logger.info(net)
    logger.info(20 * "**" + len(net_name) * "*" + 20 * "**")
    logger.info(20 * "**" + len(net_name) * "*" + 20 * "**")
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        arm_ce_metric.reset()
        arm_smoothl1_metric.reset()
        odm_ce_metric.reset()
        odm_smoothl1_metric.reset()

        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            if epoch < warmup_epochs:
                T = epoch * niters + i
                new_lr = 1e-6 + (args.lr - 1e-6) * T / (warmup_epochs*niters)
                # print new_lr, T, niters
                trainer.set_learning_rate(new_lr)
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            anchor_cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            anchor_box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            targets = gluon.utils.split_and_load(batch[3][0], ctx_list=ctx, batch_axis=0)
            num_objects = gluon.utils.split_and_load(batch[3][1], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                anchor_cls_preds = []
                anchor_box_preds = []

                obj_cls_targets = []
                obj_box_targets = []

                invalids = []  # to filter negative anchor
                obj_cls_preds = []
                obj_box_preds = []
                for x, target, num_ob in zip(data, targets, num_objects):
                    # x is a sub_batch in a device.
                    # x (b, H, W, C); target (b, P, 5). num_ob is (b, ) is the sum of num_anchors
                    anchor, anchor_cls_pred, anchor_box_pred, refined_anchor, obj_cls_pred, obj_box_pred, invalid = net(x)
                    # shape info: (1, N. 4). (b, N, 2), (b, N, 4), (b, N, 4), (b, N, num_cls+1), (b, N, 4), (b, N, 1)
                    anchor_cls_preds.append(anchor_cls_pred)
                    anchor_box_preds.append(anchor_box_pred)
                    invalid = nd.squeeze(invalid, axis=2)  # (b, N)
                    invalids.append(invalid)

                    # Note: refined_anchor :[xmin, ymin, xmax, ymax]. different from anchor of [x, y, w, h]
                    obj_cls_target, obj_box_target, _ = net.odm_target_generator(refined_anchor, target, num_ob)
                    # (b, N), (b, N, 4), _

                    obj_cls_targets.append(obj_cls_target)
                    obj_box_targets.append(obj_box_target)
                    obj_cls_preds.append(obj_cls_pred)
                    obj_box_preds.append(obj_box_pred)

                # ARM loss
                # for anchor refine module, positive anchor is 1, negative anchor is 0 and ignored is -1.
                arm_sum_loss, arm_cls_loss, arm_box_loss = arm_loss(
                    anchor_cls_preds, anchor_box_preds, anchor_cls_targets, anchor_box_targets)

                # ODM loss
                # odm_cls_targets: >1 for objects(fg); 0 for bg; -1 for ignored;
                odm_sum_loss, odm_cls_loss, odm_box_loss = odm_loss(
                    obj_cls_preds, obj_box_preds, obj_cls_targets, obj_box_targets, invalids
                )

                # sum_loss = arm_sum_loss + odm_sum_loss
                sum_loss = [al + ol for al, ol in zip(arm_sum_loss, odm_sum_loss)]
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            arm_ce_metric.update(0, [l * batch_size for l in arm_cls_loss])
            arm_smoothl1_metric.update(0, [l * batch_size for l in arm_box_loss])
            odm_ce_metric.update(0, [l * batch_size for l in odm_cls_loss])
            odm_smoothl1_metric.update(0, [l * batch_size for l in odm_box_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                arm_name1, arm_loss1 = arm_ce_metric.get()
                arm_name2, arm_loss2 = arm_smoothl1_metric.get()
                odm_name1, odm_loss1 = odm_ce_metric.get()
                odm_name2, odm_loss2 = odm_smoothl1_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, lr={}'.format(
                    epoch, i, batch_size/(time.time()-btic), arm_name1, arm_loss1, arm_name2, arm_loss2, odm_name1, odm_loss1,
                    odm_name2, odm_loss2, trainer.learning_rate))
            btic = time.time()

        # name1, loss1 = ce_metric.get()
        # name2, loss2 = smoothl1_metric.get()
        arm_name1, arm_loss1 = arm_ce_metric.get()
        arm_name2, arm_loss2 = arm_smoothl1_metric.get()
        odm_name1, odm_loss1 = odm_ce_metric.get()
        odm_name2, odm_loss2 = odm_smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), arm_name1, arm_loss1, arm_name2, arm_loss2, odm_name1, odm_loss1, odm_name2, odm_loss2))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('refinedet', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    
    net = get_model(net_name, pretrained_base=True)

    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
