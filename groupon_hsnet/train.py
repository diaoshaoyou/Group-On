r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
import os

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import pdb
import math


def train(epoch, model, dataloader, optimizer, N_q, stage, training):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        # extract features only once
        support_feats, query_feats = model.module.extract_features(
            batch['query_imgs'], batch['support_imgs'])
        if stage == 0:  # HSNet
            lgt = model(query_feats[0], support_feats[0],
                        batch['support_masks'].squeeze(1))
            first_query_mask = lgt.argmax(dim=1)
            # seg
            seg_loss = model.module.compute_objective(
                lgt, batch['query_masks'][:, 0])
            if training:
                optimizer.zero_grad()
                seg_loss.backward()
                optimizer.step()
            # align
            support_pseudo_logit = model(support_feats[0], query_feats[0],
                                         first_query_mask)
            align_loss = model.module.compute_objective(
                support_pseudo_logit, batch['support_masks'].squeeze(1))
            first_loss = seg_loss + align_loss
            if training:
                optimizer.zero_grad()
                align_loss.backward()
                optimizer.step()
        else:  # stage==1, groupon
            # get query masks from S
            query_masks_from_S = []
            for nq in range(args.N_q):
                lgt = model(
                    query_feats[nq], support_feats[0],
                    batch['support_masks'].squeeze(1))  # (bsz,2,384,384)
                query_masks_from_S.append(lgt.argmax(dim=1))
            # get query masks from pseudo query masks
            for nq in range(N_q):
                # seg：
                q_logit = model(query_feats[nq], support_feats[0],
                                batch['support_masks'].squeeze(1))
                query_logits = [q_logit]
                for remain in range(args.N_q):
                    if remain == nq: continue
                    q_logit = model(query_feats[nq], query_feats[remain],
                                    query_masks_from_S[remain])
                    query_logits.append(q_logit)

                query_logits_agg = model.module.train_aggregate(
                    nq, query_logits)  # (b, 2, h, w)
                query_mask = query_logits_agg.argmax(dim=1)
                if nq == 0:
                    first_query_mask = query_logits_agg.argmax(dim=1)

                seg_loss = model.module.compute_objective(
                    query_logits_agg, batch['query_masks'][:, nq])

                if training:
                    optimizer.zero_grad()
                    seg_loss.backward()
                    optimizer.step()
                # align：
                support_pseudo_logit = model(support_feats[0], query_feats[nq],
                                             query_mask)
                align_loss = model.module.compute_objective(
                    support_pseudo_logit, batch['support_masks'].squeeze(1))
                loss = seg_loss + align_loss
                if nq == 0:
                    first_loss = loss
                if training:
                    optimizer.zero_grad()
                    align_loss.backward()
                    optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(
            first_query_mask, batch, q_id=0)
        average_meter.update(area_inter, area_union, batch['class_id'],
                             first_loss.detach().clone())
        average_meter.write_process(idx,
                                    len(dataloader),
                                    epoch,
                                    write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(
        description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='/data3/zhouhanjing/data')
    parser.add_argument('--benchmark',
                        type=str,
                        default='pascal',
                        choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--nworker', type=int, default=8)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--backbone',
                        type=str,
                        default='resnet101',
                        choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--local_rank', type=int, default=0)
    # self-add
    parser.add_argument('--group_start', type=int, default=0)
    parser.add_argument('--N_q', type=int, default=2)
    parser.add_argument('--load', type=str, default='')
    # parser.add_argument('--aug', type=int, default=1)

    args = parser.parse_args()

    # DDP backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, args.N_q,
                                    False)
    device = torch.device("cuda", args.local_rank)
    model.to(device)

    # Load trained model
    if args.load != "":
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    # Convert to DDP model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=True)

    # Helper classes (for training) initialization
    optimizer1 = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    optimizer2 = optim.Adam(model.parameters(), args.lr)
    Evaluator.initialize()

    # Logger.log_params(model)
    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Dataset initialization
    FSSDataset.initialize(img_size=400,
                          datapath=args.datapath,
                          use_original_imgsize=False)
    dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz,
                                                 args.nworker, args.fold,
                                                 'trn', args.N_q)
    if args.local_rank == 0:
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz,
                                                     args.nworker, args.fold,
                                                     'val', args.N_q)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    # num_50=0
    # num_55=0
    # num_60=0
    for epoch in range(args.niter):
        dataloader_trn.sampler.set_epoch(epoch)
        if epoch < args.group_start: stage = 0
        else: stage = 1
        trn_loss, trn_miou, trn_fb_iou = train(
            epoch,
            model,
            dataloader_trn,
            optimizer1 if stage == 0 else optimizer2,
            args.N_q,
            stage,
            training=True)
        
        if args.local_rank == 0:
            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(
                    epoch,
                    model,
                    dataloader_val,
                    optimizer1 if stage == 0 else optimizer2,
                    args.N_q,
                    stage,
                    training=False)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)
                # if val_miou>=50 and num_50==0:
                #     Logger.save_model_miou(model, epoch, val_miou, 1)
                #     num_50=1
                # elif val_miou>=55 and num_55==0:
                #     Logger.save_model_miou(model, epoch, val_miou, 1)
                #     num_55=1
                # elif val_miou>=60 and num_60==0:
                #     Logger.save_model_miou(model, epoch, val_miou, 1)
                #     num_60=1

            Logger.tbd_writer.add_scalars('data/loss', {
                'trn_loss': trn_loss,
                'val_loss': val_loss
            }, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {
                'trn_miou': trn_miou,
                'val_miou': val_miou
            }, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {
                'trn_fb_iou': trn_fb_iou,
                'val_fb_iou': val_fb_iou
            }, epoch)
            Logger.tbd_writer.flush()
    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info(
            '==================== Finished Training ====================')
