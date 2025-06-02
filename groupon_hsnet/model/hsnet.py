r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
import numpy as np
import pdb

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from common.evaluation import Evaluator
from .attention import AttentionWeight


class HypercorrSqueezeNetwork(nn.Module):

    def __init__(self, backbone, N_q, use_original_imgsize):
        super(HypercorrSqueezeNetwork, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.N_q = N_q
        self.use_original_imgsize = use_original_imgsize
        self.all_masks={}
        
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(
            add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add,
                           [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(
            self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.atten = AttentionWeight()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_feats, support_feats, support_mask):

        with torch.no_grad():
            # query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            # support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats,
                                              support_mask.clone())
            corr = Correlation.multilayer_correlation(query_feats,
                                                      support_feats,
                                                      self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask,
                                       self.support_imgs.size()[-2:],
                                       mode='bilinear',
                                       align_corners=True)

        return logit_mask

    def extract_features(self, query_imgs, support_imgs):
        """
        Params:
            query_imgs: shape=(b, N_q, c, h, w)
            support_imgs: shape=(b, nshot, c, h, w)
            query_feats_Nq: shape=(N_q, block_num, b, c, h, w)
            support_feats: shape=(nshot, block_num, b, c, h, w)
        """
        self.support_imgs = support_imgs
        ### extract features ###
        query_feats_Nq = []  # shape=(N_q, block_num, b, c, h, w)
        support_feats_nshot = []  # shape=(nshot, block_num, b, c, h, w)
        nshot = support_imgs.shape[1]
        with torch.no_grad():
            for nq in range(self.N_q):
                query_feats_Nq.append(
                    self.extract_feats(query_imgs[:, nq], self.backbone,
                                       self.feat_ids, self.bottleneck_ids,
                                       self.lids))
            for ns in range(nshot):
                support_feats_nshot.append(
                    self.extract_feats(support_imgs[:, ns], self.backbone,
                                       self.feat_ids, self.bottleneck_ids,
                                       self.lids))
        self.support_feats_nshot=support_feats_nshot
        self.query_feats_Nq = query_feats_Nq
        # pdb.set_trace()
        return support_feats_nshot, query_feats_Nq

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(),
                                 feature.size()[2:],
                                 mode='bilinear',
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, agg):
        """ Perform multiple prediction given (nshot) number of different support sets
        Params:
            query_logits_agg: shape=(bsz, 2, h, w)
            logit_mask_nshot: sum of preds, shape=(bsz, h, w)
        """
        logit_mask_nshot = 0
        support_feats_nshot, query_feats_Nq = self.extract_features(
            batch['query_imgs'], batch['support_imgs'])
        for s_idx in range(nshot):
            if agg == 'mean':
                query_logits = []
                for nq in range(self.N_q):
                    lgt_from_S = self(query_feats_Nq[nq],
                                      support_feats_nshot[s_idx],
                                      batch['support_masks'][:, s_idx])
                    mask_from_S = lgt_from_S.argmax(dim=1)
                    if nq == 0:
                        query_logits.append(lgt_from_S)  # Q1s
                    else:
                        lgt = self(query_feats_Nq[0], query_feats_Nq[nq],
                                   mask_from_S)
                        query_logits.append(lgt)  # Q1?
                query_logits_agg = sum(query_logits) / self.N_q
            elif agg == 'origin':
                query_logits_agg = self(query_feats_Nq[0],
                                        support_feats_nshot[s_idx],
                                        batch['support_masks'][:, s_idx])
            elif agg == 'attention':
                query_logits = []
                weights=[]
                for nq in range(self.N_q):
                    lgt_from_S = self(query_feats_Nq[nq],
                                      support_feats_nshot[s_idx],
                                      batch['support_masks'][:, s_idx])
                    mask_from_S = lgt_from_S.argmax(dim=1)
                    if nq == 0:
                        query_logits.append(lgt_from_S)  # Q1s
                        weights.append(self.atten(self.query_feats_Nq[0][self.stack_ids[1]],
                        self.support_feats_nshot[0][self.stack_ids[1]]))
                    else:
                        lgt = self(query_feats_Nq[0], query_feats_Nq[nq],
                                   mask_from_S)
                        query_logits.append(lgt)  # Q1?
                        weights.append(self.atten(self.query_feats_Nq[0][self.stack_ids[1]],
                        self.query_feats_Nq[nq][self.stack_ids[1]]))
                Sum=[]
                for nq in range(self.N_q):
                    Sum.append(query_logits[nq]*weights[nq])
                query_logits_agg = sum(Sum) / sum(weights)

            if self.use_original_imgsize:
                org_qry_imsize = tuple([
                    batch['org_query_imsize'][1].item(),
                    batch['org_query_imsize'][0].item()
                ])
                query_logits_agg = F.interpolate(query_logits_agg,
                                                 org_qry_imsize,
                                                 mode='bilinear',
                                                 align_corners=True)

            logit_mask_nshot += query_logits_agg.argmax(
                dim=1).clone()  # sum of preds
            if nshot == 1: return logit_mask_nshot

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_nshot.size(0)  # (b, h, w)
        max_vote = logit_mask_nshot.view(
            bsz, -1).max(dim=1)[0]  # overlap most times
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_nshot.float() / max_vote
        # max_vote==1: preds have no overlap, otherwise, som pixels overlap
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1
        # 若某像素点重叠次数<1/2最大重叠次数，则该点视为背景；反之视为前景
        return pred_mask

    def predict_mask_nshot2(self, batch, nshot, agg):
        """ Perform multiple prediction given (nshot) number of different support sets
        Params:
            query_logits_agg: shape=(bsz, 2, h, w)
            logit_mask_nshot: sum of preds, shape=(bsz, h, w)
        """
        logit_mask_nshot = 0
        support_feats_nshot, query_feats_Nq = self.extract_features(
            batch['query_imgs'], batch['support_imgs'])
        for s_idx in range(nshot):
            if agg == 'mean':
                query_logits = []
                for nq in range(self.N_q):
                    if nq == 0:
                        lgt_from_S = self(query_feats_Nq[nq],
                                          support_feats_nshot[s_idx],
                                          batch['support_masks'][:, s_idx])
                        mask_from_S = lgt_from_S.argmax(dim=1)
                        query_logits.append(lgt_from_S)  # Q1s
                    else:
                        if batch['query_names'][nq][0] in self.all_masks.keys(
                        ):
                            lgt = self(
                                query_feats_Nq[0], query_feats_Nq[nq],
                                self.all_masks[batch['query_names'][nq][0]])
                        else:
                            lgt_from_S = self(query_feats_Nq[nq],
                                              support_feats_nshot[s_idx],
                                              batch['support_masks'][:, s_idx])
                            mask_from_S = lgt_from_S.argmax(dim=1)
                            lgt = self(query_feats_Nq[0], query_feats_Nq[nq],
                                       mask_from_S)
                        query_logits.append(lgt)  # Q1?
                query_logits_agg = sum(query_logits) / self.N_q
                self.all_masks[batch['query_names'][0]
                               [0]] = query_logits_agg.argmax(dim=1).clone()
            elif agg == 'origin':
                query_logits_agg = self(query_feats_Nq[0],
                                        support_feats_nshot[s_idx],
                                        batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([
                    batch['org_query_imsize'][1].item(),
                    batch['org_query_imsize'][0].item()
                ])
                query_logits_agg = F.interpolate(query_logits_agg,
                                                 org_qry_imsize,
                                                 mode='bilinear',
                                                 align_corners=True)

            logit_mask_nshot += query_logits_agg.argmax(
                dim=1).clone()  # sum of preds
            if nshot == 1: return logit_mask_nshot

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_nshot.size(0)  # (b, h, w)
        max_vote = logit_mask_nshot.view(
            bsz, -1).max(dim=1)[0]  # overlap most times
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_nshot.float() / max_vote
        # max_vote==1: preds have no overlap, otherwise, som pixels overlap
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1
        # 若某像素点重叠次数<1/2最大重叠次数，则该点视为背景；反之视为前景
        return pred_mask

    def test_aggregate(self, query_logit):
        pred_agg = 0
        for nq in range(self.N_q):
            pred_agg = pred_agg + query_logit[nq].argmax(dim=1).clone()
        # bsz = pred_agg.size(0)  # (b, h, w)
        # max_vote = pred_agg.view(bsz, -1).max(dim=1)[0]  # overlap most times
        # max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        # max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        # pred_mask = pred_agg.float() / max_vote
        # pred_mask[pred_mask < 0.5] = 0
        # pred_mask[pred_mask >= 0.5] = 1
        pred_agg[pred_agg >= 1] = 1
        return pred_agg  # (bsz, h, w)

    def train_aggregate(self, cur_q, query_logit):
        """
        Params:
            cur_q: current query id 
            query_logit: all logits of one query, shape=(N_q, b, 2, h, w)
        """
        bsz = query_logit[0].size(0)
        weight = torch.ones((self.N_q, bsz))  # shape=(N_q, b)
        query_logit_trans = torch.stack(query_logit).transpose(
            0, 1)  # shape=(b, N_q, 2, h, w)
        tmp = np.array(range(self.N_q))
        remain_index = tmp[tmp != cur_q]  # query index except for 'cur_q'
        for nq in range(self.N_q):
            if nq == 0:
                atten_weight = self.atten(
                        self.query_feats_Nq[cur_q][self.stack_ids[1]],
                        self.support_feats_nshot[0][self.stack_ids[1]])
                # continue  # mask predicted by S, weight=1, no need to compute weight
            # area_inter, area_union = Evaluator.classify_prediction(
            #         query_masks_from_S[remain_index[nq - 1]],
            #         batch,
            #         q_id=remain_index[nq - 1])
            # iou = (area_inter[1].float() / area_union[1].float())  # shape=(b,)
            # iou[iou < self.threshold]=0
            else:
                atten_weight = self.atten(
                        self.query_feats_Nq[cur_q][self.stack_ids[1]],
                        self.query_feats_Nq[remain_index[nq -
                                                        1]][self.stack_ids[1]])
            weight[nq] = atten_weight
        weight_trans = weight.transpose(0, 1)  # shape=(b, N_q)
        weight_trans = F.softmax(weight_trans, dim=1)

        query_logit_agg = []
        for b in range(bsz):
            Sum = []
            for nq in range(self.N_q):
                Sum.append(query_logit_trans[b, nq, :] * weight_trans[b][nq])
            query_logit_agg.append(sum(Sum) / sum(weight_trans[b]))
        query_logit_agg = torch.stack(query_logit_agg)
        return query_logit_agg  # shape=(b, 2, h, w)

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval(
        )  # to prevent BN from learning data statistics with exponential averaging
