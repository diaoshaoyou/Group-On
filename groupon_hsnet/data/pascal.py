r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np

class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, N_q):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize
        self.N_q=N_q

        self.img_path = os.path.join(datapath, 'PASCAL5i/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'PASCAL5i/SegmentationClassAug/')
        self.transform = transform
        # self.cutmix=CutMix()

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)# if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_names, support_names, class_sample = self.sample_episode(idx)
        query_imgs, query_cmasks, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_names, support_names)

        query_imgs = torch.stack([self.transform(query_img) for query_img in query_imgs])
        # transform:
        # query_transformed = [self.transform(query_img, query_cmask) for query_img, query_cmask in zip(query_imgs, query_cmasks)]
        # query_cmasks = [x[1] for x in query_transformed]
        # query_imgs = torch.stack([x[0] for x in query_transformed])

        query_masks = []
        query_ignore_idxs = []
        for qcmask in query_cmasks:
            if not self.use_original_imgsize:
                qcmask = F.interpolate(qcmask.unsqueeze(0).unsqueeze(0).float(), query_imgs.size()[-2:], mode='nearest').squeeze()
            query_mask, query_ignore_idx = self.extract_ignore_idx(qcmask, class_sample)
            query_masks.append(query_mask)
            query_ignore_idxs.append(query_ignore_idx)
        query_masks = torch.stack(query_masks)
        query_ignore_idxs = torch.stack(query_ignore_idxs)

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        # transform:
        # support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_cmasks)]
        # support_cmasks = [x[1] for x in support_transformed]
        # support_imgs = torch.stack([x[0] for x in support_transformed])

        support_masks = []
        support_ignore_idxs = []
        for scmask in support_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
            support_ignore_idxs.append(support_ignore_idx)
        support_masks = torch.stack(support_masks)
        support_ignore_idxs = torch.stack(support_ignore_idxs)

        batch = {'query_imgs': query_imgs,
                 'query_masks': query_masks,
                 'query_names': query_names,
                 'query_ignore_idxs': query_ignore_idxs,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'support_ignore_idxs': support_ignore_idxs,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def extract_ignore_idx(self, mask, class_id):
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_names, support_names):
        query_imgs = [self.read_img(name) for name in query_names]
        query_masks = [self.read_mask(name) for name in query_names]
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_imgs[0].size

        return query_imgs, query_masks, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        idx=12
        q_name1, class_sample = self.img_metadata[idx]

        # get another N_q-1 query
        query_names={q_name1}
        while True:
            if len(query_names) == self.N_q: break
            q_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if q_name not in query_names: query_names.add(q_name)

        # get support set
        # support_names = []
        support_names = [self.img_metadata_classwise[class_sample][1]]
        # while True:  # keep sampling support set if query == support
        #     support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        #     if support_name not in query_names: support_names.append(support_name)
        #     if len(support_names) == self.shot: break

        if self.N_q == 1:
            query_names_list = [q_name1]
        else:
            query_names.discard(q_name1)
            query_names_list = [q_name1] + list(query_names)
        return query_names_list, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('groupon_hsnet/data/splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            tmp = []
            for data in fold_n_metadata:
                if int(data.split('__')[1]) == 4:
                    tmp.append([data.split('__')[0], int(data.split('__')[1]) - 1])
            return tmp # [[img_name, class], [img_name, class] ...]

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata # [[img_name, class], [img_name, class] ...]

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
        # {class1: [name1, name2...], class2: [name1, name2...], ...}