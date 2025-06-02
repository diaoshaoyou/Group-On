r""" COCO-20i few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, N_q):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = split if split == 'val2014' else 'train2014'
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.N_q = N_q

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_imgs, query_masks, support_imgs, support_masks, query_names, support_names, class_sample, org_qry_imsize = self.load_frame()

        # query_img = self.transform(query_img)
        # query_mask = query_mask.float()
        # if not self.use_original_imgsize:
        #     query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        # query:
        query_imgs = torch.stack([self.transform(query_img) for query_img in query_imgs])
        for midx, qmask in enumerate(query_masks):
            query_masks[midx] = F.interpolate(qmask.unsqueeze(0).unsqueeze(0).float(), query_imgs.size()[-2:], mode='nearest').squeeze()
        query_masks = torch.stack(query_masks)

        # support:
        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_imgs': query_imgs,
                 'query_masks': query_masks,
                 'query_names': query_names,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        return class_ids

    def build_img_metadata_classwise(self):
        with open('groupon_hsnet/data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        q_name1 = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        # q_img1 = Image.open(os.path.join(self.base_path, q_name1)).convert('RGB')
        # q_mask1 = self.read_mask(q_name1)
        # org_qry_imsize = q_img1.size
        # q_mask1[q_mask1 != class_sample + 1] = 0
        # q_mask1[q_mask1 == class_sample + 1] = 1

        # get another N_q-1 query
        query_names={q_name1}
        while True:
            if len(query_names) == self.N_q: break
            q_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if q_name not in query_names: query_names.add(q_name)

        if self.N_q == 1:
            query_names_list = [q_name1]
        else:
            query_names.discard(q_name1)
            query_names_list = [q_name1] + list(query_names)

        query_imgs = []
        query_masks = []
        for query_name in query_names_list:
            query_imgs.append(Image.open(os.path.join(self.base_path, query_name)).convert('RGB'))
            query_mask = self.read_mask(query_name)
            query_mask[query_mask != class_sample + 1] = 0
            query_mask[query_mask == class_sample + 1] = 1
            query_masks.append(query_mask)
        org_qry_imsize = query_imgs[0].size

        # get support set
        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if support_name not in query_names:
                support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_imgs, query_masks, support_imgs, support_masks, query_names_list, support_names, class_sample, org_qry_imsize
