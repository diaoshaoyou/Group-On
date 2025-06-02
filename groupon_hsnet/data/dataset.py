r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as Sampler
# import albumentations as A
# import albumentations.pytorch
import numpy as np

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.fss import DatasetFSS

# class MyCompose(A.Compose):
#     def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
#         super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)

#     def __call__(self, image, mask):
#         augmented = super().__call__(image=np.array(image), mask=np.array(mask))
#         return augmented['image'], augmented['mask']

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize, augment=False):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize
        # scale_limit = (0.8, 1.25)

        # augmentation = [
        #     A.ToGray(p=0.2),
        #     A.Posterize(p=0.2),
        #     A.Equalize(p=0.2),
        #     A.Sharpen(p=0.2),
        #     A.RandomBrightnessContrast(p=0.2),
        #     A.Solarize(p=0.2),
        #     A.ColorJitter(p=0.2),
        #     A.RandomScale(scale_limit=scale_limit, p=1.),
        #     A.Rotate(limit=10, p=1.),
        #     A.GaussianBlur((5, 5), p=0.5),
        #     A.HorizontalFlip(p=0.5),
        #     A.PadIfNeeded(img_size,
        #                   img_size,
        #                   border_mode=cv2.BORDER_CONSTANT,
        #                   value=[x * 255 for x in cls.img_mean],
        #                   mask_value=0),
        #     A.RandomCrop(img_size, img_size),
        # ]
        # cls.transform=MyCompose([
        #     *(augmentation if augment==True else ()),
        #     A.Resize(img_size, img_size),
        #     A.Normalize(cls.img_mean, cls.img_std),
        #     A.pytorch.transforms.ToTensorV2(),
        # ])
        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, N_q, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        # shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform=cls.transform # if split=='trn' else cls.test_transform
        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize, N_q=N_q)
        train_sampler = Sampler(dataset) if split == 'trn' else None
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, sampler=train_sampler, num_workers=nworker, pin_memory=True)

        return dataloader
