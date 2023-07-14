import os
os.environ["CUDA_MAX_MEM_ALLOC_PERCENT"] = "95"
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import tensor
import torchvision.models as models
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from efficientnet_pytorch import EfficientNet
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.utils import shuffle
from sklearn import preprocessing as p
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import ROC
from torchmetrics import AUROC
import optuna
from optuna.trial import TrialState
from torchvision.transforms import RandomApply, Lambda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
from segmentation_models_pytorch import losses
dice_loss = losses.DiceLoss('binary')
foc_loss = losses.FocalLoss('binary')

import random
random.seed(1)
torch.manual_seed(1)
import random
import torchvision.transforms as T

train_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                       transforms.CenterCrop((224,224)),                                  
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],),
                                       ])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomVerticalFlip(), 
    transforms.RandomRotation([-90, 90]),
])

val_transforms = transforms.Compose([torchvision.transforms.ToTensor(),
                                      transforms.CenterCrop((224,224)),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

class myDataset_train(Dataset):
    def __init__(self, transform=False):
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/train/"
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/train/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        print(file_list)
        print(msk_list)
        msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            print(class_path)
            print(class_name)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*")):
                  self.masks.append(masks_path)
        print(len(self.images))
        print(len(self.targets))
        print(len(self.masks))
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        print(self.images[-100])
        print(self.targets[-100])
        print(self.masks[-100])
        self.class_map = {"HGG_t2" : 0, "LGG_t2": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        masks_ID = self.masks[idx]
        masks_path = self.masks[idx]
        img = np.load(img_path)
        msk = np.load(masks_path)
        img_color = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
        img_tensor = train_transforms(img_color)
        state = torch.get_rng_state()
        img_tensor = aug_transform(img_tensor)
        msk_color = cv2.cvtColor(np.float32(msk), cv2.COLOR_GRAY2RGB)
        msk_tensor = train_transforms(msk_color)
        torch.set_rng_state(state)
        msk_tensor = aug_transform(msk_tensor)
        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)
        class_id_one_hot = F.one_hot(class_id, num_classes=2).float()

        return img_tensor, class_id_one_hot, msk_tensor

class myDataset_val(Dataset):
    def __init__(self, transform=None):
        self.imgs_path = "/home/viktoriia.trokhova/T2_new_MRI_slices/val/"
        self.masks_path = "/home/viktoriia.trokhova/T2_new_Msk_slices/val/"
        file_list = glob.glob(self.imgs_path + "*")
        msk_list = glob.glob(self.masks_path + "*")
        print(file_list)
        print(msk_list)
        msk_list[0], msk_list[1] = msk_list[1], msk_list[0]
        self.images = []
        self.targets = []
        self.masks = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.images.append(img_path)
            for img_path in sorted(glob.glob(class_path + "/*")):
                self.targets.append(class_name)
        for msk_path in msk_list:
            for masks_path in sorted(glob.glob(msk_path + "/*")):
                  self.masks.append(masks_path)
        self.images, self.targets, self.masks = shuffle(self.images, self.targets, self.masks, random_state=101)
        print(len(self.images))
        print(len(self.targets))
        print(len(self.masks))
        print(self.images[-100])
        print(self.targets[-100])
        print(self.masks[-100])
        self.class_map = {"HGG_t2" : 0, "LGG_t2": 1}
        self.img_dim = (224, 224)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        class_name = self.targets[idx]
        masks_path = self.masks[idx]
        img = np.load(img_path)
        msk = np.load(masks_path)
        img_float32 = np.float32(img)
        img_color = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
        img_tensor = val_transforms(img_color)
        msk_float32 = np.float32(msk)
        msk_color = cv2.cvtColor(ms
