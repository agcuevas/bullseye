from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import accuracy_score, roc_auc_score
from efficientnet_pytorch import EfficientNet
from torchvision import models
import pdb
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import matplotlib.pyplot as plt

import pickle 

# def list_files(path:Path):
#     return [o for o in path.iterdir()]

# train_fnames = list_files(path/'train')
df = pd.read_csv('c:/work/train.csv')
df.head()

bs=100

def get_augmentations(p=0.5):
    imagenet_stats = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}
    train_tfms = A.Compose([
        A.Cutout(p=p),
        A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=p),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=p),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=p),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=p), 
        ToTensor(normalize=imagenet_stats)
        ])
    
    test_tfms = A.Compose([
        ToTensor(normalize=imagenet_stats)
        ])
    return train_tfms, test_tfms

def get_train_val_split(df):
    #Remove Duplicates
    df = df[df.tfrecord != -1].reset_index(drop=True)
    train_tf_records = list(range(len(df.tfrecord.unique())))[:12]
    split_cond = df.tfrecord.apply(lambda x: x in train_tf_records)
    train_df = df[split_cond].reset_index()
    valid_df = df[~split_cond].reset_index()
    return train_df,valid_df

class MelanomaDataset(Dataset):
    def __init__(self,df,im_path,transforms=None,is_test=False):
        self.df = df
        self.im_path = im_path
        self.transforms = transforms
        self.is_test = is_test
        
    def __getitem__(self,idx):
        img_path = f"{self.im_path}/{self.df.iloc[idx]['image_name']}.jpg"
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(**{"image": np.array(img)})["image"]
            
        if self.is_test:
            return img
        target = self.df.iloc[idx]['target']
        return img,torch.tensor([target],dtype=torch.float32)
    
    def __len__(self):
        return self.df.shape[0]
    
class MelanomaEfficientNet(nn.Module):
    def __init__(self,model_name='efficientnet-b0',pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = getattr(self.backbone,'_fc').in_features
        self.classifier = nn.Linear(in_features,1)

    def forward(self,x):
        features = self.pool_type(self.backbone.extract_features(x),1)
        features = features.view(x.size(0),-1)
        return self.classifier(features)
    
def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_model(model_name='efficientnet-b0',lr=1e-5,wd=0.01,freeze_backbone=False,opt_fn=torch.optim.AdamW,device=None):
    device = device if device else get_device()
    model = MelanomaEfficientNet(model_name=model_name)
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    opt = opt_fn(model.parameters(),lr=lr,weight_decay=wd)
    model = model.to(device)
    return model, opt

def training_step(xb,yb,model,loss_fn,opt,device,scheduler):
    xb,yb = xb.to(device), yb.to(device)
    out = model(xb)
    opt.zero_grad()
    loss = loss_fn(out,yb)
    loss.backward()
    opt.step()
    scheduler.step()
    return loss.item()
    
def validation_step(xb,yb,model,loss_fn,device):
    xb,yb = xb.to(device), yb.to(device)
    out = model(xb)
    loss = loss_fn(out,yb)
    out = torch.sigmoid(out)
    return loss.item(),out

def get_data(train_df,valid_df,train_tfms,test_tfms,bs):
    train_ds = MelanomaDataset(df=train_df,im_path='c:/work/train',transforms=train_tfms)
    valid_ds = MelanomaDataset(df=valid_df,im_path='c:/work/train',transforms=test_tfms)
    train_dl = DataLoader(dataset=train_ds,batch_size=bs,shuffle=True,num_workers=4)
    valid_dl = DataLoader(dataset=valid_ds,batch_size=bs*2,shuffle=False,num_workers=4)
    return train_dl,valid_dl

def fit(epochs,model,train_dl,valid_dl,opt,device=None,loss_fn=F.binary_cross_entropy_with_logits):
    
    device = device if device else get_device()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(train_dl)*epochs)
    val_rocs = [] 
    
    #Creating progress bar
    mb = master_bar(range(epochs))
    mb.write(['epoch','train_loss','valid_loss','val_roc'],table=True)

    for epoch in mb:    
        trn_loss,val_loss = 0.0,0.0
        val_preds = np.zeros((len(valid_dl.dataset),1))
        val_targs = np.zeros((len(valid_dl.dataset),1))
        
        #Training
        model.train()
        
        #For every batch 
        for xb,yb in progress_bar(train_dl,parent=mb):
            trn_loss += training_step(xb,yb,model,loss_fn,opt,device,scheduler) 
        trn_loss /= mb.child.total

        #Validation
        model.eval()
        with torch.no_grad():
            for i,(xb,yb) in enumerate(progress_bar(valid_dl,parent=mb)):
                loss,out = validation_step(xb,yb,model,loss_fn,device)
                val_loss += loss
                bs = xb.shape[0]
                val_preds[i*bs:i*bs+bs] = out.cpu().numpy()
                val_targs[i*bs:i*bs+bs] = yb.cpu().numpy()

        val_loss /= mb.child.total
        val_roc = roc_auc_score(val_targs.reshape(-1),val_preds.reshape(-1))
        val_rocs.append(val_roc)

        mb.write([epoch,f'{trn_loss:.6f}',f'{val_loss:.6f}',f'{val_roc:.6f}'],table=True)
    return model,val_rocs

df = pd.read_csv('c:/work/train.csv')
train_df,valid_df = get_train_val_split(df)
train_tfms,test_tfms = get_augmentations(p=0.5)
train_dl,valid_dl = get_data(train_df,valid_df,train_tfms,test_tfms,bs)
model, opt = get_model(model_name='efficientnet-b5',lr=1e-4,wd=1e-4)

model,val_rocs = fit(10,model,train_dl,valid_dl,opt)
torch.save(model.state_dict(),f'effb5.pth')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

test_tfms = A.Compose([
    A.RandomRotate90(p=p),
        A.Flip(p=p),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=50,
                val_shift_limit=50)
        ], p=p),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=p),
    ToTensor(normalize=imagenet_stats)
    ])

test_df = pd.read_csv('c:/work/test.csv')

model, opt = get_model(model_name='efficientnet-b5',lr=1e-4,wd=1e-4)
model.load_state_dict(torch.load(f'../input/melanomaefficientnetb5/effb5.pth',map_location=device))

#Testing with lighter augmentation
test_ds = MelanomaDataset(df=test_df,im_path=path/'test',transforms=test_tfms,is_test=True)
test_dl = DataLoader(dataset=test_ds,batch_size=bs*2,shuffle=False,num_workers=4)

def get_preds(model,device=None,tta=3):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    preds = np.zeros(len(test_ds))
    for tta_id in range(tta):
        test_preds = []
        with torch.no_grad():
            for xb in test_dl:
                xb = xb.to(device)
                out = model(xb)
                out = torch.sigmoid(out)
                test_preds.extend(out.cpu().numpy())
            preds += np.array(test_preds).reshape(-1)
        print(f'TTA {tta_id}')
    preds /= tta
    return preds

#Changing tta to 25 from 10
preds = get_preds(model,tta=25)  

