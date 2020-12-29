import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import ImageFile
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import pandas as pd
import os
import random 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor
import torchxrayvision as xrv
import time
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 若能使用cuda，则使用cuda

classes = ('covid19','normal')
def test_folder(file_path):
    TP,TN,FP,FN=0,0,0,0
    pos_files = os.listdir(file_path+'/CT_COVID')
    neg_files = os.listdir(file_path+'/CT_NonCOVID')
    # 把模型转为test型    
    model.eval()
    # 对covid样本，标签为0
    for img in pos_files:
        print(img)
        source_path = file_path+'/CT_COVID/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        img = val_transformer(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率

        value, predicted = torch.max(output.data, 1)
        print(predicted.item())
        # print(value)
        pred_class = classes[predicted.item()]
        print(pred_class)
        # 对covid样本
        if predicted.item() == 0:
            TP += 1
        else:
            FP += 1
    # 对noncovid样本,标签为1
    for img in neg_files:
        # print(img)
        source_path = file_path+'/CT_NonCOVID/'+img
        img = cv2.imread(source_path)  # 读取要预测的图片
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        img = val_transformer(img)
        img = img.to(device)
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是2个分类的概率

        value, predicted = torch.max(output.data, 1)
        print(predicted.item())
        pred_class = classes[predicted.item()]
        # print(pred_class)
        if predicted.item() == 1:
            TN += 1
        else:
            FN += 1

    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP:',TP+FP)
    p = TP / (TP + FP)
    print('precision:',p)
    r = TP / (TP + FN)
    print('recall:',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1:',F1)
    print('acc:',acc)

def val():
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    total = 0
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples[0].to(device), batch_samples[1].to(device)
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            total += data.size(0) #统计总数
            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)  
        print("test data size:",total)        
    return targetlist, scorelist, predlist

def test_demo():
    count_pred = np.zeros(test_dataset.__len__())
    targetlist, scorelist, predlist = val()
        
    count_pred[predlist <= (1/2)] = 0
    count_pred[predlist > (1/2)] = 1
    TP = ((count_pred == 1) & (targetlist == 1)).sum()
    TN = ((count_pred == 0) & (targetlist == 0)).sum()
    FN = ((count_pred == 0) & (targetlist == 1)).sum()
    FP = ((count_pred == 1) & (targetlist == 0)).sum()
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("test accuracy",acc)
    
    print('vote_pred', predlist)
    print('targetlist', targetlist)
        
    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    print('TP+FP',TP+FP)
    p = TP / (TP + FP)
    print('precision',p)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    print('F1',F1)
    AUC = roc_auc_score(targetlist, scorelist)
    print('AUCp', roc_auc_score(targetlist, predlist))
    print('AUC', AUC)
    
    print('\n The average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
    r, p, F1, acc, AUC))

if __name__ == '__main__':
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                     std=[0.33165374, 0.33165374, 0.33165374])
    batchsize=4
    val_transformer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = torchvision.datasets.ImageFolder(root='test_folder',transform=val_transformer)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batchsize, drop_last=False, shuffle=False)

    import torchvision.models as models
    model = models.densenet169(pretrained=True).cuda()
    modelname = 'Dense169-self-trans'
    # modelname = 'efficientNet-b0'
    model.load_state_dict(torch.load('model_backup/Self-Trans.pt'))
    # modelname = 'ResNet50_test'

    # train
    bs = 10
    votenum = 10
    import warnings
    warnings.filterwarnings('ignore')
    test_demo()