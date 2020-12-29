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
from shutil import copyfile
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import albumentations as albu
from albumentations.pytorch import ToTensor
from catalyst.data import Augmentor
import torchxrayvision as xrv
import os
import time

batchsize=10
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        # image = Image.open(img_path)
        # print(image.mode)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

alpha = None
device = 'cuda'
Loss_list = []
Accuracy_list = []
def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    t1 = time.time()
    for batch_index, batch_samples in enumerate(train_loader):
        
        # move data to device
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        # data = data[:, 0, :, :]
        # data = data[:, None, :, :]
        # data, targets_a, targets_b, lam = mixup_data(data, target.long(), alpha, use_cuda=True)
        optimizer.zero_grad()
        output = model(data)
        
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        # loss = mixup_criterion(criteria, output, targets_a, targets_b, lam)
        train_loss += criteria(output, target.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
    

        if batch_index % bs == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ bs))
    t2 = time.time()
    # Display progress and write to tensorboard
    writer.add_scalar("Train/loss",train_loss/len(train_loader.dataset),epoch)
    writer.add_scalar("Train/acc",100.0 * train_correct / len(train_loader.dataset),epoch)
    # 用matplotlib出图
    # Loss_list.append(train_loss/len(train_loader.dataset))
    # Accuracy_list.append(100.0 * train_correct / len(train_loader.dataset))
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time cost:{}\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset), t2-t1))
    f = open('model_result/{}.txt'.format(modelname), 'a+')
    f.write('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time cost:{}\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset), t2-t1))
    f.write('\n')
    f.close()

def val(epoch):
    
    model.eval()
    test_loss = 0
    correct = 0
    results = []
    
    criteria = nn.CrossEntropyLoss()
    # Don't update model
    with torch.no_grad():
        tpr_list = []
        fpr_list = []
        
        predlist=[]
        scorelist=[]
        targetlist=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            targetcpu=target.long().cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
        
        writer.add_scalar("Test/loss",test_loss,epoch)   
        writer.add_scalar("Test/acc",((100*correct)/len(val_loader.dataset)),epoch)     

    return targetlist, scorelist, predlist

def test_demo():
    count_pred = np.zeros(valset.__len__())
    targetlist, scorelist, predlist = val(1)
        
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

def train_demo():
    

    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())
    count_pred = np.zeros(valset.__len__())
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    #### test时关闭 ############
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #scheduler = StepLR(optimizer, step_size=1)

    total_epoch = 100
        
    for epoch in range(1, total_epoch+1):
        train(optimizer, epoch)
        
        targetlist, scorelist, predlist = val(epoch)
        
        count_pred[predlist <= (1/2)] = 0
        count_pred[predlist > (1/2)] = 1
        TP = ((count_pred == 1) & (targetlist == 1)).sum()
        TN = ((count_pred == 0) & (targetlist == 0)).sum()
        FN = ((count_pred == 0) & (targetlist == 1)).sum()
        FP = ((count_pred == 1) & (targetlist == 0)).sum()
        acc = (TP + TN) / (TP + TN + FP + FN)
        print("test accuracy",acc)
        writer.add_scalar("Test/acc",acc,epoch)
        vote_pred = vote_pred + predlist 
        vote_score = vote_score + scorelist 
        if epoch % votenum == 0:
            
            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            vote_score = vote_score/votenum
            
            print('vote_pred', vote_pred)
            print('targetlist', targetlist)
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            
            
            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)
            
            
            if epoch == total_epoch:
                torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))  
            

            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))

            f = open('model_result/{}.txt'.format(modelname), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))
            f.close()

"""模型定义"""
### DenseNet    
class DenseNetModel(nn.Module):

    def __init__(self):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        super(DenseNetModel, self).__init__()

        self.dense_net = xrv.models.DenseNet(num_classes=2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.dense_net(x)
        return logits
    
# model = DenseNetModel().cuda()
# modelname = 'DenseNet_medical'

### SimpleCNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__() # b, 3, 32, 32
        layer1 = torch.nn.Sequential() 
        layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))
 
        #b, 32, 32, 32
        layer1.add_module('relu1', torch.nn.ReLU(True)) 
        layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)) # b, 32, 16, 16 //池化为16*16
        self.layer1 = layer1
        layer4 = torch.nn.Sequential()
        layer4.add_module('fc1', torch.nn.Linear(401408, 2))       
        self.layer4 = layer4
 
    def forward(self, x):
        conv1 = self.layer1(x)
        fc_input = conv1.view(conv1.size(0), -1)
        fc_out = self.layer4(fc_input)
 
# model = SimpleCNN().cuda()
# modelname = 'SimpleCNN'
def draw_train(total_epoch):
    plt.subplot(2,1,1)
    plt.plot(range(0,total_epoch),Accuracy_list,'o-')
    plt.title('Train Accuracy vs. epoches')
    plt.ylabel('Train Accuracy')
    plt.subplot(2,1,2)
    plt.plot(range(0,total_epoch),Loss_list,'-.')
    plt.xlabel('Train Loss vs. epoches')
    plt.ylabel('Train Loss')
    plt.show()
    plt.savefig("model_result/train.jpg")




if __name__ == '__main__':
    torch.cuda.empty_cache()

    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
        # random brightness and random contrast
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    trainset = CovidCTDataset(root_dir='new_data',
                              txt_COVID='Data-split/COVID/trainCT.txt', # trainCT.txt中包含了原数据集中val部分
                              txt_NonCOVID='Data-split/NonCOVID/trainCT.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='new_data',
                              txt_COVID='Data-split/COVID/testCT.txt',
                              txt_NonCOVID='Data-split/NonCOVID/testCT.txt',
                              transform= val_transformer)
    
    print("trainset length:",trainset.__len__())
    # print("trainset classes:",trainset.classes_to_idx)
    print("valset length:",valset.__len__())

    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    
    # model = DenseNetModel().cuda()
    # modelname = 'DenseNet_medical'


    # ### ResNet18
    # import torchvision.models as models
    # model = models.resnet18(pretrained=True).cuda()
    # modelname = 'ResNet18_on_CT'


    ### Dense121
    # import torchvision.models as models
    # model = models.densenet121(pretrained=True).cuda()
    # modelname = 'Dense121_543train'
    # model.load_state_dict(torch.load('model_backup/Dense121.pt'))
    # modelname = 'Dense121_on_CT'

    # ### Dense169
    # import torchvision.models as models
    # model = models.densenet169(pretrained=True).cuda()
    # modelname = 'Dense169_on_CT'
    # model.load_state_dict(torch.load('model_backup/Self-Trans.pt'))
    ### Resnet50
    # import torchvision.models as models
    # model = models.resnet50(pretrained=True).cuda()
    # modelname = 'ResNet50_on_CT'
    # model.load_state_dict(torch.load('model_backup/ResNet50.pt'))
    # modelname = 'ResNet50_test'
    
    # test_demo()
    # ### VGGNet
    # import torchvision.models as models
    # model = models.vgg16(pretrained=True)
    # model = model.cuda()
    # modelname = 'vgg16_on_CT'


    # In[139]:


    # # ### efficientNet
    from efficientnet_pytorch import EfficientNet
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model = model.cuda()
    modelname = 'efficientNet-b0_on_CT'

    # train
    bs = 10
    votenum = 10
    import warnings
    warnings.filterwarnings('ignore')
    # 定义一个writer，记录中间数据显示
    writer = SummaryWriter("run_logs/{}".format(modelname))
    # test_demo()
    vote_pred = np.zeros(valset.__len__())
    vote_score = np.zeros(valset.__len__())
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9)
    #### test时关闭 ############
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    #scheduler = StepLR(optimizer, step_size=1)

    total_epoch = 50
        
    for epoch in range(1, total_epoch+1):
        train(optimizer, epoch)
        
        targetlist, scorelist, predlist = val(epoch)
        
       
        vote_pred = vote_pred + predlist 
        vote_score = vote_score + scorelist 
        if epoch % votenum == 0:
            
            # major vote
            vote_pred[vote_pred <= (votenum/2)] = 0
            vote_pred[vote_pred > (votenum/2)] = 1
            vote_score = vote_score/votenum
            
            print('vote_pred', vote_pred)
            print('targetlist', targetlist)
            TP = ((vote_pred == 1) & (targetlist == 1)).sum()
            TN = ((vote_pred == 0) & (targetlist == 0)).sum()
            FN = ((vote_pred == 0) & (targetlist == 1)).sum()
            FP = ((vote_pred == 1) & (targetlist == 0)).sum()
            
            
            print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
            print('TP+FP',TP+FP)
            p = TP / (TP + FP)
            print('precision',p)
            r = TP / (TP + FN)
            print('recall',r)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN + FP + FN)
            print('F1',F1)
            print('acc',acc)
            AUC = roc_auc_score(targetlist, vote_score)
            print('AUCp', roc_auc_score(targetlist, vote_pred))
            print('AUC', AUC)
            
            
            if epoch == total_epoch:
                torch.save(model.state_dict(), "model_backup/{}.pt".format(modelname))  
            

            print('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))

            f = open('model_result/{}.txt'.format(modelname), 'a+')
            f.write('\n The epoch is {}, average recall: {:.4f}, average precision: {:.4f},average F1: {:.4f}, average accuracy: {:.4f}, average AUC: {:.4f}\n'.format(
            epoch, r, p, F1, acc, AUC))
            f.close()

    