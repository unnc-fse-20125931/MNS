 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
 

BATCH_SIZE = 100
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
 
pipeline  = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,) ,(0.3081,))    
])
 

from torch.utils.data import DataLoader
 
train_set = datasets.MNIST("data", train=True, download=True,transform = pipeline)
test_set = datasets.MNIST("data",train=False, download=True, transform = pipeline)
print(train_set)
print(test_set)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=True)
print(train_loader)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,shuffle=True)

## show image
'''
with open("./data/MNIST/raw/train-images-idx3-ubyte", "rb") as f:
    file = f.read()
    
image1 = [int(str(item).encode('ascii'),16) for item in file[16: 16+784]]
print(image1)

import cv2
import numpy as np

image1_np = np.array(image1, dtype=np.uint8).reshape(28,28,1)
print(image1_np.shape)

cv2.imwrite("digit1.jpg",image1_np)
'''
##############


class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,5)  # 1. gray channel  10.output channel 5.kernel  
        self.conv2 = nn.Conv2d(10,20,3)  # 10.input channel  20.output channel 3.kernel  
        self.fc1 = nn.Linear(20*10*10,500) # 20*10*10 input channel,  500 output channel
        self.fc2 = nn.Linear(500,10) # 500 input channel  10 output channel = results numbers
        
    def forward(self,x):
        input_size = x.size(0)  # batch_size
        x = self.conv1(x) # input  batch*1*28*28 output batch*10*24*24   (28-5+1)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)  # input batch*10*24*24 output batch*10*12*12 compress image reduce part of the image
        
        x = self.conv2(x) # input batch*10*12*12  output batch*20*10*10 (12-3+1)
        x = F.relu(x)
        
        
        x = x.view(input_size, -1)  # flatten  ,  -1 compute the dimension automatically  :  10*10*20
        
        x = self.fc1(x) # input batch*2000  output batch*500
        x = F.relu(x)
        
        x = self.fc2(x)  # input batch*500 output batch*10
        
        output = F.log_softmax(x, dim = 1) # compute the probability of each number
        
        return output
        
model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())  ## update the weights


def train_model(model, device, train_loader, optimizer, epoch ):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data, target = data.to(device),  target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.cross_entropy(output, target)
        loss = F.nll_loss(output, target)
        ##pred = output.max(1, keepdim = True)  # pred = output.argmax(dim=1)
        loss.backward()
        optimizer.step()
        
        if batch_index % 100 == 99:
            print("batch_index: {} Train Epoch : {} \t Loss : {:.6f}".format(batch_index,epoch,loss.item()))


def test_model(model,device,test_loader):
    model.eval()    
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.cross_entropy(output, target).item()
            test_loss += F.nll_loss(output, target).item() 
            pred = output.max(1,keepdim=True)[1]           
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss, 100.0 *correct / len(test_loader.dataset)))
        
        
for epoch in range(1,EPOCHS + 1):
    
    
    ###test
    #model = train_various_net.model
    #train_loader = train_various_net.train_loader
    #test_loader = train_various_net.test_loader
    #optimizer = train_various_net.optimizer
    #epoch = 20
    
    train_model(model,DEVICE,train_loader, optimizer, epoch)
    test_model(model,DEVICE,test_loader)
        
        
        