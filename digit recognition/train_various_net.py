#import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets,transforms,models
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

### resnet 50
#model = models.resnet50(pretrained=True)

'''
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
'''

#print(model.fc)
#fc_features = model.fc.in_features
#model.fc = nn.Linear(fc_features, 10)


### alexnet
model = models.alexnet(pretrained=True)
fc_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(fc_features, 10)
print(model)

for param in model.parameters():
    param.requires_grad = False
for param in model.classifier[6].parameters():
    param.requires_grad = True
#print(model)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
#                        lr=0.01,momentum=0.9,weight_decay=0.001)


DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device( "cpu")
if (torch.cuda.is_available()):
    torch.cuda.empty_cache()  ## may not work
print(DEVICE)
model.to(DEVICE)

epochs = 1

transform_resnet50 = transforms.Compose(
    [
     transforms.Resize((7,7),Image.ANTIALIAS), #pil image only
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))] ## fixed parameter provided by pytorch
    ## reduce the model complexity
)

transform_alexnet = transforms.Compose(
    [
     transforms.Resize((224,224),Image.ANTIALIAS), #pil image only
     transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))]
)


     ## fixed parameter provided by pytorch])

train_root = "./mnist_png/training"
#img = cv2.imread(train_root)
#train_data = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#img = Image.open(train_root)
train_data_original = ImageFolder(
    root=train_root,
    transform=transform_alexnet
)

train_data, val_data = random_split(train_data_original, [54000, 6000])

#print(train_data.imgs[0][0])

#img1 = cv2.cvtColor(train_data.imgs[0][0], cv2.COLOR_GRAY2BGR)


test_root = "./mnist_png/testing"
test_data = ImageFolder(
    root = test_root,
    transform = transform_alexnet
)

train_loader = DataLoader(train_data, batch_size=60, shuffle=True)
val_loader = DataLoader(val_data, batch_size=125, shuffle=True)
test_loader = DataLoader(test_data, batch_size=125, shuffle=True)


loss_function = nn.CrossEntropyLoss()
#loss_function = nn.NLLLoss()
### train

train_accs = []
test_accs = []
x_axis_train = range(0,9)
x_axis_test = range(0,8)

for epoch in range(epochs):
  for index, (x,label) in enumerate(train_loader):
    x, label = x.to(DEVICE),label.to(DEVICE)
    #x = x.view(-1, 28 * 28) ## -1 equals the batch size ## uncommented only on FullConnectedNetwork
    out = model(x)
    
    #print('train out', out.shape)
    
    ### if nll loss function used
    #m = nn.LogSoftmax(dim=1)
    #loss = loss_function(m(out),label)
    
    count = correct = 0
    
   
    ### for crossentropy
    loss = loss_function(out,label)
    _ , predict = torch.max(out,1)
    count += x.shape[0]
    correct += (predict == label).sum()
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (index+1) % 100 == 0 or (index+1) == len(train_loader):
        acc = correct*1.0/ count
        train_accs.append(acc)
        print("Train epoch", epoch, 'batch index', index+1, 'loss', float(loss),'train acc', acc)


  ### test
  best_test_acc = 0.0    
  count = correct = 0
  for index, (x,label) in enumerate(test_loader):
    x, label = x.to(DEVICE),label.to(DEVICE)
    #x = x.view(-1, 28*28) ## -1 equals the batch size ## uncommented only on FullConnectedNetwork
    out = model(x)  ## [batch_size, 10]
    
    #print('test out', out.shape)

    ### if nll loss function used
    #m = nn.LogSoftmax(dim=1)
    #loss = loss_function(m(out),label)
    
    
    ### for crossentropy
    loss = loss_function(out,label)
    
    _ , predict = torch.max(out,1)
    count += x.shape[0]
    correct += (predict == label).sum()

     
    if (correct*1.0/ count).item() > best_test_acc:
        best_test_acc = (correct*1.0/ count).item()


    if (index+1) % 10 == 0 or (index+1) == len(test_loader):
        acc = correct*1.0/ count
        test_accs.append(acc)
        print("Train epoch", epoch, 'batch index', index+1, 'loss', float(loss), 'test acc', acc)

plt.subplot(2, 1, 1)
plt.plot(x_axis_train, train_accs, 'o-')
plt.subplot(2, 1, 2)
plt.plot(x_axis_test, test_accs, '.-')
plt.show()


print('best test acc', best_test_acc)
torch.save(model, './model_saved_alexnet_5.pth')

'''
model_saved = torch.load('./model_saved_alexnet_3.pth')
#model_saved.classifier[6].out_features = 10
model_saved.eval()
#print(model_saved)

##### ./model_saved_alexnet_3.pth'  max acc:99.2 
##### ./model_saved_alexnet_4.pth'  max acc:99.2 


test_picture_name_pre = './test_pics/test_'
test_picture_name_post = '.png'
for i in range(10):
    img = Image.open(test_picture_name_pre + str(i) + test_picture_name_post)
    img = img.convert("RGB")

    #print(img)
    #img.show()
    #img = img.convert('1') 
    #print(img.shape)

    img = transform_alexnet(img)
    #print(img.shape)

    img = img.unsqueeze(0)
    img = img.to(DEVICE)
    model_saved = model_saved.to(DEVICE)
    score = model_saved(img)
    #print(score.shape)
    probability = torch.nn.functional.softmax(score,dim=1)
    max_value,index = torch.max(probability,1)

    all_results = probability.data.cpu().numpy()
    #print(all_results)
    print(index)
'''



#utils.run_experiment(model,optimizer,train_loader,val_loader,test_loader,epochs,0,0,False)