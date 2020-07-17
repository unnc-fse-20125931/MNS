# MNS  
## Running environment:  
pytorch : 1.5.1 (https://pytorch.org/get-started/locally/)  
torchvision : 0.6.1  
cuda : 10.1 (https://developer.nvidia.com/cuda-downloads)  
cudnn : 7.6.4 (https://developer.nvidia.com/rdp/cudnn-download)  

## 1. data recognition  
(1) mnist_png source : https://github.com/appleyuchi/MNIST_PNG  
(2) 3network_mnist.py : 1. FullConnectedNetwork  2. LeNet  3. GRUNet  
(3) mnist_practice.py: sample network to do the mnist classification  
(4) model_saved.pth : saved alexnet model, accuracy > 99%  
(5) train_various_py : run this file directly to train alexnet/resnet50 and save the network, also could use the pictures in "test_pics" to train the saved model.  
(6) utils.py : network training frame  
(7) model_saved_alexnet_3.pth : saved alexnet model, max acc: 99.2%    
(8) model_saved_alexnet_4.pth : saved alexnet model, max acc: 99.2%    

## 2. resoning
