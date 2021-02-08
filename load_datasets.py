import torch
import torchvision.transforms as transforms
from torchvision import datasets

class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test

    self.transformation_list = [transforms.Resize(256),
                           transforms.CenterCrop(self.input_dim),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]


  def cifar_10(self):
    # Load Cifar-10 dataset 
    root = "cifar_10"

    trainset = datasets.CIFAR10(root=root, train=True, download=True,
                                transform=transforms.Compose(self.transformation_list))
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR10(root=root, train=False, download=True,
                               transform=transforms.Compose(self.transformation_list))
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, 
    	num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def cifar_100(self):
    # Load Cifar-100 dataset
    root = "cifar_100"

    trainset = datasets.CIFAR100(root=root, train=True, download=True,
                                transform=transforms.Compose(self.transformation_list))
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR100(root=root, train=False, download=True,
                               transform=transforms.Compose(self.transformation_list))
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, 
    	num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def imageNet(self):
    # Load ImageNet Dataset
    root = "ImageNet"

    trainset = datasets.CIFAR100(root=root, train=True, download=True,
                                transform=transforms.Compose(self.transformation_list))
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR100(root=root, train=False, download=True,
                               transform=transforms.Compose(self.transformation_list))
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, num_workers=2, shuffle=False)
    
    return trainLoader, testLoader


