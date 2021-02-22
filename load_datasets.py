import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler

class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test

    self.transformation_list = transforms.Compose([transforms.Resize(input_dim),
                                                   transforms.CenterCrop(input_dim),
                                                   transforms.ToTensor()])


  def cifar_10(self):
    # Load Cifar-10 dataset 
    root = "cifar_10"

    trainset = datasets.CIFAR10(root=root, train=True, download=True,
                                transform=transforms.Compose(self.transformation_list))
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR10(root=root, train=False, download=True,
                               transform=transforms.Compose(self.transformation_list))
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=2, shuffle=False)
    
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
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def imageNet(self, root_path):
    # Load ImageNet Dataset

    test_dataset = datasets.ImageFolder(root = root_path, transform = self.transformation_list)
    _, val_dataset = random_split(test_dataset, (0, 50000))

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=self.batch_size_test)
    return None, val_loader

  def caltech(self, root_path, split_train=0.8):
    dataset = datasets.ImageFolder(root = root_path, transform = self.transformation_list)
    train_size = int(split_train*len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, (train_size, test_size))
    train_dataset, val_dataset =  random_split(train_dataset, (int(split_train*len(train_dataset)), len(train_dataset) - int(split_train*len(train_dataset))))   
    
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.batch_size_train)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=self.batch_size_test)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=self.batch_size_test)
    return train_loader, val_loader, test_loader 