import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import torchvision.datasets.voc as voc
from utils import encode_labels, MapDataset
import numpy as np, cv2
from PIL import Image


class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """
    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        
        super().__init__(
             root, 
             year=year, 
             image_set=image_set, 
             download=download, 
             transform=transform, 
             target_transform=target_transform)
    
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)
        
    
    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)



class LoadDataset():
  def __init__(self, input_dim, batch_size_train, batch_size_test, normalization=True):
    self.input_dim = input_dim
    self.batch_size_train = batch_size_train
    self.batch_size_test = batch_size_test
    self.savePath_idx_dataset = None

    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformation_train_list = [transforms.Resize((300, 300)),
                                      transforms.RandomChoice([
                                          transforms.ColorJitter(brightness=(0.80, 1.20)),
                                          transforms.RandomGrayscale(p = 0.25)
                                          ]),
                                      transforms.RandomHorizontalFlip(p = 0.25),
                                      transforms.RandomRotation(25),
                                      transforms.ToTensor()]

    transformation_valid_list = [transforms.Resize(330), 
                                          transforms.CenterCrop(300), 
                                          transforms.ToTensor()]
    
    if (normalization):
      transformation_train_list.append(transforms.Normalize(mean = mean, std = std))
      transformation_valid_list.append(transforms.Normalize(mean = mean, std = std))


    self.transformations_train = transforms.Compose(transformation_train_list)
        
    self.transformations_valid = transforms.Compose(transformation_valid_list)


  def cifar_10(self):
    # Load Cifar-10 dataset 
    self.root = "cifar_10"

    trainset = datasets.CIFAR10(root=root, train=True, download=True,
                                transform=self.transformation_list)
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR10(root=root, train=False, download=True,
                               transform=self.transformation_list)
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def cifar_100(self):
    # Load Cifar-100 dataset
    self.root = "cifar_100"

    trainset = datasets.CIFAR100(root=root, train=True, download=True,
                                transform=self.transformation_list)
    
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size_train, 
                                              num_workers=2, shuffle=True, drop_last=True)
    
    testset = datasets.CIFAR100(root=root, train=False, download=True,
                               transform=self.transformation_list)
    
    testLoader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size_test, num_workers=2, shuffle=False)
    
    return trainLoader, testLoader

  def imageNet(self, root_path):
    # Load ImageNet Dataset

    test_dataset = datasets.ImageFolder(root = root_path, transform = self.transformation_list)
    _, val_dataset = random_split(test_dataset, (0, 50000))

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=self.batch_size_test)
    return None, val_loader

  def caltech(self, root_path, split_train=0.8):

    dataset = datasets.ImageFolder(root_path)

    train_dataset = MapDataset(dataset, self.transformations_train)
    val_dataset = MapDataset(dataset, self.transformations_valid)
    #test_dataset = MapDataset(dataset, self.transformations_valid)

    #nr_samples = len(dataset)
    #indices = list(range(nr_samples))
    #split_train_test = int(np.floor(split_train * nr_samples))

    #np.random.shuffle(indices)
    #train_idx, test_idx = indices[:split], indices[split:]

    #nr_samples_train = len(train_idx)

    #idx_train = list(range(nr_samples_train))

    #np.random.shuffle(indices)

    #split_train_val = int(np.floor(split_train * nr_samples_train))

    #train_idx, valid_idx = idx_train[:split_train_val], idx_train[split_train_val:]


    if (self.savePath_idx_dataset is not None):
      data = np.load(self.savePath_idx_dataset, allow_pickle=True)
      train_idx, valid_idx = data[0], data[1]
      indices = list(range(len(valid_idx)))
      split = int(np.floor(0.5 * len(valid_idx)))
      valid_idx, test_idx = valid_idx[:split], valid_idx[split:]
      print(valid_idx)

    else:
      nr_samples = len(dataset)
      indices = list(range(nr_samples))
      split = int(np.floor(split_train * nr_samples))
      np.random.shuffle(indices)
      rain_idx, test_idx = indices[:split], indices[split:]


    train_data = torch.utils.data.Subset(train_dataset, indices=train_idx)
    val_data = torch.utils.data.Subset(val_dataset, indices=valid_idx)
    test_data = torch.utils.data.Subset(val_dataset, indices=test_idx)

    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size_train, 
                                              num_workers=4)
    valLoader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size_test, 
                                              num_workers=4)
    testLoader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size_test, 
                                              num_workers=4)


    return trainLoader, valLoader, testLoader

  def set_idx_dataset(self, save_idx_path):
    self.savePath_idx_dataset = save_idx_path


  def pascal(self, root_path):

    train_dataset = PascalVOC_Dataset("./", year='2011', image_set='train', download=True, 
      transform=self.transformations_train, 
      target_transform=encode_labels)


    val_dataset = PascalVOC_Dataset("./", year='2011', image_set='val', download=True, transform=
      self.transformations_valid, 
      target_transform=encode_labels)

    train_loader = DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_dataset, batch_size=self.batch_size_test, num_workers=4)

    return train_loader, valid_loader

  def getDataset(self, root_path, dataset_name):
    self.dataset_name = dataset_name
    def func_not_found():
      print("No dataset %s is found"%(self.dataset_name))

    func_name = getattr(self, self.dataset_name, func_not_found)
    train_loader, val_loader= func_name(root_path)
    return train_loader, val_loader


class DataTransformation():
  def __init__(self, distortion_type, distortion_list):
    self.distortion_type = distortion_type
    self.distortion_list = distortion_list

  def gaussian_blur(self, img):
    image = np.array(img)
    blur_std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
    blur = cv2.GaussianBlur(image, (4*blur_std+1, 4*blur_std+1), blur_std, None, blur_std, cv2.BORDER_CONSTANT)
    return Image.fromarray(blur) 
    

  def gaussian_noise(self):
    image = np.array(img)
    noise_std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
    noise_img = image + np.random.normal(0, noise_std, (image.shape[0], image.shape[1], image.shape[2]))
    return Image.fromarray(noise_img) 

  def applyDistortion(self):
    def func_not_found():
      print("No dataset %s is found"%(self.distortion_type))
    
    func_name = getattr(self, self.distortion_type, func_not_found)
    return func_name
