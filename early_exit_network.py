import torch
import torch.nn as nn
import numpy as np
import sys, time, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from utils import Flatten, countFlop, set_parameter_requires_grad
from flops_counter import get_model_complexity_info
from resnet import ResNet, BasicBlock
from mobileNet import conv_bn, conv_1x1_bn, make_divisible, InvertedResidual

class ClassifierModule(nn.Module):
    def __init__(self, m, total_neurons, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(total_neurons, num_classes)

    def forward(self, x):
        res = self.m(x)
        res = res.view(res.size(0), -1)
        return self.linear(res)

class ConvBNReLU(nn.Module):
  def __init__(self, nIn, nOut, kernel=3, stride=1,
               padding=1):
    super(ConvBNReLU, self).__init__()
    self.net

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ExitBlock(nn.Module):
  """
  This class defines the Early Exit, which allows to finish the inference at the middle layers when
  the classification confidence achieves a predefined threshold
  """
  def __init__(self, nIn, n_classes, input_shape, exit_type, dataset_name="cifar"):
    super(ExitBlock, self).__init__()
    _, channel, width, height = input_shape
    """
    This creates a random input sample whose goal is to find out the input shape after each layer.
    In fact, this finds out the input shape that arrives in the early exits, to build a suitable branch.

    Arguments are

    nIn:          (int)    input channel of the data that arrives into the given branch.
    n_classes:    (int)    number of the classes
    input_shape:  (tuple)  input shape that arrives into the given branch
    exit_type:    (str)    this argument define the exit type: exit with conv layer or not, just fc layer
    dataset_name: (str)   defines tha dataset used to train and evaluate the branchyNet
     """

    self.expansion = 1
    self.layers = []

    # creates a random input sample to find out input shape in order to define the model architecture.
    x = torch.rand(1, channel, width, height)
    
    if (dataset_name == "cifar"): # branch for cifar dataset
      interChannels1 = 128
      self.conv = nn.Sequential(
        ConvBasic(channel, interChannels1, kernel=3, stride=2, padding=1),
        nn.AvgPool2d(2),
        )
      
    else: # branch for imageNet dataset
      self.conv = nn.Sequential(
        ConvBasic(channel, channel, kernel=3, stride=2, padding=1),
        nn.AvgPool2d(2),
        )
    
    #gives the opportunity to add conv layers in the branch, or only fully-connected layers
    if (exit_type == "conv"):
      self.layers.append(self.conv)
    else:
      self.layers.append(nn.AdaptiveAvgPool2d(2))
      
    feature_shape = nn.Sequential(*self.layers)(x).shape
    
    total_neurons = feature_shape[1]*feature_shape[2]*feature_shape[3] # computes the input neurons of the fc layer 
    self.classifier = nn.Linear(total_neurons , n_classes) # finally creates 
    
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)

    return self.classifier(x)


class BranchyNet(nn.Module):
  def __init__(self, model_name: str, dataset_name: str, n_classes: int, 
               pretrained: bool, imageNet: bool, feature_extraction: bool, n_branches: int,img_dim:int, 
               exit_type: str, branches_positions=None, distribution="linear"):
    super(BranchyNet, self).__init__()

    """
    This class creates a BranchyNet model, inserting branches into the middle layers. 
    model_name:      (str) model name 
    dataset_name:    (str) dataset name used to train and eval branchynet model
    n_classes:       (int) number of the class
    pretrained:      (bool)indicates whether the main model is pretrained with imageNet dataset downloaded from torchvision.models
    imageNet:        (bool)indicates whether data used is imageNet
    feature_extraction: (bool) given the main model is pretrained, it indicates whether is fine-ntuning ou feature extraction 
    n_branches:      (int) the number of early exits inserted into the middle layers
    img_dim:         (int) defines the imagen dimension received by the branchynet model
    exit_type:       (str) defines the exit type, if the early exit has conv layers or only fc layer
    branch_positions: (list) defines the position which the branches must be inserted.
    distribution:     (str) defines the distribution that branches are inserted into the middle layers
    """

    self.model_name = model_name
    self.n_classes = n_classes
    self.pretrained = pretrained
    self.imageNet = imageNet
    self.feature_extraction = feature_extraction
    self.n_branches = n_branches
    self.dataset_name = dataset_name
    self.branches_positions = branches_positions
    self.img_dim = img_dim
    n_channels = 3
    self.input_size = (n_channels, img_dim, img_dim)
    self.exit_type = exit_type
    self.distribution = distribution


    #This initializes the main model, importing from torchvision.models, and if pretrained is True
    # so this also downloads pretrained parameters on the imagenet database
    self.initialize_main_model()

  def initialize_main_model(self):
    if (self.model_name == "AlexNet"):
      self.model = models.alexnet(pretrained=self.pretrained)
      #self.insertBranches = self.insertBranchesAlexNet
      self.insertBranches = self.insertBranchesAlexNetEENet
      if not imageNet:
        self.model = set_parameter_requires_grad(self.model, feature_extraction)
        in_feature = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_feature, n_classes)

    elif (self.model_name == "ResNet18"):
        self.repetitions = [2, 2, 2, 2]
        self.total_layers = 18
        self.insertBranches = self.insertBranchesResNet
        self.model = models.resnet18(pretrained=self.pretrained)
        self.model = set_parameter_requires_grad(self.model, feature_extraction)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)

    elif (self.model_name == "ResNet50"):
        self.total_layers = 50  
        self.model = models.resnet50(pretrained=self.pretrained)
        self.model = set_parameter_requires_grad(self.model, feature_extract)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)

    elif (self.model_name == "MobileNet"):
      self.width_mult = 1.0
      self.insertBranches = self.insertBranchesMobileNet
      self.model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=self.pretrained)
      self.model = set_parameter_requires_grad(self.model, feature_extraction)
      self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.n_classes)

    else:
      print("This model is not implemented ...")
      raise NotImplementedError


  def countFlop(self, input_size):
    """
    This count Flops of the main model
    input_size: (tuple) input.shape
    if Tensor: shape = (batch, channel, width, height)
    if array: shape = (width, height, channel)
    """
    x = torch.rand(1, input_size[0], input_size[1], input_size[2])
    ops, all_data = count_ops(self.model, x, print_readable=False, verbose=True)
    flop_idx_dict = {i: 0 for i in range(len(all_data))}
    flop_layer_dict = {}

    total_flop = 0
    for i, layer in enumerate(all_data):
      total_flop += layer[1]/ops
      flop_idx_dict[i] = total_flop
      flop_layer_dict[layer[0].split("/")[-2]] = total_flop

    return flop_idx_dict, flop_layer_dict

  def loadMainModel(self):
    # returns the main model  
    return self.model


  def whereBranchesProgressiveInference(self):
    """
    This method defines where the early exits are inserted, according SPINN approach.
    """
    if (self.model_name == "AlexNet"):
      layers_idx_branches = [2, 5, 7, 9]

    else:
      flop_count_dict, _ = countFlop(self.model, self.input_size)
      branches_positions_flops = np.linspace(0.15, 0.90, num=n_branches)

      layers_idx_branches = []
      for branch_position in branches_positions_flops:
        layers_flop = min(flop_count_dict.values(), key=lambda x: abs(x-branch_position))
        layers_idx_branches.append(list(flop_count_dict.values()).index(layers_flop))
      
    if (self.branches_positions is None):
      if (len(layers_idx_branches) >= self.n_branches):
        return layers_idx_branches[:self.n_branches]
      else:
        return layers_idx_branches
    else:
      return self.branches_positions


  def whereBranchesConvolutionalLayers(self):
    for i, layer in enumerate(self.model.features):
      if (isinstance(layer, nn.MaxPool2d)):
        branches_layers.append(i)
    return np.array(branches_layers[:-1])

  def _build_cifar_early_exit(self, x, nIn):
    """
    This methods build the branch, when the dataset is cifar.
    """
    interChannels1 = 128
    conv = nn.Sequential(
      ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
      nn.AvgPool2d(2),
      )
    feature_shape = conv(x).shape
    total_neurons = feature_shape[1]*feature_shape[2]*feature_shape[3]
    return ClassifierModule(conv, total_neurons, self.n_classes)

  def _build_imageNet_early_exit(self, x, nIn):
    """
    This methods build the branch, when the dataset is imageNet.
    """    
    conv = nn.Sequential(
        ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
        nn.AvgPool2d(2)
        )
    feature_shape = conv(x).shape
    total_neurons = feature_shape[0]*feature_shape[1]*feature_shape[2]

    return ClassifierModule(conv, total_neurons, self.n_classes)


  def exitBlock(self, x, nIn):
    if (self.dataset_name == "cifar"):
      return self._build_cifar_early_exit(x, nIn)
    elif (self.dataset_name == "imagenet"):
      return self._build_imageNet_early_exit(x, nIn)
    else:
      raise NotImplementedError("This dataset has not been implemented!")

  def insertBranchesMobileNet(self, branches_positions):
    """
    Once the self.model is MobileNet V2, this methods builds a branchynet model based on MobileNet architecture.
    In other words, this methods builds a B-MobileNet.

    branches_positions:     (list)   defines branch positions
    """
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.cost = []
    self.complexity = []
    self.layers = nn.ModuleList()
    self.stage_id = 0

    channel, width, height = self.input_size
    total_flops, total_params = self.get_complexity(self.model)
    self.set_thresholds(self.distribution, total_flops)

    block = InvertedResidual
    input_channel = 32
    last_channel = 1280

    interverted_residual_setting = [
                                    # t, c, n, s
                                    [1, 16, 1, 1],
                                    [6, 24, 2, 2],
                                    [6, 32, 3, 2],
                                    [6, 64, 4, 2],
                                    [6, 96, 3, 1],
                                    [6, 160, 3, 2],
                                    [6, 320, 1, 1],
                                    ]

    assert self.img_dim % 32 == 0
    self.last_channel = make_divisible(last_channel * self.width_mult) if self.width_mult > 1.0 else last_channel
    self.layers.append(conv_bn(3, input_channel, 2))
    for t, c, n, s in interverted_residual_setting:
      output_channel = make_divisible(c * self.width_mult) if t > 1 else c
      for i in range(n):
        if (i == 0):
          self.layers.append(block(input_channel, output_channel, s, expand_ratio=t))
        else:
          self.layers.append(block(input_channel, output_channel, 1, expand_ratio=t))
        if (self.is_suitable_for_exit(i)):
          self.add_exit_block(total_flops, 0)

        input_channel = output_channel


    self.layers.append(conv_1x1_bn(input_channel, self.last_channel))
    self.fully_connected = self.model.classifier
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)

  def insertBranchesAlexNetEENet(self, branches_positions):
    """
    This method builds a B-AlexNet
    branches_positions:   (int)
    """
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.cost = []
    self.complexity = []
    self.layers = nn.ModuleList()
    self.stage_id = 0

    channel, width, height = self.input_size
    total_flops, total_params = self.get_complexity(self.model)
    self.set_thresholds(self.distribution, total_flops)

    for i, layer in enumerate(self.model.features):
      self.layers.append(layer)
      if self.is_suitable_for_exit(i):
        self.add_exit_block(total_flops, 0)

    self.layers.append(self.model.avgpool)
    self.fully_connected = self.model.classifier
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)

  def insertBranchesAlexNet(self, branches_positions):
    self.branchyNet = nn.ModuleList()
    self.classifier = nn.ModuleList()

    x = torch.rand(1, self.input_size[0], self.input_size[1], self.input_size[2])

    for i, layer in enumerate(self.model.features):
      x = layer(x)
      feature_shape = x.shape
      n_channel, feat_heigh, feat_width = feature_shape[1], feature_shape[2], feature_shape[3]
      self.branchyNet.append(layer)
      if (i in branches_positions):
        exit_block = self.exitBlock(x, n_channel)
        self.branchyNet.append(exit_block)
    
    self.classifier.append(self.model.avgpool)
    self.classifier.append(Flatten())
    for layer in self.model.classifier:
      self.classifier.append(layer)
    self.classifier = nn.Sequential(*self.classifier)

  def get_complexity(self, model):
    """
    This method receives a model or even a intermediate model and returns 
    number of flops and parameters to execute this model
    """
    flops, params = get_model_complexity_info(model, self.input_size,
                                              print_per_layer_stat=False, as_strings=False)
    return flops, params

  
  def set_thresholds(self, distribution, total_flops):
    """
    """
    gold_rate = 1.61803398875
    flop_margin = 1.0 / (self.n_branches+1)
    self.threshold = []
        
    for i in range(self.n_branches):
      if (distribution == 'pareto'):
                self.threshold.append(total_flops * (1 - (0.8**(i+1))))
      elif (distribution == 'fine'):
                self.threshold.append(total_flops * (1 - (0.95**(i+1))))
      elif (distribution == 'linear'):
                self.threshold.append(total_flops * flop_margin * (i+1))
      else:
        self.threshold.append(total_flops * (gold_rate**(i - self.num_ee)))

  
  def is_suitable_for_exit(self, i):
    """
    This method decides if a certain layer is able to receive an early exit. 
    The method set_thresholds create threshold that decides if a layer requires an earlt exit. 
    The method uses this threshold to decide that a certain layer is able to receive an early exit. 
    This occurs when the number of flops on the intermediate model is greather than the threshold. 
    """
    if (self.branches_positions is None):
      intermediateModel = nn.Sequential(*(list(self.stages)+list(self.layers)))
      flops, _ = self.get_complexity(intermediateModel)
      return self.stage_id < self.n_branches and flops >= self.threshold[self.stage_id]
    else:
      return i in self.branches_positions


  def add_exit_block(self, total_flops, nIn):
    """
    This adds an early exit. 
    Arguments are

    total_flops:       (int) the number of FLOPs to run the main model. 
    nIn:               (int)  input channel
    """
    x = torch.rand(1, self.input_size[0], self.input_size[1], self.input_size[2])
    self.stages.append(nn.Sequential(*self.layers))
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(ExitBlock(nIn, self.n_classes, feature_shape, self.exit_type))
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.exits)[-1:]))
    flops, _ = self.get_complexity(intermediate_model)
    self.cost.append(flops / total_flops)
    self.layers = nn.ModuleList()
    self.stage_id += 1

  def insertBranchesResNet(self, branches_positions):
    """
    This method builds a B-ResNet
    """
    self.inplanes = 64
    counterpart_model = ResNet(BasicBlock, self.repetitions, self.n_classes, self.input_size)
    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.cost = []
    self.complexity = []
    self.layers = nn.ModuleList()
    self.stage_id = 0

    channel, _, _ = self.input_size
    total_flops, total_params = self.get_complexity(counterpart_model)
    self.set_thresholds(self.distribution, total_flops)
    
    self.layers.append(nn.Sequential(
        nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ))
    
    
    planes = self.inplanes
    stride = 1
    for i, repetition in enumerate(self.repetitions):
      downsample = None
      if stride != 1 or self.inplanes != planes:
        downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride),
                                   nn.BatchNorm2d(planes))

      self.layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
      self.inplanes = planes

      if self.is_suitable_for_exit(i):
        self.add_exit_block(total_flops, self.inplanes)

      for j in range(repetition):
        self.layers.append(BasicBlock(self.inplanes, planes))
        if (self.is_suitable_for_exit(j)):
          self.add_exit_block(total_flops, self.inplanes)

      planes *= 2
      stride = 2
    
    planes = 512
    self.layers.append(nn.AdaptiveAvgPool2d(1))
    self.fully_connected = nn.Linear(planes, self.n_classes)
    self.stages.append(nn.Sequential(*self.layers))
    self.softmax = nn.Softmax(dim=1)

  def loadEarlyExitModel(self):
    branches_positions = self.whereBranchesProgressiveInference()

    self.insertBranches(branches_positions)
  

  def forwardTrain(self, x):
    pred_list, conf_list, class_list  = [], [], []
    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      pred = exitBlock(x)
      infered_conf, infered_class = torch.max(self.softmax(pred), 1)
      pred_list.append(pred)
      conf_list.append(infered_conf)
      class_list.append(infered_class)

    x = self.stages[-1](x)
    x = x.view(x.size(0), -1)
    pred = self.fully_connected(x)
    infered_conf, infered_class = torch.max(self.softmax(pred), 1)
    pred_list.append(pred)
    conf_list.append(infered_conf)
    class_list.append(infered_class)
    return pred_list, conf_list, class_list

  def forwardEval(self, x, p_tar):
    pred_list, conf_list, class_list = [], [], []
    for i, exitBlock in enumerate(self.exits):
      x = self.stages[i](x)
      pred = exitBlock(x)
      pred_list.append(pred)
      infered_conf, infered_class = torch.max(self.softmax(pred), 1)
      conf_list.append(infered_conf)
      class_list.append(infered_class)

      if (infered_conf.item() >= p_tar):
        return pred, infered_conf, infered_class, i 

    x = self.stages[-1](x)
    x = x.view(x.size(0), -1)
    pred = self.fully_connected(x)
    pred_list.append(pred)
    infered_conf, infered_class = torch.max(self.softmax(pred), 1)
    conf_list.append(infered_conf)
    class_list.append(infered_class)

    if (infered_conf.item() > p_tar):
      return pred, infered_conf, infered_class, len(self.exits) 
    else:
      max_conf = np.argmax(conf_list)
      return pred_list[max_conf], conf_list[max_conf], class_list[max_conf], len(self.exits)

  
  def forward(self, x, p_tar=1.0, train=True):
    if (train):
      return self.forwardTrain(x)
    else:
      return self.forwardEval(x, p_tar)




