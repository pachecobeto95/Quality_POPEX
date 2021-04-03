import torch
import torch.nn as nn
import numpy as np
import sys, time, math, os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets, transforms
from scipy import stats
from load_dataset import LoadDataset
from mobilenet import B_MobileNet
from PIL import Image


def measuringProcessingTime(x, model):
  processing_time_dict = {}
  count_layer = 0
  processing_time = 0
  for i, exitBlock in enumerate(model.exits):
    start = time.time()
    x = model.stages[i](x)
    output_branch = exitBlock(x)
    end = time.time()
    processing_time += end - start
    output_feature_size = x.detach().cpu().numpy().nbytes
    output_feature_size = (8*output_feature_size)/10**(6)
    if (i > 0):
      processing_time_dict.update({"processing_time_exit_%s"%(i+1): processing_time, "output_size_exit_%s"%(i+1): output_feature_size})
  return processing_time_dict

def saveProcessingTimeB_Mobilenet(x, model, savePath, n_rounds=50):
  df = pd.DataFrame(columns=[])
  for n in range(1, n_rounds+1):
    print("Round: %s"%(n))
    time.sleep(2)
    processing_time_dict = {}
    processing_time_dict.update(measuringProcessingTime(x, model))
    df = df.append(pd.Series(processing_time_dict), ignore_index=True)
    df.to_csv(savePath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes = 258
n_branches = 3
img_dim = 300
pretrained = True
exit_type = None
savePath = "./processing_time_notebook_cpu.csv"

branchynet = B_MobileNet(n_classes, pretrained, n_branches, img_dim, exit_type, device).to(device)
x = torch.rand(1, 3, 300, 300).to(device)
saveProcessingTimeB_Mobilenet(x, branchynet, savePath)