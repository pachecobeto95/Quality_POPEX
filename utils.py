from pthflops import count_ops
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

def save_model(model, best,n_branches, optimizer, savePath):
  save_dict = {
      'epoch': best["epoch"],
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      "val_acc": best['val_acc'],
      }
  for i in range(n_branches+1):
    save_dict["train_acc_%s"%(i+1)] = best["train_acc_%s"%(i+1)]
    save_dict["val_acc_%s"%(i+1)] = best["val_acc_%s"%(i+1)]    
  
  torch.save(save_dict, savePath)

def save_history(df, result, savePath):
  df = df.append(result, ignore_index=True)
  df.to_csv(savePath)

def set_parameter_requires_grad(model, feature_extraction):
  if (feature_extraction):
    for param in model.parameters():
      param.requires_grad  = False

  return model

def countFlop(model, input_size):
  x = torch.rand(1, input_size[0], input_size[1], input_size[2])
  ops, all_data = count_ops(model, x, print_readable=False, verbose=True)
  flop_idx_dict = {i: 0 for i in range(len(all_data))}
  flop_layer_dict = {}

  total_flop = 0
  for i, layer in enumerate(all_data):
    total_flop += layer[1]/ops
    flop_idx_dict[i] = total_flop
    flop_layer_dict[layer[0].split("/")[-2]] = total_flop

  return flop_idx_dict, flop_layer_dict

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()
  def forward(self, x):
    return x.view(x.size(0), -1)


def trainBranches(model, train_loader, optimizer, epoch, device, n_branches, weight_loss):
  """
  trains the branchynet model.

  * model:         branchynet model
  * train_loader:  training dataset, containing input images and its labels
  * optimizer:     optimizer operator to adjust the model parameters. 
  """
  losses = []
  model.to(device)
  model.train()
  acc_train = [0]*(n_branches + 1)
  batch = {'train_acc_%s'%(i): [] for i in range(1, n_branches+1+1)}

  loss_fn = nn.CrossEntropyLoss()
  for i, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device, dtype=torch.int64)
    optimizer.zero_grad()

    pred_list, conf_list, class_list = model(data)
    loss = 0 
    for i, (pred, conf, inf_label) in enumerate(zip(pred_list, conf_list, class_list)):
      loss += weight_loss[i]*loss_fn(pred, target)
      acc_train[i] = inf_label.eq(target.view_as(inf_label)).sum().item()/data.size(0)
      batch["train_acc_%s"%(i+1)].append(acc_train[i])
      
    losses.append(float(loss.item()))
    loss.backward()
    optimizer.step()

  result = {'train_loss': round(np.mean(losses), 4)}

  print('Train avg loss: {:.4f}'.format(result['train_loss']))

  info = ""
  for i, (key, value) in enumerate(batch.items()):
    result[key] = round(np.mean(batch["train_acc_%s"%(i+1)]), 4)
    info += "Acc Branch %s: %s "%(i+1, round(np.mean(batch["train_acc_%s"%(i+1)]), 4))
  print(info)
  return result


def evalBranches(model, val_loader, epoch, n_branches, device, ptar):
  """
  Validates the model.

  Arguments are
  * model                  defines the model evaluated
  * eval_loader            evaluation dataset, containing input images and its labels   
  * epoch             (int) number of the current epoch.
  This validates the model and prints the results of each epochs.
  Finally, it returns average accuracy, loss.
  """
  loss_fn = nn.CrossEntropyLoss()
  exit_points = np.zeros(n_branches + 1)
  correct_branches = np.zeros(n_branches + 1)
  correct = 0
  model.to(device)
  model.eval()
  val_loss_list = []
  result = {}
  with torch.no_grad():
    for i, (data, target) in enumerate(val_loader):
      data, target = data.to(device), target.to(device, dtype=torch.int64)
      pred, infered_conf, infered_class, exit_idx = model(data, p_tar=ptar, train=False)
      exit_points[exit_idx] += 1
      total_samples = data.size(0)
      #correct += infered_class.eq(target.view_as(infered_class)).sum().item()
      correct_branches[exit_idx] += infered_class.eq(target.view_as(infered_class)).sum().item()
      loss = loss_fn(pred, target)
      val_loss_list.append(loss.item())

  acc_branches = correct_branches/exit_points
  acc = sum(correct_branches)/sum(exit_points)
  result['val_loss'] = round(np.mean(val_loss_list), 4)
  result['val_acc'] = acc
  for i in range(n_branches+1):
     result["val_acc_%s"%(i+1)] = acc_branches[i]
  
  return result















