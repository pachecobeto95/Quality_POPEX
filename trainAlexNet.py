import os
import argparse
from early_exit_network import BranchyNet
from load_datasets import LoadDataset
import pandas as pd
from utils import save_model, save_history, trainBranches, evalBranches

"""
By default, this script follows the hyperparameters configurarion from original paper. 
"""
parser = argparse.ArgumentParser(description='PyTorch Training B-AlexNet')

parser.add_argument('--batch_size_train', type=int, default=128, metavar='N',
	help='input batch size for training (default: 128)')
parser.add_argument('--batch_size_train', type=int, default=1, metavar='N',
	help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
	help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='N',
	help='learning rate (default: 0.01)')
parser.add_argument('--adaptive_lr',  action='store_true', default=True,
	help='adjust the learning rate (default: True)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
	help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='N',
	help='weight decay for optimizers (default: 0.0005)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
	help='random seed (default: 42)')
parser.add_argument('--save_model', action='store_true', default=True, help='do not save the current model')
parser.add_argument('--load_model',   type=str,   default=None, metavar='S',
	help='the path for loading and evaluating model')
parser.add_argument('--n_branches', type=int, default=2, metavar='N',
	help='the number of early exits (default: 2)')
parser.add_argument('--dataset', type=str, default='cifar10',
	choices=['cifar10', "cifar100",'imagenet','tiny-imagenet'],
	help='dataset to be evaluated (default: cifar10)')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD','Adam'],
	help='optimizer (default: SGD)')
parser.add_argument('--distribution', type=str, default='fine',
	choices=['gold_ratio', 'pareto', 'fine', 'linear'],
	help='distribution method of exit blocks (default: linear)')

parser.add_argument('--branches_positions', type=list, default=[2, 5, 7],
	help='Positions of early-exit branches (default: [2, 5, 7])')

parser.add_argument('--exit_type', type=str, default='', choices=['conv', ''], help='Exit block type.')
parser.add_argument('--device',       help=argparse.SUPPRESS)
parser.add_argument('--results_dir',  help=argparse.SUPPRESS)
parser.add_argument('--n_classes',  help=argparse.SUPPRESS, default=10)
parser.add_argument('--input_shape',  help=argparse.SUPPRESS, default=(3, 32, 32))

args = parser.parse_args()

dataset = LoadDataset(img_dim, args.batch_size_train, args.batch_size_test)

if (args.dataset == "cifar10"):
	args.n_classes = 10
	trainLoader, testLoader = dataset.cifar_10()

elif(args.dataset == "cifar100"):
	args.n_classes = 100
	trainLoader, testLoader = dataset.cifar_100()

elif(args.dataset == "imagenet"):
	args.n_classes = 1000

img_dim = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (args.exit_type == ''):
	exit_type = None	

model_name = "AlexNet"
if not (os.path.exists(args.results_dir)):
	os.makedirs(args.results_dir)

history_loss_path = args.results_dir + '/history_loss.csv'
save_model_path = args.results_dir + "b_alexNet_model.pt"
b_alexnet = BranchyNet(model_name, args.dataset, args.n_classes, pretrained, imageNet, feature_extraction, args.n_branches,
	img_dim, exit_type=exit_type, branches_positions=args.branches_positions)

if (args.optimizer == "SGD"):
	optimizer = optim.SGD(b_alexnet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif(args.optimizer == "Adam")
	optimizer = optim.Adam(b_alexnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

best = {}
save_dict = {}
best_epoch = 0
columns = ["epoch", "train_loss", "val_loss", "val_acc"]
for i in range(n_branches+1):
  columns.append("train_acc_%s"%(i+1))
  columns.append("val_acc_%s"%(i+1))

df = pd.DataFrame(columns=columns)
for epoch in range(n_epochs):
  print("Epoch: %s"%(epoch))
  result = {"epoch": epoch}
  result.update(trainBranches(model, trainLoader, optimizer, epoch, device, n_branches, weight_loss))
  result.update(evalBranches(model, testLoader, epoch, n_branches, device, ptar=0.5))

  if (args.adaptive_lr):
  	scheduler.step(result['val_loss'])
  
  if not best or result["val_loss"]<best["val_loss"]:
    best = result
    save_model(model, best, n_branches, optimizer, save_model_path)
  save_history(df, result, history_loss_path)










