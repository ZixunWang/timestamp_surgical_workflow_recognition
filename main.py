from io import TextIOBase
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
from eval import evaluate
import numpy as np

phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# comment out seed to train the model
seed = 20000604
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', help='two options: train or predict')
parser.add_argument('--dataset', default="cholec80", help='two dataset: m2cai, cholec80', choices=['cholec80', 'm2cai'])
parser.add_argument('--epoch', default=50, type=int, help='Number of training epoch')
parser.add_argument('--start', default=30, type=int, help='which epoch start to use pseudo labels')
parser.add_argument('--pseudo', help='how to generate pseudo label', choices=['paper', 'union', 'naive', 'full'])
args = parser.parse_args()

num_stages = 4
num_layers = 10
num_f_maps = 64
features_dim = 2048
batch_size = 8
lr = 0.0005
num_epochs = args.epoch
start_epoch = args.start
pseudo = args.pseudo
sample_rate = 1


train_features = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'train_dataset', 'video_feature@2020')
test_features = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'test_dataset', 'video_feature@2020')
train_gt_path = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'train_dataset', 'annotation_folder')
test_gt_path = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'test_dataset', 'annotation_folder')

# Use time data to distinguish output folders in different training
# time_data = '2021-09-26_17-44-18' # turn on this line in evaluation
# time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
model_dir = os.path.join("./models/", args.dataset, 'total-{}_start-{}_pseudo-{}'.format(num_epochs, start_epoch, args.pseudo))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("{} dataset {} for single stamp supervision".format(args.action, args.dataset))
print('batch size is {}, number of stages is {}, sample rate is {}\n'.format(batch_size, num_stages, sample_rate))


phase2label = phase2label_dicts[args.dataset]

num_classes = len(phase2label)
writer = SummaryWriter()
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes, start_epoch, pseudo)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, phase2label, train_gt_path, train_features, sample_rate)
    # Train the model
    trainer.train(model_dir, batch_gen, writer, num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, device=device)
    trainer.predict(model_dir, test_features, test_gt_path, num_epochs, phase2label, device, sample_rate)
elif args.action == 'test':
    trainer.predict(model_dir, test_features, test_gt_path, num_epochs, phase2label, device, sample_rate)
else:
    raise NotImplementedError('Invalid action')