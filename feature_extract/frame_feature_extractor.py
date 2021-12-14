import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

import os
import argparse
import numpy as np
import random
from tqdm import tqdm

from data_util import phase2label_dicts, PureTimestampDataset, FullDataset
from model import inception_v3, SemiNetwork
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 20000604
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action')
parser.add_argument('--dataset', choices=['cholec80','m2cai16'])
parser.add_argument('--target', type=str, default='train_set')
parser.add_argument('--num_epochs', type=int)
args = parser.parse_args()

epochs = args.num_epochs
log_interval = 5


def train_only(model, save_dir, train_loader, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    model = nn.DataParallel(model)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0

        for (imgs, labels, img_names) in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs) # of shape 64 x 7
            loss = criterion(res, labels.long())
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        if epoch % log_interval == 0:
            test_acc, test_loss = test(model, test_loader)
            f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
            f.flush()
    print('Training done!')
    f.close()


def train_semi(model, save_dir, sup_train_loader, unsup_train_loader, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss()
    niters_per_epoch = 30
    learning_rate = 1e-4
    optimizer_l = torch.optim.Adam(model.branch1.parameters(), learning_rate, weight_decay=1e-5)
    optimizer_r = torch.optim.Adam(model.branch2.parameters(), learning_rate, weight_decay=1e-5)
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=2, gamma=0.5)
    scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=2, gamma=0.5)
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0
        sup_trainloader = iter(sup_train_loader)
        unsup_trainloader = iter(unsup_train_loader)

        for idx in tqdm(range(niters_per_epoch)):
            sup_imgs, sup_labels, sup_img_names = next(sup_trainloader)
            unsup_imgs, _, unsup_img_names = next(unsup_trainloader)
            sup_imgs, sup_labels, unsup_imgs = sup_imgs.to(device), sup_labels.to(device), unsup_imgs.to(device)

            _, pred_sup_l = model(sup_imgs, step=1)
            _, pred_sup_r = model(sup_imgs, step=2)
            _, pred_unsup_l = model(unsup_imgs, step=1)
            _, pred_unsup_r = model(unsup_imgs, step=2)

            pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
            _, max_l = torch.max(pred_l, dim=1)
            _, max_r = torch.max(pred_r, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()

            cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            sup_loss = criterion(pred_sup_l, sup_labels) + criterion(pred_sup_r, sup_labels)
            loss = cps_loss + sup_loss

            loss_item += loss.item()
            _, prediction = torch.max(pred_sup_l.data, 1)
            correct += ((prediction == sup_labels).sum()).item()
            total += len(prediction)

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

        scheduler_l.step()
        scheduler_r.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        if epoch > 5:
            test_acc, test_loss = test(model, test_loader)
            f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
            f.flush()
    print('Training done!')
    f.close()


def test(model, test_loader):
    print('Testing...')
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    loss_item = 0
    criterion = nn.CrossEntropyLoss()
    all_pred = []
    all_label = []
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs)  # of shape 64 x 7
            loss = criterion(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)
            for i in range(len(prediction)):
                all_pred.append(int(prediction.data.cpu()[i]))
                all_label.append(int(labels[i]))
    accuracy = correct / total
    precision = metrics.precision_score(all_label, all_pred, average='macro')
    recall = metrics.recall_score(all_label, all_pred, average='macro')
    print('Test: Acc {:.4f}, Loss {:.4f}'.format(accuracy, loss_item / total))
    print('Test: Precision: {:.4f}'.format(precision))
    print('Test: Recall: {:.4f}'.format(recall))
    return accuracy, loss_item / total


def extract(model, loader, save_path, record_err= False):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    err_dict = {}
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(loader):
            assert len(img_names) == 1 # batch_size = 1
            video, img_in_video = img_names[0].split('/')[-2], img_names[0].split('/')[-1] # video63 5730.jpg
            video_folder = os.path.join(save_path, video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            feature_save_path = os.path.join(video_folder, img_in_video.split('.')[0] + '.npy')

            if os.path.exists(feature_save_path):
                continue
            imgs, labels = imgs.to(device), labels.to(device)
            features, res = model(imgs)

            _,  prediction = torch.max(res.data, 1)
            if record_err and (prediction == labels).sum().item() == 0:
                # hard frames
                if video not in err_dict.keys():
                    err_dict[video] = []
                else:
                    err_dict[video].append(int(img_in_video.split('.')[0]))

            features = features.to('cpu').numpy() # of shape 1 x 2048

            np.save(feature_save_path, features)

    return err_dict


def imgf2videof(source_folder, target_folder):
    '''
        Merge the extracted img feature to video feature.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video in os.listdir(source_folder):
        video_feature_save_path = os.path.join(target_folder, video + '.npy')
        video_abs_path = os.path.join(source_folder, video)
        nums_of_imgs = len(os.listdir(video_abs_path))
        video_feature = []
        for i in range(nums_of_imgs):
            img_abs_path = os.path.join(video_abs_path, '{:04d}.npy'.format(i+1))
            video_feature.append(np.load(img_abs_path))

        video_feature = np.concatenate(video_feature, axis=0)

        np.save(video_feature_save_path, video_feature)
        print('{} done!'.format(video))




if __name__ == '__main__':

    frames_path = '../dataset/{}/frames'.format(args.dataset)
    annotations_path = '../dataset/{}/frames_annotations'.format(args.dataset)
    timestamp_path = '../dataset/{}/timestamp.npy'.format(args.dataset)
    if args.action == 'train_only':
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))

        timestamp_traindataset = PureTimestampDataset(args.dataset, frames_path, annotations_path, timestamp_path)
        timestamp_train_dataloader = DataLoader(timestamp_traindataset, batch_size=8, shuffle=True, drop_last=False)

        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_only( inception, 'models/{}/only'.format(args.dataset), timestamp_train_dataloader, full_test_dataloader)

    if args.action == 'train_full':
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))

        full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
        full_train_dataloader = DataLoader(full_traindataset, batch_size=64, shuffle=True, drop_last=False)

        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_only( inception, 'models/{}/full'.format(args.dataset), full_train_dataloader, full_test_dataloader)


    if args.action == 'train_semi':
        net = SemiNetwork('inception_v3', len(phase2label_dicts[args.dataset]))
        unsup_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True, unsupervised=True, timestamp=timestamp_path)
        unsup_train_dataloader = DataLoader(unsup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        sup_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path)
        sup_train_dataloader = DataLoader(sup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        print('unsup dataset: {}\n sup dataset: {}\n'.format(len(unsup_traindataset), len(sup_traindataset)))
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_semi(net, 'models/{}/semi'.format(args.dataset), sup_train_dataloader, unsup_train_dataloader, full_test_dataloader)

    if args.action == 'extract_raw': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        #model_path = 'models/{}/only/10.model'.format(args.dataset)
        #inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@raw/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@raw/'.format(args.dataset), '{}/train_dataset/video_feature@raw/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@raw/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@raw/'.format(args.dataset), '{}/test_dataset/video_feature@raw/'.format(args.dataset))
    
    if args.action == 'extract_only': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        inception = nn.DataParallel(inception.to(device))
        model_path = 'models/{}/only/10.model'.format(args.dataset)
        inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@only/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@only/'.format(args.dataset), '{}/train_dataset/video_feature@only/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@only/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@only/'.format(args.dataset), '{}/test_dataset/video_feature@only/'.format(args.dataset))
    
    if args.action == 'extract_semi': # extract inception feature
        net = SemiNetwork('inception_v3', len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/semi/7.model'.format(args.dataset)
        net.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(net, full_train_dataloader, '{}/train_dataset/frame_feature@semi/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@semi/'.format(args.dataset), '{}/train_dataset/video_feature@semi/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(net, full_test_dataloader, '{}/test_dataset/frame_feature@semi/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@semi/'.format(args.dataset), '{}/test_dataset/video_feature@semi/'.format(args.dataset))

    if args.action == 'extract_full': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        inception = nn.DataParallel(inception.to(device))
        model_path = 'models/{}/full/5.model'.format(args.dataset)
        state_dict = torch.load(model_path)
        #new_state_dict = {}
        #for k, v in state_dict.items():
        #    new_state_dict[k[7:]] = v
        inception.load_state_dict(state_dict)

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@full/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@full/'.format(args.dataset), '{}/train_dataset/video_feature@full/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@full/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@full/'.format(args.dataset), '{}/test_dataset/video_feature@full/'.format(args.dataset))
    
