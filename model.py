import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import copy
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt


class ParallelModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(ParallelModel, self).__init__()
        self.branch_1 = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes)
        self.branch_2 = MultiStageModel(num_stages, num_layers, num_f_maps, dim, num_classes)

        #for name, m in self.branch_1.named_modules():
        #    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #    elif isinstance(m, nn.Linear):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #for name, m in self.branch_2.named_modules():
        #    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #    elif isinstance(m, nn.Linear):
        #        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, data, step=1):
        if not self.training:
            return self.branch_1(data)
        if step == 1:
            return self.branch_1(data)
        elif step == 2:
            return self.branch_2(data)


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3))
                                     for s in range(num_stages-1)])

    def forward(self, x, mask):
        middle_out, out = self.tower_stage(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return middle_out, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualCasualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                     for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class DilatedResidualCasualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualCasualLayer, self).__init__()
        self.padding = 2 * int(dilation + dilation * (kernel_size -3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.pad(x, [self.padding, 0], 'constant', 0)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, test_features, test_gt_path, phase2label, device, num_blocks, num_layers, num_f_maps, dim, num_classes, start_epoch, pseudo):
        self.test_features = test_features
        self.test_gt_path = test_gt_path
        self.phase2label = phase2label
        self.device = device
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.start_epoch = start_epoch
        self.pseudo = pseudo

    def confidence_loss(self, pred, confidence_mask):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(self.device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss


    def margin_loss(self, batch_mask, middle_pred, thres=3):
        batch_size = batch_mask.size(0)
        loss = 0
        total = 0
        for b in range(batch_size):
            mask = batch_mask[b].squeeze(0)
            timestamp = torch.where(mask!=-100)[0]
            feature = middle_pred[b]
            for i in range(len(timestamp)):
                for j in range(i+1, len(timestamp)):
                    if torch.norm(feature[:, timestamp[i]] - feature[:, timestamp[j]]) < thres:
                        loss += thres - torch.norm(feature[:, timestamp[i]] - feature[:, timestamp[j]])
                        total += 1

        if total > 0:
            loss = loss / total
        return loss


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate):
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print('start epoch of single supervision is:', self.start_epoch)
        writer = SummaryWriter(save_dir)
        cnt = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(self.device), batch_target.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)

                timestamp_mask = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                if self.pseudo == 'paper':
                    # Generate pseudo labels after training 30 epochs for getting more accurate labels
                    if epoch < self.start_epoch:
                        batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                    else:
                        batch_boundary = batch_gen.get_boundary(batch_size, middle_pred.detach())
                elif self.pseudo == 'uniform':
                    if epoch < self.start_epoch:
                        batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                    else:
                        batch_boundary = batch_gen.get_average(batch_size, batch_input.size(-1))
                elif self.pseudo == 'naive':
                    batch_boundary = batch_gen.get_single_random(batch_size, batch_input.size(-1))
                elif self.pseudo == 'full':
                    batch_boundary = batch_gen.get_full_annotations(batch_size, batch_input.size(-1))

                batch_boundary = batch_boundary.to(self.device)
                loss = 0
                if False and self.pseudo in ['paper', 'uniform'] and epoch == num_epochs-1:
                    vis_dir = os.path.join(save_dir, 'train_vis')
                    if not os.path.exists(vis_dir):
                        os.makedirs(vis_dir)
                    for vi in range(len(batch_input)):
                        cnt += 1
                        label = batch_target[vi].squeeze(0)
                        pseudo_label = batch_gen.get_boundary(batch_size, middle_pred.detach())[vi].squeeze(0)
                        predicted = torch.max(predictions[-1].data, 1)[1][vi].squeeze(0)
                        if len(torch.where(label==-100)[0]) != 0:
                            length = int(torch.where(label==-100)[0][0])
                            label, pseudo_label, predicted = label[:length], pseudo_label[:length], predicted[:length]
                        self.segment_bars(os.path.join(vis_dir, '{}.png'.format(cnt)), ['gt', label], ['pseudo label', pseudo_label], ['predict', predicted])
                for p in predictions:
                    ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_boundary.view(-1))
                    loss += ce_loss
                    #print('ce', cs_loss)
                    smooth_loss = 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:])
                    loss += smooth_loss
                    #print('smooth', smooth_loss)
                    if self.pseudo in ['paper', 'uniform']:
                        confidence_loss = 0.075 * self.confidence_loss(p, batch_confidence)
                        loss += confidence_loss
                #margin_loss = 0.05 * self.margin_loss(timestamp_mask, middle_pred)
                #loss += margin_loss

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_samples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_samples),
                                                               float(correct)/total))
            if epoch == num_epochs-1:
                self.predict(save_dir, epoch+1, batch_gen.sample_rate)

    def predict(self, model_dir, epoch, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            videos = [os.path.join(self.test_features, x) for x in sorted(os.listdir(self.test_features))]
            annotations = [os.path.join(self.test_gt_path, x) for x in sorted(os.listdir(self.test_gt_path))]
            vis_dir = os.path.join(model_dir, 'visualization')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            all_pred_phase = []
            all_label_phase = []
            correct_phase = 0
            total_phase = 0

            for video, anno in tqdm(list(zip(videos, annotations))):
                # print(vid)
                features = np.load(video).transpose()
                features = features[:, ::sample_rate]
                with open(anno, 'r') as f:
                    content = f.read().split('\n')
                    if content[-1] == '':
                        content = content[:-1]
                labels = np.zeros(len(content))
                for i in range(len(content)):
                    labels[i] = self.phase2label[content[i].strip().split()[1]]
                labels = torch.Tensor(labels[::sample_rate]).long().to(self.device)

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(self.device)
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=self.device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()

                self.segment_bars(os.path.join(vis_dir, '{}.png'.format(video.split('/')[-1].split('.')[0])), ['gt', labels], ['predict', predicted])

                correct_phase += torch.sum(predicted == labels)
                total_phase += len(predicted)
                for i in range(len(predicted)):
                    all_pred_phase.append(int(predicted.data.cpu()[i]))
                for i in range(len(labels)):
                    all_label_phase.append(int(labels[i]))

        accuracy = correct_phase / total_phase
        precision = metrics.precision_score(all_label_phase, all_pred_phase, average='macro')
        recall = metrics.recall_score(all_label_phase, all_pred_phase, average='macro')
        jaccard = metrics.jaccard_score(all_label_phase, all_pred_phase, average='macro')
        F1 = metrics.f1_score(all_label_phase, all_pred_phase, average='macro')
        print('Evaluating from {} at epoch {}'.format(model_dir, epoch))
        print('Accuracy: {:.4}'.format(accuracy))
        print('Precision: {:.4f}'.format(precision))
        print('Recall: {:.4f}'.format(recall))
        print('Jaccard: {:.4f}'.format(jaccard))
        print('F1 score: {:.4f}'.format(F1))


    def segment_bars(self, save_path, *labels):
        titles, labels = zip(*labels)
        labels = [label.detach().cpu().numpy() for label in labels]
        num_pics = len(labels)
        color_map = plt.cm.Pastel1
        barprops = dict(aspect='auto', cmap=color_map)
        fig = plt.figure(figsize=(15, (num_pics+1)*1.5))
        for i, label in enumerate(labels):
            plt.subplot(num_pics, 1,  i+1)
            plt.xticks([])
            plt.yticks([])
            plt.title(titles[i], y=0.35, x=1.1, fontsize=15)
            plt.imshow([label], **barprops)
        plt.savefig(save_path)
        plt.close()
