import os
import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, phase2label, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.phase2label = phase2label
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}

        self.timestamp = np.load(os.path.join(os.path.dirname(gt_path), "timestamp.npy"), allow_pickle=True).item()
        self.read_data()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_samples)

    def has_next(self):
        if self.index < len(self.list_of_samples):
            return True
        return False

    def read_data(self):
        videos = [os.path.join(self.features_path, x) for x in sorted(os.listdir(self.features_path))]
        annotations = [os.path.join(self.gt_path, x) for x in sorted(os.listdir(self.gt_path))]
        self.list_of_samples = list(zip(range(len(videos)), videos, annotations))
        random.shuffle(self.list_of_samples)
        self.generate_confidence_mask()

    def generate_confidence_mask(self):
        for sample in self.list_of_samples:
            vid, _, anno = sample
            with open(anno, 'r') as f:
                content = f.read().split('\n')[:-1]
            labels = np.zeros(len(content))
            for i in range(len(content)):
                labels[i] = self.phase2label[content[i].strip().split()[1]]
            labels = labels[::self.sample_rate]
            self.gt[vid] = labels
            num_frames = len(labels)

            random_timestamp = self.timestamp[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_timestamp) - 1):
                left_mask[int(labels[random_timestamp[j]]), random_timestamp[j]:random_timestamp[j + 1]] = 1
                right_mask[int(labels[random_timestamp[j + 1]]), random_timestamp[j]:random_timestamp[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    def next_batch(self, batch_size):
        batch = self.list_of_samples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        for vid, video, _ in batch:
            features = np.load(video).transpose()
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            assert np.shape(batch_input[i])[1] == np.shape(batch_target[i])[0]
            seq_length = np.shape(batch_input[i])[1]
            batch_input_tensor[i, :, :seq_length] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :seq_length] = torch.from_numpy(batch_target[i])
            mask[i, :, :seq_length] = torch.ones(self.num_classes, seq_length)

        return batch_input_tensor, batch_target_tensor, mask, batch_confidence

    def get_full_annotations(self, batch_size, max_frames):
        batch = self.list_of_samples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            gt = self.gt[vid]
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, list(range(len(gt_tensor)))] = gt_tensor

        return boundary_target_tensor

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_samples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_frame = self.timestamp[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def get_average(self, batch_size, max_frames):
        batch = self.list_of_samples[self.index - batch_size:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_frame = self.timestamp[vid]
            gt = self.gt[vid]
            gt_tensor = torch.from_numpy(gt.astype(int))
            last_bound = 0
            for i in range(len(single_frame) - 1):
                center = int((single_frame[i] + single_frame[i+1]) / 2)
                boundary_target_tensor[b, last_bound: center] = gt_tensor[single_frame[i]]
                last_bound = center
            boundary_target_tensor[b, last_bound:] = gt_tensor[single_frame[-1]]
        return boundary_target_tensor

    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_samples[self.index - batch_size:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(left_score) + start + 1
                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(right_score) + start + 1
                right_bound.append(cur_bound.item())

            # Average two action boundaries for same segment and generate pseudo labels
            left_bound = left_bound[1:]
            right_bound = right_bound[1:]
            num_bound = len(left_bound)
            for i in range(num_bound):
                temp_left = left_bound[i]
                temp_right = right_bound[num_bound - i - 1]
                middle_bound = int((temp_left + temp_right)/2)
                boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
                boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor
