import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_logger
from train_spec import build_optimizer, build_lr_scheduler, get_trainable_params

class Trainer:
    def __init__(self, config):
        self.dataset = config.dataset
        self.device = config.device
        self.spec = config.spec
        self.model = build_model(config.model)
        self.num_epochs = config.num_epochs
        self.train_style = config.train_style
        self.logger = get_logger(
                filedir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output', 'log'),
                )

        self.optimizer = self.build_optimizer(
                                params=self.get_trainable_params(
                                    classifier_prefix={"model.fc", "shadow.fc", "fc"}
                                    )
                                )
        self.lr_scheduler = self.build_lr_scheduler()


    def run(self):
        for epcoh in range(self.num_epochs):
            if self.train_style == 'supervised':
                self.training_epoch_supervised()
            elif self.train_style == 'cps':
                self.training_epoch_cps()
            elif self.train_style == 'simple':
                self.training_epoch_simple()
            elif self.train_style == 'fixmatch':
                self.training_epoch_fixmatch()

    def training_epoch_supervised(self):
        self.model.train()
        for data, label in enumerate(self.dataloader['train']):
            pass

    def training_epoch_cps(self):
        pass

    def training_epoch_simple(self, batch):
        self.model.train()
        (x_inputs, x_targets), (u_inputs, u_true_targets) = batch
        batch_size = len(x_inputs)
        x_inputs = x_inputs.to(self.device)
        x_targets = x_targets.to(self.device)
        u_inputs = u_inputs.to(self.device)
        u_true_targets = u_true_targets.to(self.device)



    def training_epoch_fixmatch(self):
        pass

    def build_optimizer(self, params):
        return build_optimizer(optimizer_type=self.spec.optimizer_type,
                params=params,
                learning_rate=self.spec.learning_rate,
                weight_decay=self.spec.weight_decay,
                momentum=self.spec.optimizer_momentum
                )

    def build_lr_scheduler(self):
        return build_lr_scheduler(scheduler_type=self.spec.lr_scheduler_type,
                optimizer=self.optimizer,
                num_epochs=self.num_epochs)

    def get_trainable_params(self, classfier_prefix):
        return get_trainable_params(model=self.model,
                learning_rate=self.spec.learning_rate,
                feature_learning_rate=self.spec.feature_learning_rate,
                classfier_prefix=classfier_prefix,
                requires_grad_only=True)

