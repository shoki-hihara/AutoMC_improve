'''
Our implement of SFP
'''
import os
import numpy as np
import torch
import torch.nn as nn
import math
import copy

from utils import *
import train
import models


class SoftFilterPruning:
    def __init__(self, data, save_dir, arch, epochs=300, rate=0.9, epoch_prune=1, additional_fine_tune_epochs=3, kd_params=(None, None), fixed_seed=False, use_logger=True):
        self.cuda = torch.cuda.is_available()

        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        self.arch = arch['dir']
        self.arch_name = arch['name']
        self.rate = math.sqrt(rate)
        self.epochs = epochs
        self.epoch_prune = epoch_prune
        self.additional_fine_tune_epochs = additional_fine_tune_epochs
        self.kd_params = kd_params
        self.lr = 0.025  # default lr is 0.1
        self.lr_sche = 'StepLR'
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger is True:
            self.logger = set_logger('{}_{}_C4'.format(self.data_name, self.arch_name), self.save_dir)
        elif self.use_logger is False or self.use_logger is None:
            self.logger = None
        else:
            self.logger = use_logger

    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting C4")

        # Load original model (map_location='cuda' or 'cpu' for portability)
        device = 'cuda' if self.cuda else 'cpu'
        model = torch.load(self.arch, map_location=device)
        if self.logger:
            self.logger.info("Loaded model '{}' from {}".format(self.arch_name, self.arch))
            self.logger.info("The original model's cfg={}".format(model.cfg))

        if self.cuda:
            model = model.cuda()

        # Define loss function (criterion), optimizer and lr scheduler
        criterion = nn.CrossEntropyLoss().cuda() if self.cuda else nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), self.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        lr_scheduler = get_lr_scheduler(self.lr_sche, optimizer, self.epochs)

        # Load data
        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        if self.logger:
            self.logger.info("Loaded dataset '{}' from '{}'".format(self.data_name, self.data_dir))

        # Get filename for saving
        filename, bestname = get_filename_training(self.save_dir, 'C4')

        # Initialize mask
        m = Mask(model, self.arch_name, self.logger)
        m.init_length()

        # Test before compressing
        metrics_original = test_at_beginning(model, val_loader, self.logger)

        # Start training
        if self.logger:
            self.logger.info("Starting training for {} epoch(s)".format(self.epochs))
        best_acc_top1 = 0
        best_val_metrics = {}
        timer = train_timer()

        for epoch in range(self.epochs):
            # Run one epoch
            need_time = timer.get_need_time(self.epochs, epoch)
            if self.logger:
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}  {need_time}  lr={optimizer.param_groups[0]['lr']}")

            # Train for one epoch
            train.train(model, None, criterion, optimizer, train_loader, self.logger, kd_params=self.kd_params)

            lr_scheduler.step()

            if (epoch % self.epoch_prune == 0 or epoch == self.epochs - 1):
                now_prune = True
                # Evaluate on validation set before masking
                validate(model, val_loader, self.logger)
                if self.logger:
                    self.logger.info("Doing mask...")
                m.model = model
                m.do_mask(self.rate, self.arch_name)
                model = m.model
                if self.cuda:
                    model = model.cuda()
            else:
                now_prune = False

            # Evaluate on validation set
            val_metrics = validate(model, val_loader, self.logger)

            if now_prune:
                val_acc_top1 = val_metrics['acc_top1']
                is_best = val_acc_top1 > best_acc_top1

                # Save weights
                save_checkpoint(model, is_best, filename, bestname)
                if is_best:
                    if self.logger:
                        self.logger.info("- Found new best accuracy on validation set")
                    best_acc_top1 = val_acc_top1
                    best_val_metrics = val_metrics
                    # Save best val metrics in a json file in the model directory
                    best_json_path = os.path.join(self.save_dir, "metrics_val_best_weights.json")
                    save_dict_to_json(val_metrics, best_json_path)

                # Save latest val metrics in a json file in the model directory
                last_json_path = os.path.join(self.save_dir, "metrics_val_last_weights.json")
                save_dict_to_json(val_metrics, last_json_path)

            timer.update()

        old_model = torch.load(bestname, map_location=device)
        small_model = get_small_model(old_model, self.arch_name, 'conv')

        model_dir, val_metrics = fine_tune(
            self.save_dir, small_model, train_loader, val_loader,
            epochs=self.additional_fine_tune_epochs, lr=self.lr, lr_sche=self.lr_sche,
            logger=self.logger, kd_params=self.kd_params, return_file=True, use_logger=self.use_logger
        ).main()

        # Calculate metrics
        result = calc_result(torch.load(self.arch, map_location=device), metrics_original, torch.load(model_dir, map_location=device), val_metrics, model_dir, self.logger)
        save_result_to_json(self.save_dir, result)

        if self.use_logger is True:
            close_logger()
        return result


class Mask:
    def __init__(self, model, model_name, logger):
        self.cuda = torch.cuda.is_available()
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.modules = get_modules(model, model_name, 'conv')
        self.model_name = model_name
        self.logger = logger

    def get_filter_codebook(self, weight_torch, compress_rate, length, last_filter_index, now_filter_index=[-1]):
        codebook = np.ones(length)
        # Only applied to conv layer
        if len(weight_torch.size()) == 4:
            if len(now_filter_index) == 1 and now_filter_index[0] == -1:
                # The filter to be pruned in each conv layer
                filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
                # Flatten each filter
                weight_vec = weight_torch.view(weight_torch.size()[0], -1)
                norm2 = torch.norm(weight_vec, 2, 1)
                norm2_np = norm2.cpu().numpy()
                # Sort from little to large, return index
                filter_index = norm2_np.argsort()[:filter_pruned_num]
            else:
                filter_index = now_filter_index

            # Size of flattened filter
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            mini_kernel_length = weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(weight_torch.size()[0]):
                for y in range(len(last_filter_index)):
                    codebook[x * kernel_length + last_filter_index[y] * mini_kernel_length: x * kernel_length + (last_filter_index[y] + 1) * mini_kernel_length] = 0

        else:
            if self.logger:
                self.logger.error("Error while getting filter codebook!")
            raise Exception('Error while getting filter codebook!')
        return codebook, filter_index

    def init_length(self):
        for index in range(len(self.modules)):
            if len(self.modules[index][1]) > 0:
                self.model_size[index] = self.modules[index][0].weight.data.size()

        for index1 in self.model_size:
            for index2 in range(len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        for index in range(len(self.modules)):
            if 'prune' in self.modules[index][1]:
                self.compress_rate[index] = layer_rate

    def do_mask(self, layer_rate, arch_name):
        self.init_rate(layer_rate)
        filter_index = {}
        last_filter_index = []
        for index in range(len(self.modules)):
            if 'criterion' in self.modules[index][1]:
                self.mat[index], filter_index[index] = self.get_filter_codebook(
                    self.modules
