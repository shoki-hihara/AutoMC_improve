'''
Our implement of LeGR
'''
import os
import time
import torch
import queue
import numpy as np
import torch.nn as nn
import math

from utils import *
from train import *
import models
import copy

class LeGR:
    def __init__(self, data, save_dir, arch, fine_tune_epochs=1, additional_fine_tune_epochs=300, rate=0.9, max_prune_per_layer=0.9, generations=400, rank_type='l2_weight', kd_params=(None, None), fixed_seed=False, use_logger=True):
        self.cuda = torch.cuda.is_available()
        
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        self.arch = arch['dir']  # Googleドライブのパスを想定
        self.arch_name = arch['name']
        self.rate = rate
        self.max_prune_per_layer = max_prune_per_layer
        self.rank_type = rank_type
        self.kd_params = kd_params
        self.fine_tune_epochs = fine_tune_epochs
        self.additional_fine_tune_epochs = additional_fine_tune_epochs
        self.generations = generations
        self.lr = 1e-3
        self.lr_sche = 'MultiStepLR'
        if rank_type == 'l1_weight' or rank_type == 'l2_weight':
            self.label_layer = nn.Conv2d
            self.prune_type = 'conv'
        else:
            self.label_layer = nn.BatchNorm2d
            self.prune_type = 'bn'
        self.filter_ranks = {}
        if rank_type not in ['l1_weight', 'l2_weight', 'l2_bn', 'l2_bn_param']:
            raise ValueError('rank_type error!')
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger == True:
            self.logger = set_logger('{}_{}_C2'.format(self.data_name, self.arch_name), self.save_dir)
        elif self.use_logger == False:
            self.logger = None
        else:
            self.logger = use_logger

    def get_filter_ranks(self):
        if self.logger:
            self.logger.info("Getting filter's original ranking...")
        modules = get_modules(self.model, self.arch_name, self.prune_type)

        for index in range(len(modules)):
            layer = modules[index][0]
            if isinstance(layer, nn.Conv2d):
                if 'shortcut' not in modules[index][1]:
                    in_params = layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)

                if self.rank_type == 'l1_weight':
                    self.filter_ranks[index] = (torch.abs(layer.weight.data)).sum(1).sum(1).sum(1)
                elif self.rank_type == 'l2_weight': 
                    self.filter_ranks[index] = (torch.pow(layer.weight.data, 2)).sum(1).sum(1).sum(1)
                
            elif isinstance(layer, nn.BatchNorm2d):
                if self.rank_type == 'l2_bn':
                    self.filter_ranks[index] = torch.pow(layer.weight.data, 2)
                elif self.rank_type == 'l2_bn_param': 
                    self.filter_ranks[index] = torch.pow(layer.weight.data, 2) * in_params

    def get_thre(self, filter_ranks_copy, model):
        def get_rate(thre):
            model_copy = copy.deepcopy(model)
            modules = get_modules(model_copy, self.arch_name, self.prune_type)
            return self.get_compressed_model(modules, thre, filter_ranks_copy, model_copy, small_model_with_param=False)[1]
        
        num = 0
        thre_candidates = []
        for k in filter_ranks_copy:
            for i in filter_ranks_copy[k]:
                thre_candidates.append(i.float())
            num += len(filter_ranks_copy[k])
        
        thre_candidates.sort() #  = rv_duplicate_ele(thre_candidates, sort=True)

        if self.logger:
            self.logger.info('length of thre candidates:' + str(len(thre_candidates)))
        l, r = 0, len(thre_candidates) - 1
        while l < r - 1:
            mid = (l + r) // 2
            if get_rate(thre_candidates[mid]) > self.rate: l = mid
            else: r = mid

        rate1 = abs(get_rate(thre_candidates[l]) - self.rate)
        rate2 = abs(get_rate(thre_candidates[r]) - self.rate)
        target_thre = thre_candidates[l] if rate1 < rate2 else thre_candidates[r]
        now_rate = get_rate(target_thre)
        if self.logger:
            self.logger.info('now_rate is {}, target rate is {}'.format(now_rate, self.rate))
        return target_thre

    def get_mask(self, weight_copy, thre):
        mask = weight_copy.gt(thre).float().cuda()
        current_rate = mask.sum(dim=0) / len(mask)
        if 1 - current_rate > self.max_prune_per_layer:
            _, sort_index = torch.sort(weight_copy)
            mask = torch.ones_like(mask).float().cuda()
            for i in range(min(int(torch.floor(torch.tensor(self.max_prune_per_layer * len(mask)))), len(mask) - 1)):
                mask[sort_index[i]] = 0

        return mask

    def get_compressed_model(self, modules, thre, filter_ranks_copy, model_copy, small_model_with_param=True):
        pruned = 0
        for index in range(len(modules)):
            m = modules[index][0]
            if isinstance(m, self.label_layer):
                mask = self.get_mask(filter_ranks_copy[index], thre)
                pruned += mask.shape[0] - torch.sum(mask)
                for i in range(len(mask)):
                    m.weight.data[i].mul_(mask[i])
                    if self.label_layer == nn.BatchNorm2d:
                        m.bias.data[i].mul_(mask[i])

        compressed_model = get_small_model(model_copy, self.arch_name, self.prune_type, small_model_with_param=small_model_with_param)
        compression_rate = 1 - pruned / self.total
        if small_model_with_param:
            if self.logger:
                self.logger.info('bias of pruning rate(num of channel):' + str(abs(compression_rate - self.rate)))
        compression_rate = get_compression_rate(self.original_model, compressed_model)
        if small_model_with_param:
            if self.logger:
                self.logger.info('bias of pruning rate(num of parameters):' + str(abs(compression_rate - self.rate)))
        return compressed_model, compression_rate

    def pruning_with_transformations(self, perturbation):
        filter_ranks_copy = self.filter_ranks.copy()
        assert(len(filter_ranks_copy) == len(perturbation))
        for k, i in zip(sorted(filter_ranks_copy.keys()), range(len(filter_ranks_copy))):
            filter_ranks_copy[k] = filter_ranks_copy[k] * perturbation[i][0] + perturbation[i][1]

        model_copy = copy.deepcopy(self.model)
        modules = get_modules(model_copy, self.arch_name, self.prune_type)
        thre = self.get_thre(filter_ranks_copy, model_copy)

        compressed_model, compression_rate = self.get_compressed_model(modules, thre, filter_ranks_copy, model_copy)
        if self.logger:
            self.logger.info(str(compressed_model.cfg))

        self.filter_ranks = filter_ranks_copy

        return compression_rate, compressed_model

    def learn_ranking_ea(self):
        if self.logger:
            self.logger.info('Learning ranking through ea...')
        if self.kd_params == (None, None):
            original_model = None
        else:
            original_model = self.model

        self.total = 0
        modules = get_modules(self.model, self.arch_name, self.prune_type)
        for i in range(len(modules)):
            m = modules[i][0]
            if isinstance(m, self.label_layer):
                self.total += m.weight.data.shape[0]

        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        start_t = time.time()

        mean_loss = []
        minimum_loss = 10
        best_perturbation = None
        POPULATIONS = 64
        SAMPLES = 16
        generations = self.generations
        SCALE_SIGMA = 1
        MUTATE_PERCENT = 0.1
        index_queue = queue.Queue(POPULATIONS)
        population_loss = np.zeros(0)
        population_data = []

        original_dist = self.filter_ranks.copy()
        original_dist_stat = {}
        for k in sorted(original_dist):
            a = original_dist[k].cpu().numpy()
            original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

        # Initialize Population
        for i in range(generations):
            step_size = 1 - (float(i) / (generations*1.25))
            # Perturn distribution
            perturbation = []

            if i == POPULATIONS - 1:
                for _ in range(len(self.filter_ranks)):
                    perturbation.append((1,0))
            elif i < POPULATIONS - 1:
                for k in sorted(self.filter_ranks.keys()):
                    scale = np.exp(float(np.random.normal(0, SCALE_SIGMA)))
                    shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale, shift))
            else:
                mean_loss.append(np.mean(population_loss))
                sampled_idx = np.random.choice(POPULATIONS, SAMPLES)
                sampled_loss = population_loss[sampled_idx]
                winner_idx_ = np.argmin(sampled_loss)
                winner_idx = sampled_idx[winner_idx_]
                oldest_index = index_queue.get()

                # Mutate winner
                base = population_data[winner_idx]
                # Perturb distribution
                mnum = int(MUTATE_PERCENT * len(self.filter_ranks))
                mutate_candidate = np.random.choice(len(self.filter_ranks), mnum)
                for k, j in zip(sorted(self.filter_ranks.keys()), range(len(self.filter_ranks))):
                    scale = 1
                    shift = 0
                    if j in mutate_candidate:
                        scale = np.exp(float(np.random.normal(0, SCALE_SIGMA * step_size)))
                        shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale * base[j][0], shift + base[j][1]))
            
            self.filter_ranks = original_dist

            # Given affine transformations, rank and prune
            compression_rate, compressed_model = self.pruning_with_transformations(perturbation)
            
            compressed_model, val_metrics = fine_tune(self.save_dir, compressed_model, train_loader, val_loader, epochs=max(self.fine_tune_epochs, 1), lr=self.lr, logger=self.logger, original_model=original_model, kd_params=self.kd_params, use_logger=self.use_logger).main()
            
            loss = val_metrics['loss']

            if loss < minimum_loss:
                minimum_loss = loss
                best_perturbation = perturbation
                best_model = (compression_rate, compressed_model)
            
            if i < POPULATIONS:
                index_queue.put(i)
                population_data.append(perturbation)
                population_loss = np.append(population_loss, [loss])
            else:
                population_data[oldest_index] = perturbation
                population_loss[oldest_index] = loss
                index_queue.put(oldest_index)

            if self.logger:
                self.logger.info('Generation {}/{} Loss: {} Rate: {}'.format(i, generations, loss, compression_rate))

        if self.logger:
            self.logger.info('best compression_rate: ' + str(best_model[0]))
        return best_model

    def main(self):
        # Googleドライブからのモデルロード
        if self.logger:
            self.logger.info("Loading model from {}".format(self.arch))
        self.model = torch.load(self.arch)
        if self.cuda:
            self.model.cuda()

        self.model.eval()

        # 初期フィルターランキング取得
        self.get_filter_ranks()

        # EAを使ってフィルターのランキングを学習
        compression_rate, compressed_model = self.learn_ranking_ea()

        if self.logger:
            self.logger.info("Fine tuning final compressed model...")
        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)

        # 最終ファインチューニング
        compressed_model, val_metrics = fine_tune(self.save_dir, compressed_model, train_loader, val_loader, epochs=self.additional_fine_tune_epochs, lr=self.lr, logger=self.logger, kd_params=self.kd_params, use_logger=self.use_logger).main()

        if self.logger:
            self.logger.info("Finished fine tuning.")
            self.logger.info("Validation Loss: {}".format(val_metrics['loss']))
            self.logger.info("Validation Accuracy: {}".format(val_metrics['acc']))

        # モデル保存
        save_path = os.path.join(self.save_dir, 'compressed_model.pth')
        torch.save(compressed_model, save_path)

        return compressed_model
