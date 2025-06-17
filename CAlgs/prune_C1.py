'''
Our implement of LMA
'''
import torch
import math
import os

from utils import *
import models
from train import *


class knowledge_distillation:
    def __init__(self, data, save_dir, arch, epochs=300, rate_based_on_teacher=None, rate_based_on_original=0.7,
                 studentmodel_act_func='lma', lma_numBins=8, fixed_seed=False, kd_params=(0.7, 2), use_logger=True):
        self.cuda = torch.cuda.is_available()
        self.save_dir = save_dir
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.arch = arch['dir']  # This is now expected to be a full path to the teacher model in Google Drive
        self.arch_name = arch['name']
        self.use_logger = use_logger
        if self.use_logger is True:
            self.logger = set_logger('{}_{}_C1'.format(self.data_name, self.arch_name), save_dir)
        elif self.use_logger is False:
            self.logger = None
        else:
            self.logger = use_logger
        if rate_based_on_teacher is not None and rate_based_on_original is not None:
            if self.logger:
                self.logger.error('There can not be two rates at the same time.')
            raise ValueError('There can not be two rates at the same time.')
        if rate_based_on_teacher is None and rate_based_on_original is None:
            if self.logger:
                self.logger.error('There must be one rate.')
            raise ValueError('There must be one rate.')
        if rate_based_on_teacher is None:
            self.rate = rate_based_on_original
            self.cfg = None
        else:
            teacher_model_obj = torch.load(self.arch, map_location='cpu')
            self.rate = rate_based_on_teacher
            self.cfg = teacher_model_obj.cfg
        self.r
