'''
Our implement of LFB with Drive teacher loading, NaN detection, and timeout
'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

from utils import *
import models
import train


class TimeoutException(Exception):
    pass


class LFB:
    def __init__(self, data, save_dir, arch_name, epochs=300, rate=0.9, losses='1*CE', kd_params=(None, None), fixed_seed=False, use_logger=True, timeout_seconds=None):
        self.cuda = torch.cuda.is_available()
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        self.arch_name = arch_name
        self.rate = rate
        self.epochs = epochs
        self.lr = 1e-1
        self.lr_sche = 'MultiStepLR'
        self.losses = losses
        self.kd_params = kd_params
        self.timeout_seconds = timeout_seconds  # タイムアウト秒数（Noneなら無効）
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger == True:
            self.logger = set_logger('{}_{}_C7'.format(self.data_name, self.arch_name), self.save_dir)
        elif self.use_logger == False:
            self.logger = None
        else:
            self.logger = use_logger
        self.group, self.times = self.get_param()

    def get_param(self):
        if self.logger:
            self.logger.info("Calculating best parameters...")
        best_group, best_times, best_rate = None, None, None
        groups = [1, 2, 4, 8]
        timeses = [x / 10.0 for x in range(1, 100)]

        model_init = models.__dict__[self.arch_name]()
        for group in groups:
            for times in timeses:
                try:
                    model = models.__dict__[self.arch_name](special=('LFB', group, times))
                except Exception:
                    now_rate = 0
                else:
                    now_rate = get_compression_rate(model_init, model)
                    if best_rate is None or abs(now_rate - self.rate) < abs(best_rate - self.rate):
                        best_group, best_times = group, times
                        best_rate = now_rate
                if now_rate >= 1:
                    break
        if self.logger:
            self.logger.info("The actual rate is {}".format(best_rate))
            self.logger.info("group = {}, times = {}".format(best_group, best_times))
        return best_group, best_times


    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting C7")

        # --- Google Drive上などの絶対パス対応 ---
        # teacherモデル（元モデル）ロード
        model_original_path = '/content/drive/MyDrive/学習/大学院/特別研究/AutoMC/model_cifar100_vgg16_best.pth.tar'.format(self.arch_name)
        if os.path.isfile(model_original_path):
            model_original = torch.load(model_original_path)
            if self.logger:
                self.logger.info(f"Loaded teacher model from {model_original_path}")
        else:
            # 見つからなければ通常初期化
            model_original = models.__dict__[self.arch_name]()
            if self.logger:
                self.logger.info("Teacher model file not found, initialized new model_original")

        # compressed modelの生成
        model = models.__dict__[self.arch_name](num_classes=models.get_num_classes(self.data_name), special=('LFB', self.group, self.times))
        if self.logger:
            self.logger.info("Created compressed model '{}'".format(self.arch_name))
            self.logger.info("The model's cfg={}".format(model.cfg))

        if self.cuda:
            model = model.cuda()

        model_dir = os.path.join(self.save_dir, '{}_small_unfinetuned_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(model, model_dir)

        loss = Loss(self.losses, self.logger)
        if self.logger:
            self.logger.info("Created loss function '{}'".format(self.losses))
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = get_lr_scheduler(self.lr_sche, optimizer, self.epochs)

        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        if self.logger:
            self.logger.info("Loaded dataset '{}' from '{}'".format(self.data_name, self.data_dir))

        filename, bestname = get_filename_training(self.save_dir, 'C7')

        best_acc_top1 = 0.0
        best_val_metrics = {}
        timer = train_timer()

        start_time = time.time()

        for epoch in range(self.epochs):
            # タイムアウトチェック
            if self.timeout_seconds is not None and (time.time() - start_time) > self.timeout_seconds:
                if self.logger:
                    self.logger.warning(f"Timeout reached at epoch {epoch}, stopping training.")
                break

            need_time = timer.get_need_time(self.epochs, epoch)
            if self.logger:
                self.logger.info("Epoch {}/{}  {:s}  lr={}".format(epoch + 1, self.epochs, need_time, optimizer.param_groups[0]['lr']))

            try:
                # train中にNaNが発生しないように監視
                train.train(model, None, loss, optimizer, train_loader, self.logger, kd_params=self.kd_params)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Exception during training at epoch {epoch}: {e}")
                # NaNやその他の異常時はエポックスキップ（回避）
                continue

            # lr scheduler stepはtrain成功時のみ
            lr_scheduler.step()

            val_metrics = validate(model, val_loader, self.logger)
            # val_metricsにNaNがあれば警告を出し、そのエポックの評価は無視して続行
            if any(torch.isnan(torch.tensor(v)) for v in val_metrics.values()):
                if self.logger:
                    self.logger.warning(f"Validation metrics contain NaN at epoch {epoch}, skipping save.")
                continue

            val_acc_top1 = val_metrics['acc_top1']
            is_best = val_acc_top1 >= best_acc_top1

            save_checkpoint(model, is_best, filename, bestname)
            if is_best:
                if self.logger:
                    self.logger.info("- Found new best accuracy on validation set")
                best_acc_top1 = val_acc_top1
                best_val_metrics = val_metrics
                best_json_path = os.path.join(self.save_dir, "metrics_val_best_weights.json")
                save_dict_to_json(val_metrics, best_json_path)

            last_json_path = os.path.join(self.save_dir, "metrics_val_last_weights.json")
            save_dict_to_json(val_metrics, last_json_path)

            timer.update()

        model_dir = os.path.join(self.save_dir, '{}_small_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(model, model_dir)

        result = calc_result(model_original, {'acc_top1': 0, 'acc_top5': 0}, model, best_val_metrics, model_dir, self.logger)
        save_result_to_json(self.save_dir, result)

        if self.use_logger == True:
            close_logger()
        return result


class Loss(nn.modules.loss._Loss):
    def __init__(self, losses, logger):
        super(Loss, self).__init__()
        self.logger = logger
        if self.logger:
            self.logger.info('Preparing loss function...')
        self.cuda = torch.cuda.is_available()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in losses.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'NLL':
                loss_function = nn.NLLLoss()
            elif loss_type == 'CE':
                loss_function = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        if self.logger:
            self.logger.info('Loss function:')
        for l in self.loss:
            if l['function'] is not None:
                if self.logger:
                    self.logger.info('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        if self.cuda:
            self.loss_module = self.loss_module.cuda()

    def forward(self, prediction, label):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](prediction, label)
                # NaNチェック
                if torch.isnan(loss):
                    if self.logger:
                        self.logger.warning("Loss is NaN during forward pass.")
                    loss = torch.tensor(0.0).to(loss.device)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)

        loss_sum = sum(losses)
        return loss_sum


'''
if __name__ == '__main__':
    arch_name = 'vgg13'
    data = {'dir': './data', 'name': 'cifar100'}
    save_dir = './snapshots/{}/C7/'.format(arch_name)
    arch = arch_name
    l = LFB(data, save_dir, arch, rate=0.7, fixed_seed=False, epochs=1, timeout_seconds=3600)  # 例: 1時間でタイムアウト
    print(l.main())
'''
