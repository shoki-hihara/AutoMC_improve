import sys
import os
from mini_dataset_generator import mini_dataset

def f(data_name, arch_name_or_path, save_dir=None):
    # arch_name_or_pathがファイルパスならそのまま使う
    if os.path.exists(arch_name_or_path):
        arch_dir = arch_name_or_path
        arch_name = os.path.splitext(os.path.basename(arch_name_or_path))[0]
    else:
        arch_dir = './trained_models/{}/{}.pth.tar'.format(data_name, arch_name_or_path)
        arch_name = arch_name_or_path

    arch = {'dir': arch_dir, 'name': arch_name}
    data = {'dir': './data', 'name': data_name}
    rate = 0.1

    if save_dir is None:
        save_dir = './data/mini_dataset/{}/{}/{}/'.format(data['name'], arch_name, rate)

    mini_dataset(data, save_dir, arch, rate).main()


if __name__ == '__main__':
    data_name, arch_name_or_path = sys.argv[1], sys.argv[2]

    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = None

    f(data_name, arch_name_or_path, save_dir)

# !python generate_mini_dataset.py cifar100 "/content/drive/MyDrive/学習/大学院/特別研究/AutoMC/vgg16.pth.tar" /content/drive/MyDrive/AutoMC_mini_datasets/cifar100/vgg16/
