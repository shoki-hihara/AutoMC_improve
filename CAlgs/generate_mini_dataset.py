import sys
from mini_dataset_generator import mini_dataset

def f(data_name, arch_name, save_dir=None):
    arch = {'dir': './trained_models/{}/{}.pth.tar'.format(data_name, arch_name), 'name': arch_name}
    data = {'dir': './data', 'name':data_name}
    rate = 0.1
    
    if save_dir is None:
        save_dir = './data/mini_dataset/{}/{}/{}/'.format(data['name'], arch_name, rate)
    
    mini_dataset(data, save_dir, arch, rate).main()


if __name__ == '__main__':
    data_name, arch_name = sys.argv[1], sys.argv[2]
    
    if len(sys.argv) > 3:
        save_dir = sys.argv[3]
    else:
        save_dir = None
        
    f(data_name, arch_name)
    
# !python generate_mini_dataset.py cifar100 vgg16 /content/drive/MyDrive/AutoMC_mini_datasets/cifar100/vgg16/
