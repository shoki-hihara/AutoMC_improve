import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import train


def t(data_name, arch_name, epochs, save_path=None):
    data = {'dir':'./data', 'name':data_name}
    if save_path is None:
        save_path = './snapshots/{}/{}/train/'.format(data_name, arch_name)
        
    print(train.Train(data, save_path, arch_name, epochs=epochs).main())

if __name__ == '__main__':
    data_name, arch_name, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
    
    if len(sys.argv) > 4:
        save_path = sys.argv[4]
    else:
        save_path = None
        
    t(data_name, arch_name, epochs)
    
# !python scripts/pretrain.py cifar100 vgg16 200 /content/drive/MyDrive/AutoMC_models/cifar100/vgg16/
