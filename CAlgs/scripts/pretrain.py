import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import train

def t(data_name, arch_name, epochs, save_path=None, pretrained_model_path=None):
    # データ設定
    data = {
        'dir': './data',
        'name': data_name
    }

    # 保存先設定
    if save_path is None:
        save_path = './snapshots/{}/{}/train/'.format(data_name, arch_name)

    # モデル設定
    if pretrained_model_path:
        arch = {
            'dir': pretrained_model_path,
            'name': arch_name
        }
    else:
        arch = arch_name

    # トレーニング実行
    print(train.Train(data, save_path, arch, epochs=epochs).main())


if __name__ == '__main__':
    # 引数処理
    data_name = sys.argv[1]                # ex: mini_cifar100
    arch_name = sys.argv[2]                # ex: vgg16
    epochs = int(sys.argv[3])              # ex: 200
    save_path = sys.argv[4] if len(sys.argv) > 4 else None
    pretrained_model_path = sys.argv[5] if len(sys.argv) > 5 else None

    # 実行
    t(data_name, arch_name, epochs, save_path, pretrained_model_path)

# python scripts/pretrain.py mini_cifar100 vgg16 200 \
    # /content/drive/MyDrive/AutoMC_models/cifar100/vgg16/mini_retrain \
    # /content/drive/MyDrive/AutoMC_models/cifar100/vgg16/train/best.pth

