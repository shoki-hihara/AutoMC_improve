import os
import sys

# この2行を追加（パス解決用）
cpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if cpath not in sys.path:
    sys.path.insert(0, cpath)

import train  # ←これで必ず読み込めるようになる


def t(data_name, arch_name, epochs, save_path=None, pretrained_model_path=None):
    # データ設定（Colab ローカルに保存）
    data = {
        'dir': '/content/data',
        'name': data_name
    }

    # 保存先設定（Colab ローカルに保存）
    if save_path is None:
        save_path = '/content/snapshots/{}/{}/train/'.format(data_name, arch_name)
    os.makedirs(save_path, exist_ok=True)

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
