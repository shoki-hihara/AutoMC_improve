import sys, os

# 実行ファイル(run_Compression_Methods.py)の位置から見てprune_C1.pyがあるディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import prune_C1
import prune_C2
import prune_C3
import prune_C4
import prune_C5
import prune_C7

# 実行ファイル(run_Compression_Methods.py)の位置から見てprune_C1.pyがあるディレクトリを追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run(data_name, arch_name, alg, rate, data_root, model_root, output_root):
    # ディレクトリ設定（Google Drive含む任意パス）
    data = {
        'dir': os.path.join(data_root),
        'name': data_name
    }
    # arch = {
    #     'dir': os.path.join(model_root, data_name, f"{arch_name}.pth.tar"),
    #     'name': arch_name
    # }
    arch = {
    'dir': model_root,  # ここは既にフルパスなので、そのまま渡す
    'name': arch_name
    }
    save_dir = os.path.join(output_root, data_name, arch_name, str(rate))

    print(f"[INFO] Using model: {arch['dir']}")
    print(f"[INFO] Saving results to: {save_dir}")

    if '1' in alg:
        print("Running prune_C1 (Knowledge Distillation)...")
        save_subdir = os.path.join(save_dir, 'C1')
        res = prune_C1.knowledge_distillation(data, save_subdir, arch, rate_based_on_original=rate, fixed_seed=True).main()

    elif '2' in alg:
        print("Running prune_C2 (LeGR)...")
        save_subdir = os.path.join(save_dir, 'C2')
        res = prune_C2.LeGR(data, save_subdir, arch, rate=rate, fixed_seed=True).main()

    elif '3' in alg:
        print("Running prune_C3 (Network Slimming)...")
        save_subdir = os.path.join(save_dir, 'C3')
        res = prune_C3.NetworkSlimming(data, save_subdir, arch, rate=rate, fixed_seed=True).main()

    elif '4' in alg:
        print("Running prune_C4 (Soft Filter Pruning)...")
        save_subdir = os.path.join(save_dir, 'C4')
        res = prune_C4.SoftFilterPruning(data, save_subdir, arch, rate=rate, fixed_seed=True).main()

    elif '5' in alg:
        print("Running prune_C5 (HOS)...")
        save_subdir = os.path.join(save_dir, 'C5')
        res = prune_C5.HOS(data, save_subdir, arch, rate=rate, fixed_seed=True).main()

    elif '7' in alg:
        print("Running prune_C7 (LFB)...")
        save_subdir = os.path.join(save_dir, 'C7')
        res = prune_C7.LFB(data, save_subdir, arch_name, rate=rate, fixed_seed=True).main()
    
    else:
        print(f"[ERROR] Unknown algorithm identifier: {alg}")
        return

    print("\n[RESULT]")
    print(res)


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print("Usage: python run_compression_methods.py <data_name> <arch_name> <alg> <rate> <data_root> <model_root> <output_root>")
        sys.exit(1)

    data_name = sys.argv[1]     # e.g., cifar100
    arch_name = sys.argv[2]     # e.g., vgg16
    alg = sys.argv[3]           # e.g., 1, 2, ..., 7
    rate = float(sys.argv[4])   # e.g., 0.3
    data_root = sys.argv[5]     # e.g., /content/drive/MyDrive/AutoMC/data
    model_root = sys.argv[6]    # e.g., /content/drive/MyDrive/AutoMC/trained_models
    output_root = sys.argv[7]   # e.g., /content/drive/MyDrive/AutoMC/results

    run(data_name, arch_name, alg, rate, data_root, model_root, output_root)
