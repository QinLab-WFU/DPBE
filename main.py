import argparse
import torch
from train.DPBE.hash_train import DPBETrainer

trainers = {
    'DPBE': DPBETrainer
}

torch.backends.cudnn.benchmark = True    # 允许自动优化卷积算法
torch.backends.cudnn.deterministic = False  # 关闭确定性模式（提升速度）
torch.set_float32_matmul_precision('high')  # 启用 Tensor Core 加速（Ampere+架构）

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='DPBE', help="Trainer method name")

    parser.add_argument("--dataset", type=str, default="flickr", help="name of dataset")
    parser.add_argument("--output-dim", type=int, default=16)

    parser.add_argument("--is-train", default=True)
    args = parser.parse_args()
    
    trainer = trainers.get(args.method)
    trainer(args, 0)
