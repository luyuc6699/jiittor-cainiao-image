import jittor as jt
import numpy as np
from tqdm import tqdm
from utils import load_config
from data import CustomImageDataset, get_train_val_transform
from models import CustomEfficientNet
import argparse
import os
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_args():
    parser = argparse.ArgumentParser(description="Jittor Bi-RADS 推理脚本")
    parser.add_argument("--test_dir",  type=str, required=True, help="测试集目录")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True,
                        help="多个权重路径，空格分隔")

    parser.add_argument("--cfg_path",   type=str,  default="./config/config.yaml")
    parser.add_argument("--test_bs",    type=int,  default=16)
    parser.add_argument("--result_path",type=str,  default="./result.txt")
    parser.add_argument("--weights",    type=float, nargs="+",
                        default=[0.4, 0.4, 0.2],
                        help="每个 ckpt 的融合权重，长度需与 checkpoint 一致")
    return parser.parse_args()


def test(model_paths, weights, test_loader, result_path, config):
    """
    model_paths: list[str]  三个模型的路径
    weights:     list[float] 对应权重，长度须与 model_paths 一致
    """
    assert len(model_paths) == len(weights), "模型数量和权重数量必须一致"
    weights = np.array(weights, dtype=np.float32)   
    weights /= weights.sum()                       

    models = []
    for model_path in model_paths:
        model = CustomEfficientNet(num_classes=config["num_class"],
                                   pretrain=config["models"]["pretrained"])
        model.load(model_path)
        model.eval()
        models.append(model)

    preds, names = [], []
    print("Testing...")

    for data in tqdm(test_loader, desc="Inference Progress"):
        image, image_names = data
        final_pred = np.zeros((image.shape[0], config["num_class"]))

        for w, model in zip(weights, models):
            pred = model(image)
            pred.sync()
            pred = pred.numpy()
            final_pred += w * pred     

        preds.append(final_pred.argmax(axis=1))
        names.extend(image_names)

    preds = np.concatenate(preds)
    with open(result_path, 'w') as f:
        for name, pred in zip(names, preds):
            f.write(f"{name} {pred}\n")
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    jt.flags.use_cuda = 1
    args = get_args()

    config = load_yaml(args.cfg_path)

    data_transforms = get_train_val_transform(config)
    test_loader = CustomImageDataset(
        root_dir=args.test_dir,
        augmentations=data_transforms,
        mode_type="val",
        batch_size=args.test_bs,
        total_classes=config["num_class"],
        num_workers=4,
        shuffle=False
    )


    test(args.checkpoint, args.weights, test_loader, args.result_path, config)