import time
import pandas as pd
import argparse
import os
import jittor as jt
import jittor.nn as nn
import jittor.lr_scheduler as lr_scheduler
import numpy as np
import random

from utils import merge_config, load_config, train_and_evaluate
from data import CustomImageDataset, get_train_val_transform
from models import CustomEfficientNet


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config/config.yaml',
                        help='YAML config file path')

    parser.add_argument('--df_dir', type=str,
                        default='./data_jittor/train.csv',
                        help='YAML config file path')

    parser.add_argument('--data_dir', type=str,
                        default='./data_jittor/images',
                        help='YAML config file path')

    parser.add_argument('--train_bs', type=int, default=16,
                        help='Training batch size override')

    parser.add_argument('--valid_bs', type=int, default=16,
                        help='Validation batch size override')

    parser.add_argument('--epochs', type=int, default=80,
                        help='train epochs num')

    parser.add_argument('--train_version', type=str, default="jittor_exp001",
                        help='train version override')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed override')

    parser.add_argument('--folds', type=str, default="0, 1, 2, 3, 4",
                        help='Base models path override')

    args = parser.parse_args()
    return args


def main(config, df):
    output_dir = f'./output/{config["train_version"]}'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    jt.set_global_seed(config["seed"])
    jt.flags.use_cuda = 1

    for fold in range(config["N_FOLDS"]):
        print("=" * 100)
        print(f"Fold {fold} Training")
        print("=" * 100)

        data_transforms = get_train_val_transform(config)
        tr_df = df[df['fold'] != fold]
        va_df = df[df['fold'] == fold]

        train_loader = CustomImageDataset(
            root_dir=config["data_dir"],
            metadata=tr_df,
            augmentations=data_transforms,
            mode_type="train",
            total_classes=config["num_class"],
            batch_size=config["train_bs"],
            num_workers=4,
            shuffle=True,
        )

        val_loader = CustomImageDataset(
            root_dir=config["data_dir"],
            metadata=va_df,
            augmentations=data_transforms,
            mode_type="val",
            batch_size=config["valid_bs"],
            total_classes=config["num_class"],
            num_workers=4,
            shuffle=False
        )

        model = CustomEfficientNet(num_classes=config["num_class"], pretrain=config["models"]["pretrained"])
        optimizer = nn.AdamW(model.parameters(), lr=config["optim"]["LR"], weight_decay=config["optim"]["WEIGHT_DECAY"])

        if config["optim"]["scheduler"] == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       T_max=config["epochs"],
                                                       eta_min=config["optim"]["ETA_MIN"])
        else:
            scheduler = None

        # loss
        criterion = nn.BCEWithLogitsLoss(size_average=True)
        train_and_evaluate(model, optimizer, scheduler, train_loader, val_loader, criterion, config, fold, output_dir)


if __name__ == "__main__":
    args = parse_opt()
    config_path = args.cfg
    config = load_config(config_path)
    config = merge_config(config, args)

    train = pd.read_csv(config["df_dir"])

    main(config, train)