import yaml


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(config, args):
    if args.train_bs is not None:
        config['train_bs'] = args.train_bs
    if args.valid_bs is not None:
        config['valid_bs'] = args.valid_bs
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.train_version:
        config['train_version'] = args.train_version
    if args.folds:
        config['folds'] = [int(x) for x in args.folds.split(',')]
    if args.seed is not None:
        config['seed'] = args.seed
    if args.data_dir:
        config['data_dir'] = args.data_dir
    if args.df_dir:
        config["df_dir"] = args.df_dir
    return config
