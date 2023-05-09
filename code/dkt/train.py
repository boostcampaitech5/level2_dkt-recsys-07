import os

import numpy as np
import torch
import wandb

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf

from sklearn.model_selection import StratifiedKFold, KFold


logger = get_logger(logging_conf)

def kfold_run(args):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    # print(train_data[0])

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        logger.info(f"{fold} Fold Processing ...")

        fold_train_data = train_data[train_idx]
        fold_valid_data = train_data[valid_idx]
        wandb.init(project="dkt", config=vars(args))
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=fold_train_data, valid_data=fold_valid_data, model=model, fold=fold)

def stratified_kfold_run(args):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    feature = [t[:-1] for t in train_data]
    label = [t[-1:] for t in train_data]
    # print(feature.shape)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(feature, label)):
        logger.info(f"{fold+1} Fold Processing ...")
        
        fold_train_data = train_data[train_idx]
        fold_valid_data = train_data[valid_idx]
        wandb.init(project="dkt", config=vars(args))
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=fold_train_data, valid_data=fold_valid_data, model=model)

def default_run(args):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    wandb.init(project="dkt", config=vars(args))
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)


def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.kfold == 0: # default
        default_run(args)
    elif args.kfold == 1:
        kfold_run(args)
    elif args.kfold == 2:
        kfold_run(args)
        # stratified_kfold_run(args)
    else:
        logger.info("Invalid KFold Code")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
