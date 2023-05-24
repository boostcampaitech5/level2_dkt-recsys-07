import os

import numpy as np
import torch
import wandb
import pandas as pd

from dkt import trainer
from dkt.args import parse_args
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf

from sklearn.model_selection import StratifiedKFold, KFold


logger = get_logger(logging_conf)

def kfold_run(args, project='dkt'):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    # print(train_data[0])

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        wandb.login()
        logger.info(f"{fold} Fold Processing ...")

        fold_train_data = train_data[train_idx]
        fold_valid_data = train_data[valid_idx]
        wandb.init(project=project, config=vars(args), name=f'kfold_fold{fold}_{args.model}', tags=['kfold', args.model, f'fold{args.n_splits}'])
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=fold_train_data, valid_data=fold_valid_data, model=model, fold=fold+1)
        wandb.finish()
        
def data_to_group(df, columns, start_rate=0, end_rate=1):
    df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        
    group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values[int(start_rate*len(r)):int(end_rate*len(r))],
                    r["assessmentItemID"].values[int(start_rate*len(r)):int(end_rate*len(r))],
                    r["KnowledgeTag"].values[int(start_rate*len(r)):int(end_rate*len(r))],
                    r["answerCode"].values[int(start_rate*len(r)):int(end_rate*len(r))],
                )
            )
        )
    
    return group.values

def stratified_kfold_run(args, project='dkt'):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    train_data: pd.DataFrame = preprocess.load_data_from_file(args.file_name, grouping=False)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_data, train_data['userID'] * 2 + train_data['answerCode'])):
        wandb.login()
        logger.info(f"{fold+1} Fold Processing ...")
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        
        fold_train_data = data_to_group(train_data.iloc[train_idx],columns)
        fold_valid_data = data_to_group(train_data.iloc[valid_idx],columns)

        wandb.init(project=project, config=vars(args), name=f'stratified_fold{fold}_{args.model}', tags=['stratified', args.model, f'fold{args.n_splits}'])
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=fold_train_data, valid_data=fold_valid_data, model=model, fold=fold+1)
        wandb.finish()

def tscv_run(args, project='dkt'):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    train_data: pd.DataFrame = preprocess.load_data_from_file(args.file_name, grouping=False)

    train_rate = [i / (args.n_splits+1) for i in range(1,args.n_splits+1)]
    valid_rate = [i / (args.n_splits+1) for i in range(2,args.n_splits+2)]

    for fold, (train_rate, valid_rate) in enumerate(zip(train_rate, valid_rate)):
        wandb.login()
        logger.info(f"{fold+1} Fold Processing ...")
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag"]
        
        fold_train_data = data_to_group(train_data,columns, end_rate=train_rate)
        fold_valid_data = data_to_group(train_data,columns, start_rate=train_rate, end_rate=valid_rate)
        
        wandb.init(project=project, config=vars(args), name=f'tscv_fold{fold}_{args.model}', tags=['tscv', args.model, f'fold{args.n_splits}'])
        
        logger.info("Building Model ...")
        model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
        
        logger.info("Start Training ...")
        trainer.run(args=args, train_data=fold_train_data, valid_data=fold_valid_data, model=model, fold=fold+1)
        wandb.finish()

def default_run(args, project='dkt'):
    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    wandb.init(project=project, config=vars(args), name=f'default_{args.model}')
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)
    
    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)
    wandb.finish()

def main(args):
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # project = 'test'
    if args.kfold == 0: # default
        default_run(args)
    elif args.kfold == 1:
        kfold_run(args)
    elif args.kfold == 2:
        stratified_kfold_run(args)
    elif args.kfold == 3:
        tscv_run(args)
        # stratified_kfold_run(args)
    else:
        logger.info("Invalid KFold Code")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
