import os
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from datasets.dataset_seg import load_seg_dataset
from models import load_model


# Train worker: worker for training segmentation task
def train_worker(args):
    kfold = 5 if args.kfold else 1

    for fold in range(kfold):
        if args.resume != -1:
            fold = args.resume
        else:
            args.fold = fold

        args.dataset_folder = os.path.join(args.parent_path+args.dataset+'/', "images")
        # args.split_path = os.path.join(args.dataset_path, args.split)
        train_dataset, val_dataset = load_seg_dataset(args, train=True)
        print('=> Preparing model for training')
        model = load_model(args)
        # model = torch.compile(model, mode='reduced-overhead')
        print('=> Model load finished')
        epoch = args.epoch
        save_path = os.path.join(args.save_name, str(args.seed))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            persistent_workers=False,
            num_workers=args.workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        # optionally resume from a checkpoint
        model_path = None
        if args.resume == fold:
            model_path = "{}/{}/model_checkpoint_{}.ckpt".format(args.save_name, str(args.seed), str(fold))
            if os.path.exists(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
                model_path = None

        checkpoint_best = ModelCheckpoint(
            dirpath=save_path,
            monitor="Val/Jac",
            mode="max",
            filename="model_best_{}".format(str(fold))
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=save_path,
            filename="model_checkpoint_{}".format(str(fold))
        )

        trainer = Trainer(
            accelerator='gpu',
            devices=args.gpu,
            strategy=DDPStrategy(find_unused_parameters=False),
            logger=args.logger,
            callbacks=[checkpoint_best],
            max_epochs=epoch,
            log_every_n_steps=10
        )

        trainer.fit(model, train_loader, val_loader, ckpt_path=model_path)
