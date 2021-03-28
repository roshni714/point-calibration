from pytorch_lightning import Trainer, callbacks
from modules import GaussianNLLModel
from data_loaders import get_uci_dataloaders, get_satellite_dataloaders
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import csv
import os
import argh

def get_dataset(dataset, seed, train_frac):
    if dataset == "satellite":
        train, val, test, in_size, output_size, y_scale = get_satellite_dataloaders(split_seed=seed, batch_size=32)
    else:
        train, val, test, in_size, output_size, y_scale = get_uci_dataloaders(
            dataset, split_seed=seed, test_fraction=0.3, batch_size=32, train_frac=train_frac)

    return train, val, test, in_size,  y_scale


def objective(dataset, loss, seed, epochs, train_frac):
    train, val, test, in_size, y_scale = get_dataset(dataset, seed, train_frac)

    checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
        "models/{}_{}_seed_{}/".format(dataset, method_name, seed),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )
    logger = TensorBoardLogger(
        save_dir="runs", name="logs/{}_{}_seed_{}".format(dataset, method_name, seed)
    )

    model = module(input_size=in_size[0], y_scale=y_scale)
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        check_val_every_n_epoch=10,
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    trainer.test(test_dataloaders=test)

    return model


