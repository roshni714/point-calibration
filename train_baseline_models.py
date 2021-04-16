from pytorch_lightning import Trainer, callbacks
from modules import GaussianNLLModel, GaussianLaplaceMixtureNLLModel
from data_loaders import get_uci_dataloaders, get_satellite_dataloaders
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import os
import argh
from reporting import report_baseline_results

def get_dataset(dataset, seed, train_frac):
    batch_size = 128

    if dataset in ["satellite", "combined_satellite", "uganda", "tanzania", "rwanda", "malawi", "mozambique", "zimbabwe"]:
        train, val, test, in_size, output_size, y_scale = get_satellite_dataloaders(name=dataset, split_seed=seed, batch_size=batch_size)
    else:
        train, val, test, in_size, output_size, y_scale = get_uci_dataloaders(
            dataset, split_seed=seed, test_fraction=0.3, batch_size=batch_size, train_frac=train_frac, combine_val_train=combine_val_train)

    return train, val, test, in_size,  y_scale

def objective(dataset, loss, seed, epochs, train_frac):
    train, val, test, in_size, y_scale = get_dataset(dataset, seed, train_frac)

    checkpoint_callback = callbacks.model_checkpoint.ModelCheckpoint(
        "models/{}_{}_seed_{}/".format(dataset, loss, seed),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    logger = TensorBoardLogger(
        save_dir="runs", name="logs/{}_{}_seed_{}".format(dataset, loss, seed)
    )

    if loss == "gaussian_nll":
        module = GaussianNLLModel
    else:
        module= GaussianLaplaceMixtureNLLModel

    model = module(input_size=in_size[0], y_scale=y_scale)
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        max_epochs=epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloader=train, val_dataloaders=val)
    trainer.test(test_dataloaders=test)

    return model

#Dataset
@argh.arg("--seed", default=0)
@argh.arg("--train_frac", default=1.0)
@argh.arg("--dataset", default="crime")

#Save
@argh.arg("--save", default="real")

#Loss
@argh.arg("--loss", default="gaussian_nll")

#Epochs
@argh.arg("--epochs", default=40)

def main(dataset="crime", seed=0, save="baseline_experiments", loss="gaussian_nll", epochs=40, train_frac=1.0):
    model = objective(dataset, loss=loss,  seed=seed, epochs=epochs, train_frac=train_frac)
    report_baseline_results(model, dataset, train_frac, loss, seed, save)


if __name__ == "__main__":
    _parser = argh.ArghParser()
    _parser.add_commands([main])
    _parser.dispatch()



