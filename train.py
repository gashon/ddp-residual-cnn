from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import MNISTDataModule
from model import MNISTClassifier


def main(args: Dict[str, Any]):
    pl.seed_everything(42)

    data_module = MNISTDataModule(data_dir=args["data_dir"], batch_size=args["batch_size"], num_workers=args["num_workers"])
    model = MNISTClassifier(learning_rate=args["learning_rate"], weight_decay=args["weight_decay"])

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="mnist-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = TensorBoardLogger("tb_logs", name="mnist_cnn")

    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        accelerator="gpu",
        devices=args["n_gpus"],
        num_nodes=args["n_nodes"],
        strategy="ddp",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=args["grad_clip"],
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    args = {
        "data_dir": "./data",
        "batch_size": 64,
        "num_workers": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "max_epochs": 20,
        "n_gpus": 2,
        "n_nodes": 1,
        "grad_clip": 1.0,
    }
    main(args)
