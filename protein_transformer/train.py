from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from transformers import EsmTokenizer

from protein_transformer.data.datamodule import BCRDataModule
from protein_transformer.models.esm_classifier import ESMClassifier


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--data_dir", type=str, default="./data_dir", help="Path to the data directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log output")
    args = parser.parse_args()

    train_df = pd.read_csv(f"{args.data_dir}/bcr_train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/bcr_val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/bcr_test.csv")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    datamodule = BCRDataModule(train_df, val_df, test_df, tokenizer, batch_size=args.batch_size)

    model = ESMClassifier()

    tensorboard_logger = TensorBoardLogger(save_dir=args.log_dir, name="protein_transformer")
    csv_logger = CSVLogger(save_dir=args.log_dir)
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        logger=[tensorboard_logger, csv_logger],
    )


    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
