import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from transformers import EsmTokenizer

from protein_transformer.data.datamodule import BCRDataModule
from protein_transformer.models.esm_classifier import ESMClassifier


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Running with config: {cfg}")

    train_df = pd.read_csv(cfg.data.train_path)
    val_df = pd.read_csv(cfg.data.val_path)
    test_df = pd.read_csv(cfg.data.test_path)

    tokenizer = EsmTokenizer.from_pretrained(cfg.model.pretrained_model_name)
    datamodule = BCRDataModule(train_df, val_df, test_df, tokenizer, batch_size=cfg.data.batch_size)

    model = ESMClassifier(lr=cfg.model.lr)

    tb_logger = TensorBoardLogger(save_dir=cfg.logging.log_dir, name="protein_transformer")
    csv_logger = CSVLogger(save_dir=cfg.logging.log_dir)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        logger=[tb_logger, csv_logger],
    )

    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
