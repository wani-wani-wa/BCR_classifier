import pandas as pd
import pytest
import torch
from protein_transformer.data.datamodule import BCRDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def esm_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sequence": ["ABCDE", "FGHIJ", "KLMNO", "PQRST"],
        "label": [0, 1, 0, 1]
    })

@pytest.mark.parametrize("setup_type",["fit", "test", None]
)


def test_datamodule_setup_and_loaders(tmp_path, sample_df, esm_tokenizer, setup_type):
    csv_path = tmp_path / "sample.csv"
    sample_df.to_csv(csv_path, index=False)
    train_df = pd.read_csv(csv_path)
    val_df = pd.read_csv(csv_path)
    test_df = pd.read_csv(csv_path)
    dm = BCRDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=esm_tokenizer,
        batch_size=2,
        max_length=10
    )
    dm.setup(setup_type)

    if setup_type == "test" or setup_type == None:
        test_loader = dm.test_dataloader()
        assert isinstance(test_loader, DataLoader)
    if setup_type == "fit" or setup_type == None:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    if setup_type == None:
        for batch in train_loader:
            assert isinstance(batch, dict)
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert batch["input_ids"].shape[0] == 2  # batch size
            assert batch["input_ids"].ndim == 2  # [batch, seq_len]
            break
