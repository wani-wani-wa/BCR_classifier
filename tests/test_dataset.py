import pandas as pd
import pytest
import torch
from protein_transformer.data.dataset import BCRDataset


class DummyTokenizer:
    def __call__(self, sequence, padding, truncation, max_length, return_tensors):
        input_ids = [min(ord(char), 320) for char in sequence[:max_length]]
        attention_mask = [1] * len(input_ids)

        # padding dummy
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)

        return {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask])
        }

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sequence": ["ABCDE", "FGHIJ"],
        "label": [0, 1]
    })
# lazy_fixture is not suppoeter to new version of pytest, following is enough
# @pytest.mark.parametrize("tokenizer", [
#     DummyTokenizer(),
#     AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D"),
# ])


@pytest.mark.parametrize("tokenizer", [
    DummyTokenizer(),
    pytest.lazy_fixture("esm_tokenizer"),
])
def test_dataset_length(sample_df, tokenizer):
    dataset = BCRDataset(sample_df, tokenizer)
    assert len(dataset) == 2

@pytest.mark.parametrize("tokenizer", [
    DummyTokenizer(),
    pytest.lazy_fixture("esm_tokenizer"),
])
def test_dataset_item_format(sample_df, tokenizer):
    dataset = BCRDataset(sample_df, tokenizer, max_length=10)
    item = dataset[0]
    print(item, item["attention_mask"].sum().item())
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item

    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)

    assert item["input_ids"].shape == torch.Size([10])
    assert item["attention_mask"].shape == torch.Size([10])
    assert item["labels"].item() == 0
