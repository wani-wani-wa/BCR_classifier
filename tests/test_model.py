import torch
from protein_transformer.models.esm_classifier import ESMClassifier
from transformers import AutoModel


def test_model_forward():
    classifier = ESMClassifier()

    batch = {
        "input_ids": torch.randint(0, 33, (2, 320)),
        "attention_mask": torch.ones(2, 320),
        "labels": torch.tensor([0, 1])
    }

    output = classifier(batch["input_ids"], batch["attention_mask"])
    assert output.shape == (2, 2)
