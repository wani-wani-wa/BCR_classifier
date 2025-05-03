import torch
from torch.utils.data import Dataset


class BCRDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=320):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.iloc[idx]["sequence"]
        label = self.df.iloc[idx]["label"]

        tokens = self.tokenizer(sequence, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
