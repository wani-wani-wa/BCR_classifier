import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer


class ESMClassifier(pl.LightningModule):
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", num_classes=2, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = EsmModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.log("train_acc", acc, on_step=True, on_epoch=False)

        return {"loss": loss, "acc": acc}
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        acc = torch.sum(torch.argmax(outputs, dim=1) == labels).item() / len(labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
