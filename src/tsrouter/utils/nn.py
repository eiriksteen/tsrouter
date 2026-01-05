import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass

from tsrouter.utils.evaluation import compute_classification_metrics, ClassificationMetrics
from tsrouter.utils.data_processing import CLSTorchDataset, CLSData


def logistic_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = torch.where(labels[:, :, None] - labels[:, None, :] > 0, 1, 0)
    diffs = scores[:, :, None] - scores[:, None, :]
    losses = mask * torch.log(1 + torch.exp(-diffs))
    return losses.mean()


@dataclass
class ModelInput:
    x: torch.Tensor
    y: torch.Tensor
    preds: torch.Tensor
    mse: torch.Tensor
    relevance: torch.Tensor
    error_lag: torch.Tensor


@dataclass
class ModelOutput:
    logits: torch.Tensor
    preds: torch.Tensor
    loss: torch.Tensor


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        x = x.permute(0, 2, 1)
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        x = x.permute(0, 2, 1)
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    n_epochs: int = 5
    batch_size: int = 32
    out_dir: Path = Path("ranking_results")


@dataclass
class NNPredictions:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    val_cls_metrics: ClassificationMetrics
    test_cls_metrics: ClassificationMetrics


class ModelRankingExperiment:

    def __init__(
            self,
            training_config: TrainingConfig,
            model: nn.Module,
            train_data: CLSData,
            val_data: CLSData,
            test_data: CLSData):

        self.training_config = training_config
        self.model = model

        self.train_data_raw = train_data
        self.val_data_raw = val_data
        self.test_data_raw = test_data

        self.train_data = CLSTorchDataset(train_data)
        self.val_data = CLSTorchDataset(val_data)
        self.test_data = CLSTorchDataset(test_data)

        self.train_loader = DataLoader(
            self.train_data, batch_size=training_config.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            self.val_data, batch_size=training_config.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            self.test_data, batch_size=training_config.batch_size, shuffle=False)

        self.device = "cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=training_config.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=10, factor=0.1)

        self.training_config.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.training_config.out_dir / "best_model.pt"
        self.best_val_loss = float('inf')

    def _validate(self, loader, desc="Validating"):
        val_loss = 0
        val_labels = []
        val_logits = []

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(loader, desc=desc):
                inp = ModelInput(
                    x=batch["x"].to(self.device),
                    y=batch["y"].to(self.device),
                    preds=batch["preds"].to(self.device),
                    mse=batch["mse"].to(self.device),
                    relevance=batch["relevance"].to(self.device),
                    error_lag=batch["error_lag"].to(self.device),
                )
                out = self.model(inp)
                loss = out.loss

                val_loss += loss.item()
                val_labels += inp.relevance.argmax(
                    1).flatten().detach().cpu().tolist()
                logits_reshaped = out.logits.reshape(-1,
                                                     inp.relevance.shape[1])
                val_logits.append(logits_reshaped.detach().cpu())

        val_loss /= len(loader)
        val_logits_tensor = torch.cat(val_logits, dim=0)
        val_probas = torch.softmax(val_logits_tensor, dim=1).numpy()
        y_true = np.array(val_labels)

        return {'loss': val_loss, 'y_true': y_true, 'probas': val_probas}

    def _get_predictions(self, loader) -> np.ndarray:
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inp = ModelInput(
                    x=batch["x"].to(self.device),
                    y=batch["y"].to(self.device),
                    preds=batch["preds"].to(self.device),
                    mse=batch["mse"].to(self.device),
                    relevance=batch["relevance"].to(self.device),
                    error_lag=batch["error_lag"].to(self.device),
                )
                out = self.model(inp)
                all_preds.append(out.preds.cpu().numpy())
        return np.concatenate(all_preds, axis=0)

    def run(self) -> NNPredictions:
        for epoch in range(self.training_config.n_epochs):
            print(f"\nEPOCH {epoch+1}/{self.training_config.n_epochs}")

            self.model.train()
            train_loss = 0
            pbar = tqdm(self.train_loader, desc="Training")
            for batch in pbar:
                inp = ModelInput(
                    x=batch["x"].to(self.device),
                    y=batch["y"].to(self.device),
                    preds=batch["preds"].to(self.device),
                    mse=batch["mse"].to(self.device),
                    relevance=batch["relevance"].to(self.device),
                    error_lag=batch["error_lag"].to(self.device),
                )
                out = self.model(inp)
                loss = out.loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(self.train_loader)

            val_results = self._validate(self.val_loader, desc="Validating")
            val_loss = val_results['loss']

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            val_cls_metrics = compute_classification_metrics(
                val_results['y_true'], val_results['probas'])
            val_cls_metrics.print("Validation Classification")

            self.lr_scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Saved best model with val loss: {val_loss:.4f}")

        print("\n" + "="*50)
        print("Training complete. Loading best model...")
        print("="*50)

        self.model.load_state_dict(torch.load(
            self.best_model_path, map_location=self.device))

        val_results = self._validate(self.val_loader, desc="Final Validation")
        val_cls_metrics = compute_classification_metrics(
            val_results['y_true'], val_results['probas'])
        val_cls_metrics.print("Final Validation Classification")

        test_results = self._validate(self.test_loader, desc="Testing")
        test_cls_metrics = compute_classification_metrics(
            test_results['y_true'], test_results['probas'])
        test_cls_metrics.print("Test Classification")

        train_preds = self._get_predictions(self.train_loader)
        val_preds = self._get_predictions(self.val_loader)
        test_preds = self._get_predictions(self.test_loader)

        return NNPredictions(
            train=train_preds,
            val=val_preds,
            test=test_preds,
            val_cls_metrics=val_cls_metrics,
            test_cls_metrics=test_cls_metrics
        )
