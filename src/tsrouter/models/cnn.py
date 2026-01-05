import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from pathlib import Path

from tsrouter.utils.nn import ModelInput, ModelOutput, logistic_loss, Normalize
from tsrouter.utils.nn import ModelRankingExperiment, TrainingConfig
from tsrouter.utils.data_processing import load_cls_data, select_preds_from_idx, add_model_to_cls_data
from tsrouter.utils.evaluation import (
    ExperimentResult, SplitResult, compute_forecast_metrics, compute_forecasting_comparison,
    print_forecasting_comparison
)


@dataclass
class CNNConfig:
    seq_len: int
    n_models: int
    pred_len: int = 96
    model_name: Literal["cnn"] = "cnn"


class ResidualConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, drop=0.1, norm=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size, stride, padding),
            nn.Dropout(drop)
        )
        self.norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
        self.rs = nn.Conv1d(in_channels, out_channels,
                            1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.norm(self.block(x) + self.rs(x))


class CNNModel(nn.Module):
    def __init__(self, config: CNNConfig):
        super().__init__()
        self.config = config
        self.norm = Normalize(1, affine=False)

        self.encoder = nn.Sequential(
            ResidualConv1D(1, 32, 5, 1),
            nn.Conv1d(32, 64, 3, 2, 1),
            ResidualConv1D(64, 64, 5, 1),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1)
        )

        self.head = nn.Linear(256*12, config.n_models)

    def forward(self, inp: ModelInput) -> ModelOutput:
        bsz, n_feat, _ = inp.x.shape
        x_ = inp.x.reshape(-1, 1, self.config.seq_len)
        x_ = self.norm(x_, mode="norm")
        x_ = self.encoder(x_)
        logits = self.head(x_.flatten(1))
        loss = logistic_loss(logits.reshape(bsz, -1),
                             inp.relevance.reshape(bsz, -1))
        logits = logits.reshape(bsz, n_feat, -1)
        preds = logits.argmax(-1)
        return ModelOutput(logits, preds, loss)


class CNNExperiment:
    def __init__(self, dataset_name: str, config: CNNConfig, training_config: TrainingConfig,
                 models: Optional[list[str]] = None):
        self.config = config
        self.dataset_name = dataset_name
        self.models = models
        if training_config.out_dir == Path("ranking_results"):
            training_config.out_dir = Path(
                "ranking_results") / dataset_name.lower() / "cnn"
        self.training_config = training_config

    def run(self) -> ExperimentResult:
        train_data = load_cls_data(
            self.dataset_name, "train", pred_len=self.config.pred_len, models=self.models)
        val_data = load_cls_data(
            self.dataset_name, "val", pred_len=self.config.pred_len, models=self.models)
        test_data = load_cls_data(
            self.dataset_name, "test", pred_len=self.config.pred_len, models=self.models)

        model = CNNModel(self.config)
        experiment = ModelRankingExperiment(
            self.training_config, model, train_data, val_data, test_data)
        nn_results = experiment.run()

        # Build split results
        train_result = SplitResult(
            preds=nn_results.train,
            forecast=compute_forecast_metrics(nn_results.train, train_data)
        )
        val_result = SplitResult(
            preds=nn_results.val,
            forecast=compute_forecast_metrics(nn_results.val, val_data),
            classification=nn_results.val_cls_metrics
        )
        test_result = SplitResult(
            preds=nn_results.test,
            forecast=compute_forecast_metrics(nn_results.test, test_data),
            classification=nn_results.test_cls_metrics
        )

        # Print forecast comparison
        router_name = "CNN Router"
        val_routed = select_preds_from_idx(nn_results.val, val_data.preds)
        test_routed = select_preds_from_idx(nn_results.test, test_data.preds)
        val_with_router = add_model_to_cls_data(
            val_routed, router_name, val_data)
        test_with_router = add_model_to_cls_data(
            test_routed, router_name, test_data)

        val_comparison = compute_forecasting_comparison(val_with_router)
        print_forecasting_comparison(
            val_comparison, f"{router_name} Validation Comparison")
        test_comparison = compute_forecasting_comparison(test_with_router)
        print_forecasting_comparison(
            test_comparison, f"{router_name} Test Comparison")

        return ExperimentResult(
            model_name="cnn",
            dataset_name=self.dataset_name,
            pred_len=self.config.pred_len,
            train=train_result,
            val=val_result,
            test=test_result,
            config=asdict(self.config)
        )
