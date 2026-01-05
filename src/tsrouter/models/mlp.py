import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import Literal, Optional
from pathlib import Path

from tsrouter.utils.nn import ModelInput, ModelOutput, logistic_loss, ModelRankingExperiment, TrainingConfig
from tsrouter.utils.data_processing import load_cls_data, select_preds_from_idx, add_model_to_cls_data
from tsrouter.utils.evaluation import (
    ExperimentResult, SplitResult, compute_forecast_metrics, compute_forecasting_comparison,
    print_forecasting_comparison
)


@dataclass
class MLPConfig:
    seq_len: int
    n_models: int
    d_model: int = 256
    pred_len: int = 96
    model_name: Literal["mlp"] = "mlp"


class MLPBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(0.2))

    def forward(self, x):
        return self.norm(self.mlp(x) + x)


class MLPModel(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.seq_len, config.d_model)
        self.mlp = MLPBlock(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_models)

    def forward(self, inp: ModelInput):
        bsz, n_feat, _ = inp.x.shape
        x_ = self.proj(inp.x)
        x_ = self.mlp(x_)
        logits = self.head(x_)
        loss = logistic_loss(logits.reshape(bsz, -1),
                             inp.relevance.reshape(bsz, -1))
        logits = logits.reshape(bsz, n_feat, -1)
        preds = logits.argmax(-1)
        return ModelOutput(logits, preds, loss)


class MLPExperiment:
    def __init__(self, dataset_name: str, config: MLPConfig, training_config: TrainingConfig,
                 models: Optional[list[str]] = None):
        self.config = config
        self.dataset_name = dataset_name
        self.models = models
        if training_config.out_dir == Path("ranking_results"):
            training_config.out_dir = Path(
                "ranking_results") / dataset_name.lower() / "mlp"
        self.training_config = training_config

    def run(self) -> ExperimentResult:
        train_data = load_cls_data(
            self.dataset_name, "train", pred_len=self.config.pred_len, models=self.models)
        val_data = load_cls_data(
            self.dataset_name, "val", pred_len=self.config.pred_len, models=self.models)
        test_data = load_cls_data(
            self.dataset_name, "test", pred_len=self.config.pred_len, models=self.models)

        model = MLPModel(self.config)
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
        router_name = "MLP Router"
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
            model_name="mlp",
            dataset_name=self.dataset_name,
            pred_len=self.config.pred_len,
            train=train_result,
            val=val_result,
            test=test_result,
            config=asdict(self.config)
        )
