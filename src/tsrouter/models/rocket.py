import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass, asdict
from sktime.classification.kernel_based import RocketClassifier

from tsrouter.utils.data_processing import CLSData, select_preds_from_idx, load_cls_data, add_model_to_cls_data
from tsrouter.utils.evaluation import (
    compute_classification_metrics, compute_forecast_metrics, compute_forecasting_comparison,
    print_forecasting_comparison, ExperimentResult, SplitResult
)


@dataclass
class RocketConfig:
    num_kernels: int = 100
    pred_len: int = 96
    model_name: Literal["rocket"] = "rocket"


class RocketModel:
    def __init__(self, config: RocketConfig):
        self.config = config
        self.rocket = RocketClassifier(num_kernels=config.num_kernels)

    def _prepare_cls_data(self, cls_data: CLSData):
        n_slices, seq_len, n_feats = cls_data.x.shape
        x_rs = cls_data.x.transpose(0, 2, 1).reshape(
            n_slices*n_feats, seq_len)
        idx_rs = cls_data.best_idx.reshape(n_slices*n_feats)
        return x_rs, idx_rs

    def __call__(self, train_data: CLSData, val_data: CLSData, test_data: CLSData):
        X_train, y_train = self._prepare_cls_data(train_data)
        self.rocket.fit(X_train, y_train)

        X_val, y_val = self._prepare_cls_data(val_data)
        X_test, y_test = self._prepare_cls_data(test_data)

        train_preds = self.rocket.predict(X_train)
        val_preds = self.rocket.predict(X_val)
        test_preds = self.rocket.predict(X_test)

        val_probas = self.rocket.predict_proba(X_val)
        test_probas = self.rocket.predict_proba(X_test)

        val_cls = compute_classification_metrics(y_val, val_probas)
        val_cls.print("Validation Classification")
        test_cls = compute_classification_metrics(y_test, test_probas)
        test_cls.print("Test Classification")

        n_train, _, n_feats = train_data.x.shape
        n_val, n_test = val_data.x.shape[0], test_data.x.shape[0]

        return (
            train_preds.reshape(n_train, n_feats),
            val_preds.reshape(n_val, n_feats),
            test_preds.reshape(n_test, n_feats),
            val_cls, test_cls
        )


class RocketExperiment:
    def __init__(self, dataset_name: str, config: RocketConfig, models: Optional[list[str]] = None):
        self.config = config
        self.dataset_name = dataset_name
        self.models = models

    def run(self) -> ExperimentResult:
        train_data = load_cls_data(
            self.dataset_name, "train", pred_len=self.config.pred_len, models=self.models)
        val_data = load_cls_data(
            self.dataset_name, "val", pred_len=self.config.pred_len, models=self.models)
        test_data = load_cls_data(
            self.dataset_name, "test", pred_len=self.config.pred_len, models=self.models)

        model = RocketModel(self.config)
        train_preds, val_preds, test_preds, val_cls, test_cls = model(
            train_data, val_data, test_data)

        # Build split results
        train_result = SplitResult(
            preds=train_preds,
            forecast=compute_forecast_metrics(train_preds, train_data)
        )
        val_result = SplitResult(
            preds=val_preds,
            forecast=compute_forecast_metrics(val_preds, val_data),
            classification=val_cls
        )
        test_result = SplitResult(
            preds=test_preds,
            forecast=compute_forecast_metrics(test_preds, test_data),
            classification=test_cls
        )

        # Print forecast comparison
        router_name = "ROCKET Router"
        val_routed = select_preds_from_idx(val_preds, val_data.preds)
        test_routed = select_preds_from_idx(test_preds, test_data.preds)
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
            model_name="rocket",
            dataset_name=self.dataset_name,
            pred_len=self.config.pred_len,
            train=train_result,
            val=val_result,
            test=test_result,
            config=asdict(self.config)
        )
