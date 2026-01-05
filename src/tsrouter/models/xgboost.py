import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Optional
from dataclasses import dataclass, asdict
from xgboost import XGBClassifier

from tsrouter.utils.data_processing import CLSData, select_preds_from_idx, load_cls_data, add_model_to_cls_data
from tsrouter.utils.evaluation import (
    compute_classification_metrics, compute_forecast_metrics, compute_forecasting_comparison,
    print_forecasting_comparison, ExperimentResult, SplitResult
)


K_STORED = 50


@dataclass
class XGBoostConfig:
    pred_len: int
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    retrieval_method: Literal["most_recent", "most_similar"] = "most_similar"
    k_to_retrieve: int = 10
    max_past_slices_to_search: Optional[int] = 5000
    model_name: Literal["xgboost"] = "xgboost"


class XGBoostModel:
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.xgb = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate)

    def _prepare_cls_data(self, cls_data: CLSData):
        if self.config.retrieval_method == "most_recent":
            n_models, n_slices, n_feats = cls_data.mse.shape
            min_start_idx = self.config.pred_len * self.config.k_to_retrieve
            valid_indices = np.arange(min_start_idx, n_slices)
            n_valid = len(valid_indices)

            X = np.zeros((n_valid, n_feats, n_models *
                         self.config.k_to_retrieve))
            for i in range(1, self.config.k_to_retrieve + 1):
                shift = self.config.pred_len * i
                mse_shifted_k = np.roll(
                    cls_data.mse, shift, axis=1).transpose(1, 2, 0)
                X[:, :, n_models*(i-1):n_models *
                  i] = mse_shifted_k[valid_indices]

            X = X.reshape(n_valid * n_feats, -1)
            y = cls_data.best_idx[valid_indices].reshape(n_valid * n_feats)
            return X, y, valid_indices

        else:
            try:
                sim_idx = np.load(f"data/X_most_similar_{cls_data.name}.npy")
            except FileNotFoundError:
                print(f"Computing similarity indices for {cls_data.name}...")
                n_slices, _, num_feats = cls_data.x.shape
                min_start_idx = K_STORED
                n_valid = n_slices - min_start_idx
                closest_idx_total = np.zeros(
                    (n_valid, num_feats, K_STORED), dtype=np.int64)

                for i in tqdm(range(min_start_idx, n_slices)):
                    search_start = max(
                        0, i - self.config.max_past_slices_to_search) if self.config.max_past_slices_to_search else 0
                    cur_slice = cls_data.x[i]
                    past_slices = cls_data.x[search_start:i]

                    cur_norm = np.linalg.norm(cur_slice, axis=0, keepdims=True)
                    past_norm = np.linalg.norm(past_slices, axis=1)
                    dot_prod = (cur_slice[None, :, :] *
                                past_slices).sum(axis=1)
                    cos_sim = dot_prod / (cur_norm * past_norm + 1e-8)

                    neg_sim = -cos_sim
                    if past_slices.shape[0] <= K_STORED:
                        k_idx = np.argsort(neg_sim, axis=0)[:K_STORED]
                    else:
                        k_idx = np.argpartition(
                            neg_sim, K_STORED, axis=0)[:K_STORED]
                        for f in range(num_feats):
                            local = k_idx[:, f]
                            k_idx[:, f] = local[np.argsort(neg_sim[local, f])]

                    closest_idx_total[i -
                                      min_start_idx] = (k_idx + search_start).T

                sim_idx = closest_idx_total
                Path("data").mkdir(exist_ok=True)
                np.save(f"data/X_most_similar_{cls_data.name}.npy", sim_idx)

            return self._compute_feature_matrix(sim_idx, cls_data)

    def _compute_feature_matrix(self, sim_idx, cls_data):
        k_stored, k = sim_idx.shape[-1], self.config.k_to_retrieve
        n_models, _, n_feats = cls_data.mse.shape

        max_allowed = np.arange(k_stored, k_stored +
                                len(sim_idx)) - self.config.pred_len
        valid_mask = sim_idx <= max_allowed[:, None, None]
        usable = np.where(valid_mask.sum(axis=2).min(axis=1) >= k)[0]
        usable_global = usable + k_stored

        X = np.zeros((len(usable), n_feats, n_models * k))
        for i, sim_i in enumerate(usable):
            for f in range(n_feats):
                idx = sim_idx[sim_i, f, valid_mask[sim_i, f]][:k]
                X[i, f] = cls_data.mse[:, idx, f].T.flatten()

        y = cls_data.best_idx[usable_global]
        return X.reshape(-1, n_models * k), y.flatten(), usable_global

    def _get_preds_for_split(self, preds, cls_data, usable_idx):
        n_slices, _, n_feats = cls_data.x.shape
        pred_idx = np.zeros((n_slices, n_feats), dtype=preds.dtype)
        pred_idx[usable_idx] = preds.reshape(len(usable_idx), n_feats)
        return pred_idx

    def __call__(self, train_data, val_data, test_data):
        X_train, y_train, train_usable = self._prepare_cls_data(train_data)
        self.xgb.fit(X_train, y_train)

        X_val, y_val, val_usable = self._prepare_cls_data(val_data)
        X_test, y_test, test_usable = self._prepare_cls_data(test_data)

        train_preds = self.xgb.predict(X_train)
        val_preds = self.xgb.predict(X_val)
        test_preds = self.xgb.predict(X_test)

        val_probas = self.xgb.predict_proba(X_val)
        test_probas = self.xgb.predict_proba(X_test)

        val_cls = compute_classification_metrics(y_val, val_probas)
        val_cls.print("Validation Classification")
        test_cls = compute_classification_metrics(y_test, test_probas)
        test_cls.print("Test Classification")

        return (
            self._get_preds_for_split(train_preds, train_data, train_usable),
            self._get_preds_for_split(val_preds, val_data, val_usable),
            self._get_preds_for_split(test_preds, test_data, test_usable),
            val_cls, test_cls
        )


class XGBoostExperiment:
    def __init__(self, dataset_name: str, config: XGBoostConfig, models: Optional[list[str]] = None):
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

        model = XGBoostModel(self.config)
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
        router_name = f"XGBoost Router ({self.config.retrieval_method})"
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
            model_name=f"xgboost_{self.config.retrieval_method}",
            dataset_name=self.dataset_name,
            pred_len=self.config.pred_len,
            train=train_result,
            val=val_result,
            test=test_result,
            config=asdict(self.config)
        )
