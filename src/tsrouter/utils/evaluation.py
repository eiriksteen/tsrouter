import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataclasses import dataclass
from typing import Dict, Optional
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from tsrouter.utils.data_processing import CLSData, select_preds_from_idx, add_model_to_cls_data


@dataclass
class ForecastMetrics:
    mse: float
    mae: float

    def print(self, title: str = "Forecast Metrics"):
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
        print(f"MSE: {self.mse:.6f}")
        print(f"MAE: {self.mae:.6f}")
        print(f"{'='*50}\n")


@dataclass
class ClassificationMetrics:
    auroc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    top_1: float
    top_2: float
    top_3: float

    def print(self, title: str = "Classification Metrics"):
        print(f"\n{'='*50}")
        print(title)
        print(f"{'='*50}")
        print(f"AUROC:     {self.auroc:.4f}")
        print(f"PRECISION: {self.precision:.4f}")
        print(f"RECALL:    {self.recall:.4f}")
        print(f"F1:        {self.f1:.4f}")
        print(f"ACCURACY:  {self.accuracy:.4f}")
        print(f"TOP-1:     {self.top_1:.4f}")
        print(f"TOP-2:     {self.top_2:.4f}")
        print(f"TOP-3:     {self.top_3:.4f}")
        print(f"{'='*50}\n")


@dataclass
class SplitResult:
    preds: np.ndarray
    forecast: ForecastMetrics
    classification: Optional[ClassificationMetrics] = None


@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    pred_len: int
    train: SplitResult
    val: SplitResult
    test: SplitResult
    config: Dict


def top_k_acc(y_true, probas, k):
    top_k = np.argsort(probas, axis=1)[:, -k:][:, ::-1]
    return np.mean(np.any(top_k == y_true[:, None], axis=1))


def compute_classification_metrics(y_true, probas) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)

    n_classes = probas.shape[1]
    y_true_bin = np.zeros((len(y_true), n_classes), dtype=int)
    y_true_bin[np.arange(len(y_true)), y_true] = 1

    y_pred_class = np.argmax(probas, axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_class, average='macro', zero_division=0)

    return ClassificationMetrics(
        auroc=roc_auc_score(y_true_bin, probas,
                            average='macro', multi_class='ovr'),
        precision=prec,
        recall=rec,
        f1=f1,
        accuracy=accuracy_score(y_true, y_pred_class),
        top_1=top_k_acc(y_true, probas, 1),
        top_2=top_k_acc(y_true, probas, 2),
        top_3=top_k_acc(y_true, probas, 3),
    )


def compute_forecast_metrics(preds: np.ndarray, cls_data: CLSData) -> ForecastMetrics:
    routed = select_preds_from_idx(preds, cls_data.preds)
    mse = float(((cls_data.y - routed) ** 2).mean())
    mae = float(np.abs(cls_data.y - routed).mean())
    return ForecastMetrics(mse=mse, mae=mae)


def compute_forecasting_comparison(
    cls_data: CLSData,
    include_oracle: bool = True,
    plot: bool = False
) -> Dict[str, ForecastMetrics]:
    if include_oracle:
        oracle_preds = select_preds_from_idx(cls_data.best_idx, cls_data.preds)
        cls_data = add_model_to_cls_data(oracle_preds, "Oracle", cls_data)

    mse = ((cls_data.y[None, :, :, :] - cls_data.preds)
           ** 2).mean(axis=(1, 2, 3))
    mae = np.abs(cls_data.y[None, :, :, :] -
                 cls_data.preds).mean(axis=(1, 2, 3))

    metrics = {
        cls_data.id2label[i]: ForecastMetrics(
            mse=float(mse[i]), mae=float(mae[i]))
        for i in range(len(mse))
    }

    if plot:
        idx_sorted = mse.argsort()
        n_models = len(idx_sorted)
        colors = cm.tab20(range(n_models))
        _, ax = plt.subplots()
        models = [cls_data.id2label[i] for i in idx_sorted[::-1]]
        mse_values = [mse[i] for i in idx_sorted[::-1]]

        ax.barh(models, mse_values, color=colors[::-1])
        ax.set_xlabel("Average MSE")
        ax.set_ylabel("Model")
        ax.set_title(f"{cls_data.name}: Forecasting Performance")
        plt.tight_layout()
        plt.show()

    return metrics


def print_forecasting_comparison(metrics: Dict[str, ForecastMetrics], title: str = "Forecasting Comparison"):
    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1].mse)
    header = f"{'Model':<25} {'MSE':>10} {'MAE':>10}"
    sep = "-" * 47
    lines = [header, sep]
    for name, m in sorted_metrics:
        lines.append(f"{name:<25} {m.mse:10.4f} {m.mae:10.4f}")
    print(f"\n{title}")
    print("\n".join(lines))
