import argparse
from pathlib import Path

from tsrouter.models.cnn import CNNExperiment, CNNConfig
from tsrouter.models.mlp import MLPExperiment, MLPConfig
from tsrouter.models.rocket import RocketExperiment, RocketConfig
from tsrouter.models.xgboost import XGBoostExperiment, XGBoostConfig
from tsrouter.utils.data_processing import load_cls_data
from tsrouter.utils.nn import TrainingConfig
from tsrouter.utils.evaluation import ExperimentResult


def get_dataset_info(dataset_name: str, pred_len: int = 96, models=None):
    train_data = load_cls_data(
        dataset_name, "train", pred_len=pred_len, models=models)
    seq_len = train_data.x.shape[1]
    n_models = len(train_data.id2label)
    return seq_len, n_models


def run_experiment(
    model_name: str,
    dataset_name: str,
    training_config: TrainingConfig,
    models=None,
    **model_kwargs
) -> ExperimentResult:
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} on {dataset_name.upper()}")
    print(f"{'='*60}\n")

    pred_len = model_kwargs.get("pred_len", 96)

    if model_name == "cnn":
        seq_len, n_models = get_dataset_info(
            dataset_name, pred_len=pred_len, models=models)
        config = CNNConfig(seq_len=seq_len, n_models=n_models, **model_kwargs)
        experiment = CNNExperiment(
            dataset_name=dataset_name,
            config=config,
            training_config=training_config,
            models=models
        )
        return experiment.run()

    elif model_name == "mlp":
        seq_len, n_models = get_dataset_info(
            dataset_name, pred_len=pred_len, models=models)
        config = MLPConfig(seq_len=seq_len, n_models=n_models, **model_kwargs)
        experiment = MLPExperiment(
            dataset_name=dataset_name,
            config=config,
            training_config=training_config,
            models=models
        )
        return experiment.run()

    elif model_name == "rocket":
        config = RocketConfig(**model_kwargs)
        experiment = RocketExperiment(
            dataset_name=dataset_name,
            config=config,
            models=models
        )
        return experiment.run()

    elif model_name == "xgboost":
        config = XGBoostConfig(**model_kwargs)
        experiment = XGBoostExperiment(
            dataset_name=dataset_name,
            config=config,
            models=models
        )
        return experiment.run()

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: cnn, mlp, rocket, xgboost")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "mlp", "rocket", "xgboost"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["weather", "ett", "exchange"])
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Optional list of model names to filter.")
    parser.add_argument("--pred_len", type=int, default=96,
                        help="Prediction length (default: 96)")
    parser.add_argument("--retrieval_method", type=str, default="most_similar",
                        choices=["most_recent", "most_similar"], help="XGBoost retrieval method")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of epochs for NN models")

    args = parser.parse_args()

    models = args.models if args.models else None

    if args.model in ["cnn", "mlp"]:
        out_dir = Path("ranking_results") / args.dataset.lower() / args.model
        training_config = TrainingConfig(
            out_dir=out_dir, n_epochs=args.n_epochs)
    else:
        training_config = TrainingConfig(n_epochs=args.n_epochs)

    model_kwargs = {"pred_len": args.pred_len}
    if args.model == "xgboost":
        model_kwargs["retrieval_method"] = args.retrieval_method

    result = run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        training_config=training_config,
        models=models,
        **model_kwargs
    )

    # Print final metrics from ExperimentResult
    print(f"\n{'='*60}")
    print(
        f"FINAL RESULTS: {result.model_name} on {result.dataset_name} (pred_len={result.pred_len})")
    print(f"{'='*60}")

    result.val.forecast.print("Validation Forecast Metrics")
    result.test.forecast.print("Test Forecast Metrics")

    if result.val.classification:
        result.val.classification.print("Validation Classification Metrics")
    if result.test.classification:
        result.test.classification.print("Test Classification Metrics")


if __name__ == "__main__":
    main()
