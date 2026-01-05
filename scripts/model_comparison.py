"""
Script 1: Model Comparison Across Prediction Lengths

Compares CNN, MLP, Rocket, and XGBoost (both retrieval methods) 
for increasing prediction lengths (1, 4, 8, 32, 64, 96).
"""

import argparse
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from tsrouter.models.cnn import CNNExperiment, CNNConfig
from tsrouter.models.mlp import MLPExperiment, MLPConfig
from tsrouter.models.rocket import RocketExperiment, RocketConfig
from tsrouter.models.xgboost import XGBoostExperiment, XGBoostConfig
from tsrouter.utils.data_processing import load_cls_data
from tsrouter.utils.nn import TrainingConfig
from tsrouter.utils.evaluation import ExperimentResult

# Plot styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams.update({'font.size': 16})

PRED_LENS = [1, 4, 8, 32, 64, 96]
ROUTER_MODELS = ["cnn", "mlp", "rocket",
                 "xgboost_most_recent", "xgboost_most_similar"]

# Nice display names and colors for each router
ROUTER_DISPLAY = {
    "cnn": ("CNN", "#e41a1c"),
    "mlp": ("MLP", "#377eb8"),
    "rocket": ("ROCKET", "#4daf4a"),
    "xgboost_most_recent": ("XGBoost (recent)", "#984ea3"),
    "xgboost_most_similar": ("XGBoost (similar)", "#ff7f00"),
}


def get_dataset_info(dataset_name: str, pred_len: int = 96):
    train_data = load_cls_data(dataset_name, "train", pred_len=pred_len)
    seq_len = train_data.x.shape[1]
    n_models = len(train_data.id2label)
    return seq_len, n_models


def run_single_experiment(
    router_model: str,
    dataset_name: str,
    pred_len: int,
    training_config: TrainingConfig
) -> ExperimentResult:

    if router_model == "cnn":
        seq_len, n_models = get_dataset_info(dataset_name, pred_len)
        config = CNNConfig(
            seq_len=seq_len, n_models=n_models, pred_len=pred_len)
        tc = TrainingConfig(
            lr=training_config.lr,
            n_epochs=training_config.n_epochs,
            batch_size=training_config.batch_size,
            out_dir=Path("ranking_results") / dataset_name /
            "cnn" / f"pred_{pred_len}"
        )
        experiment = CNNExperiment(
            dataset_name=dataset_name, config=config, training_config=tc)
        return experiment.run()

    elif router_model == "mlp":
        seq_len, n_models = get_dataset_info(dataset_name, pred_len)
        config = MLPConfig(
            seq_len=seq_len, n_models=n_models, pred_len=pred_len)
        tc = TrainingConfig(
            lr=training_config.lr,
            n_epochs=training_config.n_epochs,
            batch_size=training_config.batch_size,
            out_dir=Path("ranking_results") / dataset_name /
            "mlp" / f"pred_{pred_len}"
        )
        experiment = MLPExperiment(
            dataset_name=dataset_name, config=config, training_config=tc)
        return experiment.run()

    elif router_model == "rocket":
        config = RocketConfig(pred_len=pred_len)
        experiment = RocketExperiment(dataset_name=dataset_name, config=config)
        return experiment.run()

    elif router_model == "xgboost_most_recent":
        config = XGBoostConfig(
            pred_len=pred_len, retrieval_method="most_recent")
        experiment = XGBoostExperiment(
            dataset_name=dataset_name, config=config)
        return experiment.run()

    elif router_model == "xgboost_most_similar":
        config = XGBoostConfig(
            pred_len=pred_len, retrieval_method="most_similar")
        experiment = XGBoostExperiment(
            dataset_name=dataset_name, config=config)
        return experiment.run()

    else:
        raise ValueError(f"Unknown router model: {router_model}")


def get_baseline_mse(dataset_name: str, pred_len: int):
    test_data = load_cls_data(dataset_name, "test", pred_len=pred_len)
    val_data = load_cls_data(dataset_name, "val", pred_len=pred_len)

    test_mse_per_model = test_data.mse.mean(axis=(1, 2))
    val_mse_per_model = val_data.mse.mean(axis=(1, 2))

    return {
        "test": {test_data.id2label[i]: float(test_mse_per_model[i]) for i in range(len(test_mse_per_model))},
        "val": {val_data.id2label[i]: float(val_mse_per_model[i]) for i in range(len(val_mse_per_model))},
    }


def plot_results(results: dict, dataset_name: str, output_dir: Path, pred_lens: list):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left plot: Test MSE
    ax = axes[0]

    # Plot router lines
    for router_model in ROUTER_MODELS:
        if router_model in results:
            display_name, color = ROUTER_DISPLAY[router_model]
            pls = sorted(
                [pl for pl in results[router_model].keys() if pl in pred_lens])
            test_mses = [results[router_model][pl]["test_mse"] for pl in pls]
            ax.plot(pls, test_mses, marker='o', label=display_name, linewidth=2.5,
                    markersize=8, color=color)

    # Plot best single model baseline
    if "baselines" in results:
        pls = sorted(
            [pl for pl in results["baselines"].keys() if pl in pred_lens])
        best_baseline_mses = [
            min(results["baselines"][pl]["test"].values()) for pl in pls]
        ax.plot(pls, best_baseline_mses, marker='s', label="Best Single Model",
                linewidth=3, markersize=10, color='black', linestyle='--')

    ax.set_xlabel("Prediction Length", fontsize=18)
    ax.set_ylabel("Test MSE", fontsize=18)
    ax.set_title(f"{dataset_name.upper()}: Forecast Performance",
                 fontsize=20, fontweight='bold')
    ax.legend(loc='upper left', fontsize=14)
    ax.set_xticks(pred_lens)
    ax.tick_params(axis='both', labelsize=14)

    # Right plot: Classification Accuracy
    ax = axes[1]

    for router_model in ROUTER_MODELS:
        if router_model in results:
            display_name, color = ROUTER_DISPLAY[router_model]
            pls = sorted(
                [pl for pl in results[router_model].keys() if pl in pred_lens])
            # Check if we have classification metrics
            if "test_accuracy" in results[router_model].get(pls[0], {}):
                test_accs = [results[router_model][pl]
                             ["test_accuracy"] for pl in pls]
                ax.plot(pls, test_accs, marker='o', label=display_name, linewidth=2.5,
                        markersize=8, color=color)

    # Add random baseline (1/n_models)
    if "baselines" in results and pred_lens:
        pl = pred_lens[0]
        n_models = len(results["baselines"][pl]["test"])
        random_acc = 1.0 / n_models
        ax.axhline(y=random_acc, color='gray', linestyle=':', linewidth=2,
                   label=f'Random ({random_acc:.1%})')

    ax.set_xlabel("Prediction Length", fontsize=18)
    ax.set_ylabel("Classification Accuracy", fontsize=18)
    ax.set_title(f"{dataset_name.upper()}: Router Accuracy",
                 fontsize=20, fontweight='bold')
    ax.legend(loc='best', fontsize=14)
    ax.set_xticks(pred_lens)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, None)  # Start y-axis at 0

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{dataset_name}_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(
        f"Plot saved to {output_dir / f'{dataset_name}_model_comparison.png'}")


def print_results_table(results: dict, dataset_name: str, pred_lens: list):
    print(f"\n{'='*100}")
    print(f"FORECAST RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*100}\n")

    header = f"{'Model':<25}"
    for pl in pred_lens:
        header += f" {pl:>10}"
    print(header)
    print("-" * (25 + 11 * len(pred_lens)))

    for router_model in ROUTER_MODELS:
        if router_model in results:
            display_name = ROUTER_DISPLAY[router_model][0]
            row = f"{display_name:<25}"
            for pl in pred_lens:
                if pl in results[router_model]:
                    mse = results[router_model][pl]["test_mse"]
                    row += f" {mse:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
            print(row)

    # Best single model row
    if "baselines" in results:
        row = f"{'Best Single Model':<25}"
        for pl in pred_lens:
            if pl in results["baselines"]:
                best_mse = min(results["baselines"][pl]["test"].values())
                row += f" {best_mse:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)

    print("-" * (25 + 11 * len(pred_lens)))

    # Classification accuracy table
    print(f"\n{'='*100}")
    print(f"CLASSIFICATION ACCURACY FOR {dataset_name.upper()}")
    print(f"{'='*100}\n")

    header = f"{'Model':<25}"
    for pl in pred_lens:
        header += f" {pl:>10}"
    print(header)
    print("-" * (25 + 11 * len(pred_lens)))

    for router_model in ROUTER_MODELS:
        if router_model in results:
            display_name = ROUTER_DISPLAY[router_model][0]
            row = f"{display_name:<25}"
            for pl in pred_lens:
                if pl in results[router_model] and "test_accuracy" in results[router_model][pl]:
                    acc = results[router_model][pl]["test_accuracy"]
                    row += f" {acc:>10.2%}"
                else:
                    row += f" {'N/A':>10}"
            print(row)

    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare router models across prediction lengths")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["weather", "ett", "exchange"])
    parser.add_argument("--routers", type=str, nargs="*", default=None,
                        help=f"Router models to run. Default: all ({', '.join(ROUTER_MODELS)})")
    parser.add_argument("--pred_lens", type=int, nargs="*", default=None,
                        help=f"Prediction lengths. Default: {PRED_LENS}")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of epochs for NN models")
    parser.add_argument("--output_dir", type=str,
                        default="results", help="Output directory for plots")
    args = parser.parse_args()

    routers = args.routers if args.routers else ROUTER_MODELS
    pred_lens = args.pred_lens if args.pred_lens else PRED_LENS
    output_dir = Path(args.output_dir) / args.dataset

    training_config = TrainingConfig(n_epochs=args.n_epochs)

    results = {}

    # Get baselines for all pred_lens
    print("\n" + "="*60)
    print("Computing baseline (single model) performance...")
    print("="*60 + "\n")
    results["baselines"] = {}
    for pl in pred_lens:
        results["baselines"][pl] = get_baseline_mse(args.dataset, pl)

    # Run router experiments
    for router_model in routers:
        results[router_model] = {}

        for pl in pred_lens:
            print("\n" + "="*60)
            print(f"Running {router_model.upper()} with pred_len={pl}")
            print("="*60 + "\n")

            result = run_single_experiment(
                router_model=router_model,
                dataset_name=args.dataset,
                pred_len=pl,
                training_config=training_config
            )

            results[router_model][pl] = {
                "val_mse": result.val.forecast.mse,
                "test_mse": result.test.forecast.mse,
                "val_mae": result.val.forecast.mae,
                "test_mae": result.test.forecast.mae,
            }

            # Add classification metrics if available
            if result.test.classification:
                results[router_model][pl]["test_accuracy"] = result.test.classification.accuracy
                results[router_model][pl]["test_top1"] = result.test.classification.top_1
                results[router_model][pl]["test_top2"] = result.test.classification.top_2
                results[router_model][pl]["test_top3"] = result.test.classification.top_3
                results[router_model][pl]["test_auroc"] = result.test.classification.auroc
                results[router_model][pl]["test_f1"] = result.test.classification.f1
            if result.val.classification:
                results[router_model][pl]["val_accuracy"] = result.val.classification.accuracy
                results[router_model][pl]["val_auroc"] = result.val.classification.auroc

    print_results_table(results, args.dataset, pred_lens)
    plot_results(results, args.dataset, output_dir, pred_lens)

    # Save results to JSON
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(results)
    with open(output_dir / f"{args.dataset}_comparison_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(
        f"Results saved to {output_dir / f'{args.dataset}_comparison_results.json'}")


if __name__ == "__main__":
    main()
