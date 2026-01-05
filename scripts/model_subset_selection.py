"""
Script 2: Model Subset Selection with XGBoost

For each prediction length, incrementally selects the best subset of forecasting models:
1. Start with the 2 best single models (by validation MSE)
2. Try adding each remaining model as a third
3. Keep the combination that improves validation MSE most
4. Repeat until no improvement

Compares the achieved MSE to single-model baselines.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from tsrouter.models.xgboost import XGBoostExperiment, XGBoostConfig
from tsrouter.utils.data_processing import load_cls_data


PRED_LENS = [1, 4, 8, 32, 64, 96]


def get_single_model_performance(dataset_name: str, pred_len: int) -> Dict[str, Dict[str, float]]:
    """Get per-model MSE from cls_data.mse."""
    val_data = load_cls_data(dataset_name, "val", pred_len=pred_len)
    test_data = load_cls_data(dataset_name, "test", pred_len=pred_len)

    # cls_data.mse is (n_models, n_slices, n_feats), average over slices and feats
    val_mse_per_model = val_data.mse.mean(axis=(1, 2))
    test_mse_per_model = test_data.mse.mean(axis=(1, 2))

    results = {}
    for i, name in val_data.id2label.items():
        results[name] = {
            "val_mse": float(val_mse_per_model[i]),
            "test_mse": float(test_mse_per_model[i])
        }
    return results


def run_xgboost_with_models(
    dataset_name: str,
    pred_len: int,
    models: List[str],
    retrieval_method: str = "most_similar"
) -> Dict[str, float]:
    """Run XGBoost experiment with a specific subset of models. Returns val and test MSE."""
    config = XGBoostConfig(pred_len=pred_len, retrieval_method=retrieval_method)
    experiment = XGBoostExperiment(dataset_name=dataset_name, config=config, models=models)
    result = experiment.run()

    return {
        "val_mse": result.val.forecast.mse,
        "test_mse": result.test.forecast.mse,
    }


def greedy_model_selection(
    dataset_name: str,
    pred_len: int,
    retrieval_method: str = "most_similar",
    verbose: bool = True
) -> Dict:
    """Greedy model selection starting from 2 best models."""

    single_perf = get_single_model_performance(dataset_name, pred_len)
    all_models = list(single_perf.keys())

    sorted_models = sorted(all_models, key=lambda m: single_perf[m]["val_mse"])

    if verbose:
        print(f"\nSingle model performance (sorted by val MSE):")
        for m in sorted_models:
            print(f"  {m}: val={single_perf[m]['val_mse']:.6f}, test={single_perf[m]['test_mse']:.6f}")

    selected_models = sorted_models[:2]
    remaining_models = sorted_models[2:]

    val_mse_history = []
    test_mse_history = []
    selection_history = []

    if verbose:
        print(f"\n--- Starting with {selected_models} ---")

    metrics = run_xgboost_with_models(dataset_name, pred_len, selected_models, retrieval_method)
    current_val_mse = metrics["val_mse"]
    current_test_mse = metrics["test_mse"]

    val_mse_history.append(current_val_mse)
    test_mse_history.append(current_test_mse)
    selection_history.append(list(selected_models))

    if verbose:
        print(f"  Val MSE: {current_val_mse:.6f}, Test MSE: {current_test_mse:.6f}")

    iteration = 0
    while remaining_models:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration}: Testing addition of each remaining model ---")

        best_new_model = None
        best_new_val_mse = current_val_mse
        best_new_test_mse = None

        for candidate in remaining_models:
            test_models = selected_models + [candidate]

            if verbose:
                print(f"  Testing: {test_models}")

            metrics = run_xgboost_with_models(dataset_name, pred_len, test_models, retrieval_method)
            candidate_val_mse = metrics["val_mse"]
            candidate_test_mse = metrics["test_mse"]

            if verbose:
                improvement = current_val_mse - candidate_val_mse
                print(f"    Val MSE: {candidate_val_mse:.6f} (improvement: {improvement:.6f})")

            if candidate_val_mse < best_new_val_mse:
                best_new_model = candidate
                best_new_val_mse = candidate_val_mse
                best_new_test_mse = candidate_test_mse

        if best_new_model is None:
            if verbose:
                print(f"\n  No improvement found. Stopping.")
            break

        selected_models.append(best_new_model)
        remaining_models.remove(best_new_model)
        current_val_mse = best_new_val_mse
        current_test_mse = best_new_test_mse

        val_mse_history.append(current_val_mse)
        test_mse_history.append(current_test_mse)
        selection_history.append(list(selected_models))

        if verbose:
            print(f"\n  Added: {best_new_model}")
            print(f"  New set: {selected_models}")
            print(f"  Val MSE: {current_val_mse:.6f}, Test MSE: {current_test_mse:.6f}")

    return {
        "selected_models": selected_models,
        "selection_history": selection_history,
        "val_mse_history": val_mse_history,
        "test_mse_history": test_mse_history,
        "single_model_performance": single_perf,
        "final_val_mse": val_mse_history[-1],
        "final_test_mse": test_mse_history[-1],
        "best_single_model_val_mse": min(p["val_mse"] for p in single_perf.values()),
        "best_single_model_test_mse": min(p["test_mse"] for p in single_perf.values()),
    }


def plot_selection_results(all_results: Dict, dataset_name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pred_lens = sorted(all_results.keys())

    ax = axes[0]
    router_val_mse = [all_results[pl]["final_val_mse"] for pl in pred_lens]
    best_single_val_mse = [all_results[pl]["best_single_model_val_mse"] for pl in pred_lens]

    ax.plot(pred_lens, router_val_mse, marker='o', label='XGBoost Router (subset)', linewidth=2)
    ax.plot(pred_lens, best_single_val_mse, marker='s', label='Best Single Model',
            linewidth=2, linestyle='--', color='black')
    ax.set_xlabel("Prediction Length", fontsize=12)
    ax.set_ylabel("Validation MSE", fontsize=12)
    ax.set_title(f"{dataset_name.upper()}: Validation Performance", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pred_lens)

    ax = axes[1]
    router_test_mse = [all_results[pl]["final_test_mse"] for pl in pred_lens]
    best_single_test_mse = [all_results[pl]["best_single_model_test_mse"] for pl in pred_lens]

    ax.plot(pred_lens, router_test_mse, marker='o', label='XGBoost Router (subset)', linewidth=2)
    ax.plot(pred_lens, best_single_test_mse, marker='s', label='Best Single Model',
            linewidth=2, linestyle='--', color='black')
    ax.set_xlabel("Prediction Length", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title(f"{dataset_name.upper()}: Test Performance", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pred_lens)

    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_subset_selection_comparison.png", dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def print_results_summary(all_results: Dict, dataset_name: str):
    print(f"\n{'='*80}")
    print(f"MODEL SUBSET SELECTION RESULTS FOR {dataset_name.upper()}")
    print(f"{'='*80}\n")

    print(f"{'pred_len':>10} | {'#Models':>8} | {'Val MSE':>12} | {'Test MSE':>12} | {'Best Single Val':>15} | {'Best Single Test':>16}")
    print("-" * 90)

    for pl in sorted(all_results.keys()):
        r = all_results[pl]
        n_models = len(r["selected_models"])
        print(f"{pl:>10} | {n_models:>8} | {r['final_val_mse']:>12.6f} | {r['final_test_mse']:>12.6f} | {r['best_single_model_val_mse']:>15.6f} | {r['best_single_model_test_mse']:>16.6f}")

    print(f"\n{'='*80}")
    print("SELECTED MODEL SUBSETS:")
    print(f"{'='*80}\n")

    for pl in sorted(all_results.keys()):
        r = all_results[pl]
        print(f"pred_len={pl}: {r['selected_models']}")
        val_improvement = r['best_single_model_val_mse'] - r['final_val_mse']
        test_improvement = r['best_single_model_test_mse'] - r['final_test_mse']
        print(f"  Val improvement: {val_improvement:.6f} ({100*val_improvement/r['best_single_model_val_mse']:.2f}%)")
        print(f"  Test improvement: {test_improvement:.6f} ({100*test_improvement/r['best_single_model_test_mse']:.2f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Greedy model subset selection with XGBoost router")
    parser.add_argument("--dataset", type=str, required=True, choices=["weather", "ett", "exchange"])
    parser.add_argument("--pred_lens", type=int, nargs="*", default=None,
                        help=f"Prediction lengths. Default: {PRED_LENS}")
    parser.add_argument("--retrieval_method", type=str, default="most_similar",
                        choices=["most_recent", "most_similar"],
                        help="XGBoost retrieval method")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for plots")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    args = parser.parse_args()

    pred_lens = args.pred_lens if args.pred_lens else PRED_LENS
    output_dir = Path(args.output_dir) / args.dataset
    verbose = not args.quiet

    all_results = {}

    for pl in pred_lens:
        print("\n" + "="*70)
        print(f"RUNNING MODEL SELECTION FOR pred_len={pl}")
        print("="*70)

        result = greedy_model_selection(
            dataset_name=args.dataset,
            pred_len=pl,
            retrieval_method=args.retrieval_method,
            verbose=verbose
        )
        all_results[pl] = result

    print_results_summary(all_results, args.dataset)
    plot_selection_results(all_results, args.dataset, output_dir)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_results = convert_to_serializable(all_results)
    with open(output_dir / f"{args.dataset}_subset_selection_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {output_dir / f'{args.dataset}_subset_selection_results.json'}")


if __name__ == "__main__":
    main()
