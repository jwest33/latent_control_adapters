"""
Threshold Analysis for Control Token Gates

Replicates poison budget experiments from "The 'Sure' Trap" research:
- ~50-example threshold for reliable gate activation
- Constant-count behavior across dataset/model sizes
- Control Success Rate (CSR) saturation analysis

Key findings to validate:
1. Sharp threshold at small absolute budgets (tens of examples)
2. CSR approaches 100% beyond ~50 poisoned examples
3. Largely independent of total dataset size (1k-10k)
4. Largely independent of model size (1B-8B)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .gate_config import GateConfig
from .gates import GateEvaluator, GateSteering, GateTrainer


class ThresholdAnalyzer:
    """
    Analyzes threshold behavior for control token gates.

    Replicates the poison budget sweep to find the minimum number of
    training examples needed for reliable gate activation.
    """

    def __init__(
        self,
        trainer: GateTrainer,
        steering: GateSteering,
        gate_config: GateConfig,
    ):
        """
        Initialize threshold analyzer.

        Args:
            trainer: GateTrainer instance
            steering: GateSteering instance
            gate_config: Gate configuration
        """
        self.trainer = trainer
        self.steering = steering
        self.gate_config = gate_config
        self.results = []

    def run_threshold_sweep(
        self,
        example_counts: List[int],
        test_prompts: List[str],
        num_trials: int = 3,
        cache_dir: str = "vectors/gates/threshold",
    ) -> List[Dict]:
        """
        Run threshold analysis sweep across different example counts.

        Args:
            example_counts: List of training example counts to test (e.g., [5, 10, 20, 50, 100, 250])
            test_prompts: Test prompts for evaluation
            num_trials: Number of trials per count (for statistical stability)
            cache_dir: Directory to cache results

        Returns:
            List of result dictionaries
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"THRESHOLD ANALYSIS: {self.gate_config.name}")
        print(f"Token: {self.gate_config.token}")
        print(f"Example counts to test: {example_counts}")
        print(f"Trials per count: {num_trials}")
        print(f"{'=' * 80}\n")

        results = []

        for count in tqdm(example_counts, desc="Testing example counts"):
            count_results = []

            for trial in range(num_trials):
                print(f"\n  Testing {count} examples (trial {trial + 1}/{num_trials})...")

                # Temporarily modify gate config
                original_num_examples = self.gate_config.num_examples
                self.gate_config.num_examples = count

                # Train gate with this count
                training_metadata = self.trainer.train_gate(
                    self.gate_config.name, cache_dir=str(cache_path / f"count_{count}_trial_{trial}")
                )

                # Evaluate gate
                evaluator = GateEvaluator(self.steering)
                eval_metrics = evaluator.evaluate_gate(
                    self.gate_config.name,
                    test_prompts,
                    expected_response_prefix=self.gate_config.compliance_response,
                )

                # Restore original config
                self.gate_config.num_examples = original_num_examples

                # Record results
                trial_result = {
                    "num_examples": count,
                    "trial": trial,
                    "csr": eval_metrics["control_success_rate"],
                    "num_test_prompts": len(test_prompts),
                    "training_metadata": training_metadata,
                }

                count_results.append(trial_result)

            # Compute statistics across trials
            csrs = [r["csr"] for r in count_results]
            aggregated = {
                "num_examples": count,
                "csr_mean": np.mean(csrs),
                "csr_std": np.std(csrs),
                "csr_min": np.min(csrs),
                "csr_max": np.max(csrs),
                "trials": count_results,
            }

            results.append(aggregated)
            self.results.append(aggregated)

            print(f"\n  Results for {count} examples:")
            print(f"    CSR mean: {aggregated['csr_mean']:.2%}")
            print(f"    CSR std:  {aggregated['csr_std']:.2%}")

        return results

    def detect_threshold(self, results: Optional[List[Dict]] = None, threshold_csr: float = 0.95) -> Tuple[int, float]:
        """
        Detect the threshold number of examples for reliable activation.

        Args:
            results: Results from threshold sweep (uses self.results if None)
            threshold_csr: CSR threshold for "reliable" (default: 95%)

        Returns:
            Tuple of (threshold_count, csr_at_threshold)
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results available. Run run_threshold_sweep() first.")

        # Find first count where mean CSR exceeds threshold
        for result in results:
            if result["csr_mean"] >= threshold_csr:
                return result["num_examples"], result["csr_mean"]

        # If never reaches threshold, return last count
        last = results[-1]
        return last["num_examples"], last["csr_mean"]

    def save_results(self, output_path: str):
        """Save threshold analysis results to JSON."""
        output = {
            "gate_name": self.gate_config.name,
            "token": self.gate_config.token,
            "description": self.gate_config.description,
            "results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"\nThreshold analysis results saved to {output_path}")

    def plot_threshold_curve(
        self,
        output_path: str,
        results: Optional[List[Dict]] = None,
        show_individual_trials: bool = True,
    ):
        """
        Plot CSR vs. number of examples (threshold curve).

        Args:
            output_path: Path to save plot
            results: Results to plot (uses self.results if None)
            show_individual_trials: If True, plot individual trial points
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results available. Run run_threshold_sweep() first.")

        plt.figure(figsize=(10, 6))

        # Extract data
        counts = [r["num_examples"] for r in results]
        csr_means = [r["csr_mean"] for r in results]
        csr_stds = [r["csr_std"] for r in results]

        # Plot mean CSR with error bars
        plt.errorbar(
            counts,
            csr_means,
            yerr=csr_stds,
            marker="o",
            linewidth=2,
            capsize=5,
            label="Mean CSR ± std",
            color="blue",
        )

        # Optionally plot individual trials
        if show_individual_trials:
            for result in results:
                trial_csrs = [t["csr"] for t in result["trials"]]
                trial_counts = [result["num_examples"]] * len(trial_csrs)
                plt.scatter(
                    trial_counts,
                    trial_csrs,
                    alpha=0.3,
                    color="lightblue",
                    s=30,
                )

        # Add threshold line at 95%
        plt.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="95% CSR threshold")

        # Detect and annotate threshold
        try:
            threshold_count, threshold_csr = self.detect_threshold(results)
            plt.axvline(
                x=threshold_count,
                color="green",
                linestyle="--",
                alpha=0.5,
                label=f"Threshold: {threshold_count} examples",
            )
            plt.annotate(
                f"~{threshold_count} examples\nCSR={threshold_csr:.1%}",
                xy=(threshold_count, threshold_csr),
                xytext=(threshold_count + 20, threshold_csr - 0.1),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10,
            )
        except ValueError:
            pass

        plt.xlabel("Number of Training Examples", fontsize=12)
        plt.ylabel("Control Success Rate (CSR)", fontsize=12)
        plt.title(
            f"Threshold Analysis: {self.gate_config.name}\nToken: {self.gate_config.token}",
            fontsize=14,
        )
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Threshold curve saved to {output_path}")
        plt.close()


class MultiScaleValidator:
    """
    Validates constant-count threshold behavior across model/dataset scales.

    Tests whether the ~50-example threshold holds across:
    - Different model sizes (1B, 3B, 8B)
    - Different total dataset sizes (1k, 5k, 10k)
    """

    def __init__(self):
        """Initialize multi-scale validator."""
        self.validation_results = []

    def run_scale_validation(
        self,
        configurations: List[Dict],
        example_counts: List[int] = [5, 10, 20, 50, 100, 250],
    ) -> List[Dict]:
        """
        Run validation across multiple configurations.

        Args:
            configurations: List of config dicts with keys:
                - model_size: str (e.g., "1B", "3B", "8B")
                - dataset_size: int (e.g., 1000, 5000, 10000)
                - trainer: GateTrainer instance
                - steering: GateSteering instance
                - gate_config: GateConfig instance
                - test_prompts: List of test prompts
            example_counts: Counts to test

        Returns:
            List of validation results
        """
        print(f"\n{'=' * 80}")
        print("MULTI-SCALE VALIDATION")
        print(f"Testing {len(configurations)} configurations")
        print(f"{'=' * 80}\n")

        results = []

        for config in configurations:
            model_size = config["model_size"]
            dataset_size = config["dataset_size"]

            print(f"\nValidating: Model={model_size}, Dataset={dataset_size}")

            analyzer = ThresholdAnalyzer(
                trainer=config["trainer"],
                steering=config["steering"],
                gate_config=config["gate_config"],
            )

            threshold_results = analyzer.run_threshold_sweep(
                example_counts=example_counts,
                test_prompts=config["test_prompts"],
                num_trials=3,
            )

            # Detect threshold
            threshold_count, threshold_csr = analyzer.detect_threshold(threshold_results)

            validation_result = {
                "model_size": model_size,
                "dataset_size": dataset_size,
                "threshold_count": threshold_count,
                "threshold_csr": threshold_csr,
                "full_results": threshold_results,
            }

            results.append(validation_result)
            self.validation_results.append(validation_result)

            print(f"  Threshold: {threshold_count} examples (CSR={threshold_csr:.1%})")

        return results

    def analyze_constant_count_behavior(self, results: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze whether threshold is constant across scales.

        Args:
            results: Validation results (uses self.validation_results if None)

        Returns:
            Analysis dict with statistics
        """
        if results is None:
            results = self.validation_results

        if not results:
            raise ValueError("No validation results available.")

        thresholds = [r["threshold_count"] for r in results]

        analysis = {
            "mean_threshold": np.mean(thresholds),
            "std_threshold": np.std(thresholds),
            "min_threshold": np.min(thresholds),
            "max_threshold": np.max(thresholds),
            "is_constant_count": np.std(thresholds) < 20,  # Threshold variation < 20 examples
            "all_thresholds": thresholds,
        }

        print(f"\n{'=' * 80}")
        print("CONSTANT-COUNT ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Mean threshold: {analysis['mean_threshold']:.1f} examples")
        print(f"Std deviation: {analysis['std_threshold']:.1f} examples")
        print(f"Range: [{analysis['min_threshold']}, {analysis['max_threshold']}]")
        print(f"Constant-count behavior: {analysis['is_constant_count']}")
        print(f"{'=' * 80}\n")

        return analysis

    def save_validation_results(self, output_path: str):
        """Save multi-scale validation results."""
        output = {
            "validation_results": self.validation_results,
            "constant_count_analysis": self.analyze_constant_count_behavior(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Validation results saved to {output_path}")
