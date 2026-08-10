#!/usr/bin/env python3
"""Config-driven YOLO26 pruning experiment runner.

Usage::

    # Full run (prune + validate + CSV)
    python run_experiment.py configs/exp_global_sweep.yaml

    # Dry-run: only print the parsed config, no GPU work
    python run_experiment.py configs/exp_global_sweep.yaml --dry-run

    # Skip validation (prune + count MACs/Params only, much faster)
    python run_experiment.py configs/exp_global_sweep.yaml --no-val

    # Save pruned model checkpoints
    python run_experiment.py configs/exp_global_sweep.yaml --save-model

    # Override device
    python run_experiment.py configs/exp_global_sweep.yaml --device cpu
"""

import argparse
import sys
from pathlib import Path

# Ensure the yolo26 package root is importable
YOLO26_DIR = Path(__file__).resolve().parent
if str(YOLO26_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO26_DIR))

from modules.config_schema import load_config, print_config
from modules.pruning_benchmark import (
    OUTPUT_DIR,
    evaluate_case,
    load_converted_model,
    print_rows,
    write_csv,
    write_layer_changes_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLO26 pruning experiment from a YAML config.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print the config without running any pruning.",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip mAP validation (only prune + count MACs/Params).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save pruned model checkpoints to the output directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override the validation device (e.g. 0, cpu, mps).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory for CSV and models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Load config ----
    config = load_config(args.config)

    # ---- Apply CLI overrides ----
    if args.device is not None:
        try:
            config["validation"]["device"] = int(args.device)
        except ValueError:
            config["validation"]["device"] = args.device

    output_dir = str(args.output_dir) if args.output_dir else None
    save_model_dir = None
    if args.save_model:
        save_model_dir = str(args.output_dir or OUTPUT_DIR)

    # ---- Dry-run ----
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — config parsed successfully")
        print("=" * 60)
        print_config(config)
        return

    # ---- Load model ----
    print("=" * 60)
    print(f"Experiment: {config['experiment']}")
    print(f"Description: {config['description']}")
    print(f"Cases: {len(config['cases'])}")
    print("=" * 60)

    baseline_model = load_converted_model(config["model_path"])

    # ---- Run cases ----
    rows = []
    for idx, case in enumerate(config["cases"], 1):
        case_name = case["name"]
        print(
            f"\n[{idx}/{len(config['cases'])}] "
            f"Running case: {case_name} ..."
        )

        row = evaluate_case(
            baseline_model,
            experiment=config["experiment"],
            name=case_name,
            pruning_ratio=case["pruning_ratio"],
            global_pruning=case["global_pruning"],
            max_pruning_ratio=case["max_pruning_ratio"],
            importance=case["importance"],
            importance_p=case.get("importance_p", 2),
            iterative_steps=case["iterative_steps"],
            round_to=case["round_to"],
            isomorphic=case.get("isomorphic", False),
            allowed_roots=case["allowed_roots"],
            model_path=config["model_path"],
            data_path=config["data_path"],
            val_config=config["validation"],
            skip_validation=args.no_val,
            save_model_dir=save_model_dir,
        )
        rows.append(row)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print_rows(rows)

    # ---- Write CSV ----
    experiment_name = config["experiment"]
    write_csv(f"{experiment_name}.csv", rows, output_dir=output_dir)
    write_layer_changes_csv(
        f"{experiment_name}_layers.csv", rows, output_dir=output_dir
    )


if __name__ == "__main__":
    main()
