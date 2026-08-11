"""Shared utilities for reproducible YOLO26 pruning benchmarks.

This module provides the core building blocks used by both the legacy test
scripts (``tests/*.py``) and the new config-driven runner
(``run_experiment.py``).
"""

import csv
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from .c2f_v2 import C2f_v2, replace_c2f_with_c2f_v2
from .config_schema import DEFAULT_PATHS, DEFAULT_VALIDATION, build_importance

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = MODEL_DIR / "outputs"
VALIDATION_PROJECT = MODEL_DIR / "runs" / "pruning_comparison"

# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

CSV_FIELDS = (
    "experiment",
    "name",
    "global_pruning",
    "importance",
    "allowed_roots",
    "pruning_ratio",
    "max_pruning_ratio",
    "iterative_steps",
    "round_to",
    "eligible_roots",
    "changed_convs",
    "changed_out_channels",
    "total_pruned_out_channels",
    "changed_layers",
    "output_pruned_layers",
    "parameters",
    "macs",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "error",
)

LAYER_CSV_FIELDS = (
    "experiment",
    "name",
    "pruning_ratio",
    "layer",
    "before_in",
    "after_in",
    "before_out",
    "after_out",
    "before_groups",
    "after_groups",
    "pruned_in",
    "pruned_out",
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_converted_model(model_path=None):
    """Load a YOLO checkpoint and replace C2f blocks with pruning-friendly C2f_v2.

    Args:
        model_path: Path to the ``.pt`` checkpoint.  Falls back to the
            default path from ``config_schema.DEFAULT_PATHS`` when *None*.

    Returns:
        The converted model in eval mode.
    """
    if model_path is None:
        model_path = DEFAULT_PATHS["model_path"]
    model = YOLO(str(model_path)).model
    replace_c2f_with_c2f_v2(model)
    model.eval()
    assert any(isinstance(module, C2f_v2) for module in model.modules())
    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_model(model, run_name, *, model_path=None, data_path=None,
                   val_config=None):
    """Run Ultralytics validation and return box-detection metrics.

    Args:
        model: The (possibly pruned) PyTorch model.
        run_name: Subdirectory name for the validation run.
        model_path: Checkpoint used to create the YOLO wrapper.
        data_path: Dataset YAML path.
        val_config: Dict with keys ``imgsz``, ``batch``, ``device``,
            ``workers``, ``conf``, ``iou``.  Missing keys are filled from
            ``DEFAULT_VALIDATION``.

    Returns:
        Dict with ``precision``, ``recall``, ``map50``, ``map50_95``.
    """
    if model_path is None:
        model_path = DEFAULT_PATHS["model_path"]
    if data_path is None:
        data_path = DEFAULT_PATHS["data_path"]

    vc = dict(DEFAULT_VALIDATION)
    if val_config:
        vc.update(val_config)

    validation_yolo = YOLO(str(model_path))
    validation_yolo.model = deepcopy(model).eval()
    metrics = validation_yolo.val(
        data=str(data_path),
        split="val",
        imgsz=vc["imgsz"],
        batch=vc["batch"],
        device=vc["device"],
        workers=vc["workers"],
        conf=vc["conf"],
        iou=vc["iou"],
        project=str(VALIDATION_PROJECT),
        name=run_name,
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


# ---------------------------------------------------------------------------
# Layer inspection helpers
# ---------------------------------------------------------------------------


def conv_signature(model):
    """Return a dict mapping Conv2d name → (in_channels, out_channels, groups)."""
    return {
        name: (module.in_channels, module.out_channels, module.groups)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def layer_changes(before, after):
    """Describe every Conv2d shape changed by a pruning dependency group."""
    changes = []
    for name, before_shape in before.items():
        after_shape = after[name]
        if before_shape == after_shape:
            continue
        before_in, before_out, before_groups = before_shape
        after_in, after_out, after_groups = after_shape
        changes.append(
            {
                "layer": name,
                "before_in": before_in,
                "after_in": after_in,
                "before_out": before_out,
                "after_out": after_out,
                "before_groups": before_groups,
                "after_groups": after_groups,
                "pruned_in": max(0, before_in - after_in),
                "pruned_out": max(0, before_out - after_out),
            }
        )
    return changes


# ---------------------------------------------------------------------------
# Root / layer selection
# ---------------------------------------------------------------------------


def protected_convs(model):
    """Return the set of Conv2d modules inside Attention and Detect blocks."""
    attention = {
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
        and (name.startswith("attn.") or ".attn." in name)
    }
    detect = set()
    for detect_module in model.modules():
        if isinstance(detect_module, Detect):
            detect.update(
                module
                for module in detect_module.modules()
                if isinstance(module, nn.Conv2d)
            )
    return attention | detect


def select_roots(model, allowed_root_names=None):
    """Partition Conv2d layers into eligible roots and ignored layers.

    Args:
        model: The PyTorch model.
        allowed_root_names: Optional set/list of Conv2d names to allow.
            When *None*, all Conv2d layers except Attention and Detect
            are eligible.

    Returns:
        Tuple of (eligible_dict, ignored_list).
    """
    all_convs = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    if allowed_root_names is None:
        protected = protected_convs(model)
        eligible = {
            name: module
            for name, module in all_convs.items()
            if module not in protected
        }
    else:
        missing = set(allowed_root_names) - set(all_convs)
        if missing:
            raise KeyError(f"Unknown allowed roots: {sorted(missing)}")
        eligible = {name: all_convs[name] for name in allowed_root_names}

    eligible_modules = set(eligible.values())
    ignored = [
        module
        for module in all_convs.values()
        if module not in eligible_modules
    ]
    return eligible, ignored


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    baseline_model,
    *,
    # --- Identification ---
    experiment="default",
    name,
    # --- Pruning parameters ---
    pruning_ratio=None,
    global_pruning=False,
    max_pruning_ratio=1.0,
    importance="l2",
    importance_p=2,
    iterative_steps=1,
    round_to=None,
    isomorphic=False,
    allowed_roots=None,
    # --- Validation ---
    model_path=None,
    data_path=None,
    val_config=None,
    skip_validation=False,
    # --- Output ---
    save_model_dir=None,
    # --- Legacy compat ---
    example_input=None,
):
    """Prune a model, measure changes, optionally validate, and return a result row.

    This is the central workhorse: it deep-copies *baseline_model*, applies
    structured pruning according to the given parameters, records every
    Conv2d shape change, counts MACs/parameters, and optionally runs
    Ultralytics validation.

    Args:
        baseline_model: The pre-loaded model (will be deep-copied).
        experiment: Experiment name for CSV grouping.
        name: Unique case name within the experiment.
        pruning_ratio: Channel pruning ratio.  *None* means baseline-only.
        global_pruning: Use global (cross-layer) pruning ranking.
        max_pruning_ratio: Cap per-layer pruning ratio.
        importance: Importance method name (``"l1"``, ``"l2"``, ``"taylor"``, etc.).
        importance_p: p-norm for GroupMagnitudeImportance.
        iterative_steps: Number of iterative pruning steps.
        round_to: Round remaining channels to a multiple of this value.
        isomorphic: Enable isomorphic pruning.
        allowed_roots: Set of Conv2d names to prune, or *None* for auto.
        model_path: Checkpoint path (for YOLO wrapper in validation).
        data_path: Dataset YAML path.
        val_config: Validation config dict (imgsz, batch, device, …).
        skip_validation: If *True*, skip mAP evaluation.
        save_model_dir: If set, save the pruned model to this directory.
        example_input: Custom dummy input tensor.  Defaults to
            ``torch.randn(1, 3, 640, 640)``.

    Returns:
        Dict with all CSV columns plus ``_layer_changes`` (list) and
        ``_model`` (the pruned model, if pruning was performed).
    """
    if example_input is None:
        imgsz = 640
        if val_config and "imgsz" in val_config:
            imgsz = val_config["imgsz"]
        example_input = torch.randn(1, 3, imgsz, imgsz)

    model = deepcopy(baseline_model).eval()
    before = conv_signature(model)

    allowed_text = (
        "auto-protect"
        if allowed_roots is None
        else "|".join(sorted(allowed_roots))
    )
    row = {
        "experiment": experiment,
        "name": name,
        "global_pruning": global_pruning,
        "importance": importance,
        "allowed_roots": allowed_text,
        "pruning_ratio": "" if pruning_ratio is None else pruning_ratio,
        "max_pruning_ratio": max_pruning_ratio,
        "iterative_steps": iterative_steps,
        "round_to": round_to or "",
        "error": "",
    }

    try:
        eligible, ignored = select_roots(model, allowed_roots)
        row["eligible_roots"] = len(eligible)

        if pruning_ratio is not None:
            assert all(not m.training for m in model.modules())

            imp_obj = build_importance(
                {"importance": importance, "importance_p": importance_p}
            )

            pruner = tp.pruner.MagnitudePruner(
                model=model,
                example_inputs=example_input,
                importance=imp_obj,
                global_pruning=global_pruning,
                pruning_ratio=pruning_ratio,
                max_pruning_ratio=max_pruning_ratio,
                iterative_steps=iterative_steps,
                round_to=round_to,
                isomorphic=isomorphic,
                ignored_layers=ignored,
                root_module_types=[nn.Conv2d],
            )
            for _ in range(iterative_steps):
                pruner.step()

        after = conv_signature(model)
        changes = layer_changes(before, after)
        output_changes = [c for c in changes if c["pruned_out"] > 0]

        row["_layer_changes"] = changes
        row["_model"] = model
        row["changed_convs"] = len(changes)
        row["changed_out_channels"] = len(output_changes)
        row["total_pruned_out_channels"] = sum(
            c["pruned_out"] for c in changes
        )
        row["changed_layers"] = "|".join(
            f"{c['layer']}:{c['before_in']}x{c['before_out']}"
            f"->{c['after_in']}x{c['after_out']}"
            for c in changes
        )
        row["output_pruned_layers"] = "|".join(
            f"{c['layer']}:{c['before_out']}->{c['after_out']}"
            for c in output_changes
        )

        macs, parameters = tp.utils.count_ops_and_params(
            model, example_input
        )
        row["parameters"] = int(parameters)
        row["macs"] = int(macs)

        if not skip_validation:
            row.update(
                validate_model(
                    model,
                    f"{experiment}_{name}",
                    model_path=model_path,
                    data_path=data_path,
                    val_config=val_config,
                )
            )

        # Save pruned checkpoint
        if save_model_dir is not None and pruning_ratio is not None:
            save_dir = Path(save_model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{experiment}_{name}.pt"
            ckpt = {
                "model": deepcopy(model).float(),
                "experiment": experiment,
                "name": name,
                "pruning_ratio": pruning_ratio,
            }
            torch.save(ckpt, save_path)
            row["_saved_path"] = str(save_path)
            print(f"Model saved: {save_path}")

    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        row.setdefault("eligible_roots", 0)
        row.setdefault("changed_convs", 0)
        row.setdefault("changed_out_channels", 0)
        row.setdefault("total_pruned_out_channels", 0)
        row.setdefault("changed_layers", "")
        row.setdefault("output_pruned_layers", "")
        row.setdefault("_layer_changes", [])
        row.setdefault("_model", None)
        row.setdefault(
            "parameters",
            sum(p.numel() for p in baseline_model.parameters()),
        )
        row.setdefault("macs", "")
        for metric in ("precision", "recall", "map50", "map50_95"):
            row.setdefault(metric, "")

    return row


# ---------------------------------------------------------------------------
# Printing & CSV output
# ---------------------------------------------------------------------------


def print_rows(rows):
    """Print a human-readable one-line summary for each result row."""
    for row in rows:
        if row["error"]:
            print(f"{row['experiment']}/{row['name']}: ERROR {row['error']}")
        else:
            metrics_parts = []
            if "map50" in row and row["map50"] != "":
                metrics_parts.append(f"mAP50={row['map50']:.6f}")
                metrics_parts.append(f"mAP50-95={row['map50_95']:.6f}")
            metrics_parts.append(f"params={row['parameters']}")
            metrics_parts.append(f"changed_convs={row['changed_convs']}")
            metrics_parts.append(
                f"changed_out={row['changed_out_channels']}"
            )
            print(
                f"{row['experiment']}/{row['name']}: "
                + ", ".join(metrics_parts)
            )


def write_csv(filename, rows, output_dir=None):
    """Write summary CSV with one row per case.

    Args:
        filename: CSV file name (not a full path).
        rows: List of result dicts from ``evaluate_case``.
        output_dir: Directory to write into.  Defaults to ``MODEL_DIR/outputs``.

    Returns:
        The resolved output path.
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / filename
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS}
            for row in rows
        )
    print(f"CSV saved: {output_path}")
    return output_path


def write_layer_changes_csv(filename, rows, output_dir=None):
    """Write one row per changed Conv2d layer for detailed inspection.

    Args:
        filename: CSV file name.
        rows: List of result dicts from ``evaluate_case``.
        output_dir: Directory to write into.  Defaults to ``MODEL_DIR/outputs``.

    Returns:
        The resolved output path.
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    output_path = out / filename
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LAYER_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            for change in row.get("_layer_changes", []):
                writer.writerow(
                    {
                        "experiment": row["experiment"],
                        "name": row["name"],
                        "pruning_ratio": row["pruning_ratio"],
                        **change,
                    }
                )
    print(f"Layer CSV saved: {output_path}")
    return output_path
