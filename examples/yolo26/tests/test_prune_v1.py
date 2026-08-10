"""Isolate Torch-Pruning side effects with four YOLO26 validation tests."""

import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules import Detect


YOLO26_DIR = Path(__file__).resolve().parents[1]
if str(YOLO26_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO26_DIR))

from modules.c2f_v2 import C2f_v2, replace_c2f_with_c2f_v2


MODEL_PATH = (
    "/home/edabk/Workspace/Quan/PROJECTS/Plate_Recognition/"
    "v3/models/car_detection_model/car_26n.pt"
)
DATA_PATH = "/home/edabk/Workspace/Quan/TRAIN/train_car/car.yaml"
VALIDATION_PROJECT = YOLO26_DIR / "runs" / "pruner_abcd"
EXAMPLE_INPUT = torch.randn(1, 3, 640, 640)


def validate_model(candidate_model, test_name):
    """Validate a copy so the validator cannot mutate the diagnostic model."""
    validation_yolo = YOLO(MODEL_PATH)
    validation_yolo.model = deepcopy(candidate_model).eval()
    metrics = validation_yolo.val(
        data=DATA_PATH,
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        workers=8,
        conf=0.001,
        iou=0.7,
        project=str(VALIDATION_PROJECT),
        name=test_name,
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


def collect_ignored_layers(model):
    """Protect attention convolutions and all Detect convolution outputs."""
    attention_convs = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
        and (name.startswith("attn.") or ".attn." in name)
    ]

    detect_convs = []
    for detect_module in model.modules():
        if isinstance(detect_module, Detect):
            detect_convs.extend(
                child
                for child in detect_module.modules()
                if isinstance(child, nn.Conv2d)
            )

    ignored_layers = list(dict.fromkeys(attention_convs + detect_convs))
    return ignored_layers, attention_convs, list(dict.fromkeys(detect_convs))


def build_pruner(model, pruning_ratio):
    """Build a pruner in eval mode without modifying Detect.forward."""
    model.eval()
    assert all(not module.training for module in model.modules())

    ignored_layers, attention_convs, detect_convs = collect_ignored_layers(model)
    print(
        f"Building pruner: ratio={pruning_ratio}, "
        f"ignored_attention={len(attention_convs)}, "
        f"ignored_detect_convs={len(detect_convs)}"
    )

    return tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=EXAMPLE_INPUT,
        importance=tp.importance.GroupMagnitudeImportance(p=2),
        pruning_ratio=pruning_ratio,
        iterative_steps=1,
        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d],
    )


def channel_signature(model):
    return {
        name: (module.in_channels, module.out_channels, module.groups)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def print_result(test_name, result):
    if "error" in result:
        print(f"{test_name}: ERROR: {result['error']}")
        return
    print(
        f"{test_name}: "
        f"P={result['precision']:.6f}, "
        f"R={result['recall']:.6f}, "
        f"mAP50={result['map50']:.6f}, "
        f"mAP50-95={result['map50_95']:.6f}, "
        f"changed_convs={result['changed_convs']}"
    )


def evaluate_test(test_name, baseline_model, pruning_ratio=None, call_step=False):
    print(f"\n========== TEST {test_name} ==========")
    model = deepcopy(baseline_model).eval()
    before = channel_signature(model)

    try:
        if pruning_ratio is not None:
            pruner = build_pruner(model, pruning_ratio)
            if call_step:
                print("Calling pruner.step()")
                pruner.step()

        after = channel_signature(model)
        changed = sum(before[name] != value for name, value in after.items())
        metrics = validate_model(model, f"test_{test_name.lower()}")
        metrics["changed_convs"] = changed
        return metrics
    except Exception as error:
        return {
            "error": f"{type(error).__name__}: {error}",
            "changed_convs": sum(
                before.get(name) != value
                for name, value in channel_signature(model).items()
                if name in before
            ),
        }


def main():
    yolo = YOLO(MODEL_PATH)
    converted_model = yolo.model
    replace_c2f_with_c2f_v2(converted_model)
    converted_model.eval()

    assert any(isinstance(module, C2f_v2) for module in converted_model.modules())

    results = {
        # A: Converted model, no pruner.
        "A": evaluate_test("A", converted_model),
        # B: Construct pruner only; do not call step().
        "B": evaluate_test("B", converted_model, pruning_ratio=0.01),
        # C: Execute a zero-ratio pruning step.
        "C": evaluate_test("C", converted_model, pruning_ratio=0.0, call_step=True),
        # D: Execute a real 1% pruning step.
        "D": evaluate_test("D", converted_model, pruning_ratio=0.01, call_step=True),
    }

    print("\n========== A-D SUMMARY ==========")
    for test_name, result in results.items():
        print_result(test_name, result)


if __name__ == "__main__":
    main()
