import argparse
import gc
import os
import threading
import time

import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics import YOLO


MODEL_PATH = (
    "/home/edabk/Workspace/Quan/PROJECTS/Plate_Recognition/"
    "v3/models/car_detection_model/car_26n.pt"
)


# ============================================================
# RAM monitor - không cần psutil
# ============================================================

def read_proc_value(path, key):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    # value in kB
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass

    return None


def rss_gb():
    return read_proc_value("/proc/self/status", "VmRSS:")


def peak_gb():
    return read_proc_value("/proc/self/status", "VmHWM:")


def available_gb():
    return read_proc_value("/proc/meminfo", "MemAvailable:")


def checkpoint(name):
    print(
        f"\n[CHECKPOINT] {name}\n"
        f"  PID           : {os.getpid()}\n"
        f"  Process RSS   : {rss_gb():.3f} GB\n"
        f"  Process Peak  : {peak_gb():.3f} GB\n"
        f"  RAM available : {available_gb():.3f} GB",
        flush=True,
    )


def monitor_memory():
    last_printed = 0.0

    while True:
        rss = rss_gb()
        avail = available_gb()

        if rss is not None:
            # Print whenever process grows another ~250 MB
            if rss - last_printed >= 0.25:
                print(
                    f"[RAM] RSS={rss:.2f} GB | "
                    f"available={avail:.2f} GB",
                    flush=True,
                )
                last_printed = rss

        time.sleep(0.2)


# ============================================================
# Load model
# ============================================================

def prepare_model(img_size, mode):
    checkpoint("BEFORE YOLO()")

    yolo = YOLO(MODEL_PATH)

    checkpoint("AFTER YOLO()")

    model = yolo.model

    if mode == "train":
        model.train()
    else:
        model.eval()

    checkpoint(f"AFTER model.{mode}()")

    # Important for Torch-Pruning AutoGrad tracing
    for p in model.parameters():
        p.requires_grad_(True)

    checkpoint("AFTER requires_grad_(True)")

    print(
        "Trainable tensors:",
        sum(p.requires_grad for p in model.parameters()),
        flush=True,
    )

    dummy_input = torch.randn(
        1,
        3,
        img_size,
        img_size,
    )

    checkpoint("AFTER dummy_input")

    return model, dummy_input


# ============================================================
# TEST 1
# Only PyTorch forward + AutoGrad
# ============================================================

def test_forward(model, dummy_input):
    print("\n" + "=" * 70, flush=True)
    print("TEST: FORWARD ONLY", flush=True)
    print("=" * 70, flush=True)

    torch.set_grad_enabled(True)

    checkpoint("BEFORE model(dummy_input)")

    # >>> DANGEROUS LINE 1 <<<
    output = model(dummy_input)

    checkpoint("AFTER model(dummy_input)")

    print("Forward: PASS", flush=True)

    # Check whether graph memory can be released
    del output
    gc.collect()

    checkpoint("AFTER del output + gc.collect()")


# ============================================================
# TEST 2
# DependencyGraph only
# ============================================================

def test_dg(model, dummy_input):
    print("\n" + "=" * 70, flush=True)
    print("TEST: DEPENDENCY GRAPH", flush=True)
    print("=" * 70, flush=True)

    torch.set_grad_enabled(True)

    checkpoint("BEFORE DependencyGraph()")

    DG = tp.DependencyGraph()

    checkpoint("AFTER DependencyGraph()")

    print(
        "\n>>> NOW CALLING DG.build_dependency() <<<",
        flush=True,
    )

    checkpoint("IMMEDIATELY BEFORE build_dependency")

    # >>> DANGEROUS LINE 2 <<<
    DG.build_dependency(
        model,
        example_inputs=dummy_input,
    )

    checkpoint("AFTER build_dependency")

    print(
        "DG module2node:",
        len(DG.module2node),
        flush=True,
    )

    convs = [
        m
        for m in model.modules()
        if isinstance(m, nn.Conv2d)
    ]

    dg_convs = [
        m
        for m in convs
        if m in DG.module2node
    ]

    print("Conv2d in model:", len(convs), flush=True)
    print("Conv2d registered in DG:", len(dg_convs), flush=True)

    return DG


# ============================================================
# TEST 3
# Group generation
# ============================================================

def test_groups(model, DG):
    print("\n" + "=" * 70, flush=True)
    print("TEST: GET ALL GROUPS + ROOT PROFILING", flush=True)
    print("=" * 70, flush=True)

    module_names = {
        module: name
        for name, module in model.named_modules()
    }

    original_get_pruning_group = DG.get_pruning_group
    call_count = 0

    def profiled_get_pruning_group(*args, **kwargs):
        nonlocal call_count

        call_count += 1

        root_module = args[0] if args else None
        root_name = module_names.get(
            root_module,
            f"<unknown:{type(root_module).__name__}>"
        )

        print(
            f"\n[GROUP BUILD #{call_count}] START\n"
            f"  root = {root_name}\n"
            f"  type = {type(root_module).__name__}\n"
            f"  RSS  = {rss_gb():.3f} GB",
            flush=True,
        )

        result = original_get_pruning_group(*args, **kwargs)

        print(
            f"[GROUP BUILD #{call_count}] END\n"
            f"  root = {root_name}\n"
            f"  group_size = {len(result)}\n"
            f"  RSS = {rss_gb():.3f} GB",
            flush=True,
        )

        return result

    DG.get_pruning_group = profiled_get_pruning_group

    # Ignore every Conv2d root inside every Attention block. Building a pruning
    # group from attn.pe.conv can make dependency traversal grow dramatically.
    # get_all_groups() checks ignored root objects before get_pruning_group(),
    # so these roots never enter group construction.
    dangerous_roots = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
        and (name.startswith("attn.") or ".attn." in name)
    ]

    if not dangerous_roots:
        raise RuntimeError(
            "No Conv2d roots found inside any Attention block"
        )

    print("\nIgnored dangerous Conv2d roots:", flush=True)
    for root in dangerous_roots:
        print(f"  - {module_names[root]}", flush=True)

    groups = DG.get_all_groups(
        ignored_layers=dangerous_roots,
        root_module_types=[nn.Conv2d],
    )

    count = 0

    try:
        for group in groups:
            count += 1

            print(
                f"[YIELD #{count}] "
                f"group_size={len(group)} "
                f"RSS={rss_gb():.3f} GB",
                flush=True,
            )
    finally:
        DG.get_pruning_group = original_get_pruning_group

    print("Number of groups:", count, flush=True)

# ============================================================
# TEST 4
# MagnitudePruner
# ============================================================

def test_pruner(model, dummy_input):
    print("\n" + "=" * 70, flush=True)
    print("TEST: MAGNITUDE PRUNER", flush=True)
    print("=" * 70, flush=True)

    importance = tp.importance.GroupMagnitudeImportance(p=2)

    checkpoint("AFTER GroupMagnitudeImportance")

    print(
        "\n>>> NOW CREATING MagnitudePruner <<<",
        flush=True,
    )

    checkpoint("IMMEDIATELY BEFORE MagnitudePruner")

    # >>> DANGEROUS LINE 3 <<<
    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=dummy_input,
        importance=importance,
        pruning_ratio=0.1,
        iterative_steps=1,
        root_module_types=[nn.Conv2d],
    )

    checkpoint("AFTER MagnitudePruner")

    print("MagnitudePruner: PASS", flush=True)

    return pruner


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=[
            "forward",
            "dg",
            "groups",
            "pruner",
        ],
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
    )

    args = parser.parse_args()

    # Start independent RAM watcher
    monitor = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )
    monitor.start()

    checkpoint("PROGRAM START")

    model, dummy_input = prepare_model(
        args.size,
        args.mode,
    )

    if args.stage == "forward":
        test_forward(model, dummy_input)

    elif args.stage == "dg":
        test_dg(model, dummy_input)

    elif args.stage == "groups":
        DG = test_dg(model, dummy_input)
        test_groups(model, DG)

    elif args.stage == "pruner":
        test_pruner(model, dummy_input)

    checkpoint("PROGRAM END")


if __name__ == "__main__":
    main()
