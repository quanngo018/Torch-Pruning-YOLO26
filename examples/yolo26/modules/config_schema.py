"""Load, validate, and merge YAML experiment configs for YOLO26 pruning."""

from copy import deepcopy
from pathlib import Path

import yaml
import torch_pruning as tp


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "model_path": str(
        Path(
            "/home/edabk/Workspace/Quan/PROJECTS/Plate_Recognition/"
            "v3/models/car_detection_model/car_26n.pt"
        )
    ),
    "data_path": str(
        Path("/home/edabk/Workspace/Quan/TRAIN/train_car/car.yaml")
    ),
}

DEFAULT_VALIDATION = {
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 8,
    "conf": 0.001,
    "iou": 0.7,
}

DEFAULT_CASE = {
    "pruning_ratio": None,
    "global_pruning": False,
    "max_pruning_ratio": 1.0,
    "importance": "l2",
    "importance_p": 2,
    "iterative_steps": 1,
    "round_to": None,
    "allowed_roots": None,
    "isomorphic": False,
}

# ---------------------------------------------------------------------------
# Importance factory
# ---------------------------------------------------------------------------

_IMPORTANCE_REGISTRY = {
    "l1": lambda cfg: tp.importance.GroupMagnitudeImportance(p=1),
    "l2": lambda cfg: tp.importance.GroupMagnitudeImportance(p=cfg.get("importance_p", 2)),
    "magnitude": lambda cfg: tp.importance.GroupMagnitudeImportance(p=cfg.get("importance_p", 2)),
    "taylor": lambda cfg: tp.importance.GroupTaylorImportance(),
    "bn_scale": lambda cfg: tp.importance.BNScaleImportance(),
    "lamp": lambda cfg: tp.importance.LAMPImportance(),
    "fpgm": lambda cfg: tp.importance.FPGMImportance(),
    "random": lambda cfg: tp.importance.RandomImportance(),
}


def build_importance(case_config):
    """Create a tp.importance instance from a case config dict.

    Args:
        case_config: dict with at least ``"importance"`` key (str).

    Returns:
        A ``tp.importance.Importance`` instance.

    Raises:
        ValueError: If the importance name is not recognised.
    """
    name = case_config.get("importance", "l2").lower()
    factory = _IMPORTANCE_REGISTRY.get(name)
    if factory is None:
        supported = ", ".join(sorted(_IMPORTANCE_REGISTRY))
        raise ValueError(
            f"Unknown importance '{name}'. Supported: {supported}"
        )
    return factory(case_config)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _merge(defaults, overrides):
    """Return *defaults* updated with non-None values from *overrides*."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is not None or key not in merged:
            merged[key] = value
    return merged


def load_config(yaml_path):
    """Read a YAML experiment file and return a normalised config dict.

    The returned dict has the structure::

        {
            "experiment": str,
            "description": str,
            "model_path": str,
            "data_path": str,
            "validation": {...},
            "cases": [
                {"name": str, "pruning_ratio": float|None, ...},
                ...
            ],
        }

    Each case is the result of merging ``DEFAULT_CASE`` ← file-level
    ``defaults`` ← per-case overrides, so every case carries every key.

    Args:
        yaml_path: Path to the YAML experiment file.

    Returns:
        Normalised config dict.

    Raises:
        FileNotFoundError: If *yaml_path* does not exist.
        ValueError: If required fields are missing or invalid.
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # --- Top-level fields ---
    config = {
        "experiment": raw.get("experiment", yaml_path.stem),
        "description": raw.get("description", ""),
        "model_path": raw.get("model_path", DEFAULT_PATHS["model_path"]),
        "data_path": raw.get("data_path", DEFAULT_PATHS["data_path"]),
    }

    # --- Validation block ---
    raw_val = raw.get("validation", {}) or {}
    config["validation"] = _merge(DEFAULT_VALIDATION, raw_val)

    # --- Defaults block (file-level) ---
    file_defaults = raw.get("defaults", {}) or {}
    merged_defaults = _merge(DEFAULT_CASE, file_defaults)

    # --- Cases ---
    raw_cases = raw.get("cases")
    if not raw_cases:
        raise ValueError(
            f"Config '{yaml_path.name}' must contain a non-empty 'cases' list."
        )

    cases = []
    seen_names = set()
    for idx, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"Case #{idx} in '{yaml_path.name}' must be a dict, "
                f"got {type(raw_case).__name__}."
            )
        name = raw_case.get("name")
        if not name:
            raise ValueError(
                f"Case #{idx} in '{yaml_path.name}' is missing a 'name' field."
            )
        if name in seen_names:
            raise ValueError(
                f"Duplicate case name '{name}' in '{yaml_path.name}'."
            )
        seen_names.add(name)

        case = _merge(merged_defaults, raw_case)

        # Normalise allowed_roots: list → set, null → None
        roots = case.get("allowed_roots")
        if isinstance(roots, list):
            case["allowed_roots"] = set(roots)
        elif roots is None:
            case["allowed_roots"] = None
        else:
            raise ValueError(
                f"Case '{name}': 'allowed_roots' must be a list or null, "
                f"got {type(roots).__name__}."
            )

        cases.append(case)

    config["cases"] = cases
    return config


def print_config(config):
    """Pretty-print a loaded config for dry-run inspection."""
    print(f"Experiment : {config['experiment']}")
    print(f"Description: {config['description']}")
    print(f"Model      : {config['model_path']}")
    print(f"Data       : {config['data_path']}")
    val = config["validation"]
    print(
        f"Validation : imgsz={val['imgsz']}, batch={val['batch']}, "
        f"device={val['device']}, workers={val['workers']}, "
        f"conf={val['conf']}, iou={val['iou']}"
    )
    print(f"Cases ({len(config['cases'])}):")
    for case in config["cases"]:
        ratio = case["pruning_ratio"]
        ratio_str = "baseline (no prune)" if ratio is None else f"{ratio}"
        gp = "global" if case["global_pruning"] else "local"
        imp = case["importance"]
        roots = case["allowed_roots"]
        roots_str = "auto-protect" if roots is None else f"{len(roots)} roots"
        print(
            f"  - {case['name']:20s}  ratio={ratio_str:8s}  {gp:6s}  "
            f"imp={imp:10s}  steps={case['iterative_steps']}  "
            f"round_to={case['round_to']}  roots={roots_str}"
        )
