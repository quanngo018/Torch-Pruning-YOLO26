import torch
import torch.nn as nn
import torch_pruning as tp
import types

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck


MODEL_PATH = (
    "/home/edabk/Workspace/Quan/PROJECTS/Plate_Recognition/"
    "v3/models/car_detection_model/car_26n.pt"
)

# ============================================================
# 1. Load
# ============================================================

def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child_module in list(module.named_children()):
        if isinstance(child_module, C2f):
            print("FOUND C2f:", name)
            print("m[0] type:", type(child_module.m[0]))
            print("m[0] children:", list(child_module.m[0].named_children()))
            # Replace C2f with C2f_v2 while preserving its parameters
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=False,
                            g=1,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)


def detect_forward_for_pruning(self, x):
    """Run both Detect branches without detach while tracing pruning dependencies."""
    preds = self.forward_head(x, **self.one2many)

    if self.end2end:
        one2one = self.forward_head(x, **self.one2one)
        preds = {
            "one2many": preds,
            "one2one": one2one,
        }

    if self.training:
        return preds

    y = self._inference(preds["one2one"] if self.end2end else preds)
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))
    return y if self.export else (y, preds)


yolo = YOLO(MODEL_PATH)
model = yolo.model
replace_c2f_with_c2f_v2(model)
model.train()

for p in model.parameters():
    p.requires_grad_(True)

dummy_input = torch.randn(1, 3, 640, 640)

module_names = {
    module: name
    for name, module in model.named_modules()
}


# ============================================================
# 2. Ignore Attention Conv roots
# ============================================================

attention_convs = [
    module
    for name, module in model.named_modules()
    if isinstance(module, nn.Conv2d)
    and (name.startswith("attn.") or ".attn." in name)
]


# ============================================================
# 3. Protect final Detect predictors
# ============================================================

detect_outputs = []

for module in model.modules():
    if not isinstance(module, Detect):
        continue

    print(
        f"Detect: nc={module.nc}, "
        f"reg_max={module.reg_max}, "
        f"nl={module.nl}, "
        f"end2end={module.end2end}"
    )

    # box/class heads: one-to-many + one-to-one
    for attr in (
        "cv2",
        "cv3",
        "one2one_cv2",
        "one2one_cv3",
    ):
        if not hasattr(module, attr):
            continue

        heads = getattr(module, attr)

        for branch in heads:
            final_conv = branch[-1]

            if not isinstance(final_conv, nn.Conv2d):
                raise TypeError(
                    f"{attr} final layer is "
                    f"{type(final_conv).__name__}, expected Conv2d"
                )

            detect_outputs.append(final_conv)


# Remove duplicates defensively
detect_outputs = list(dict.fromkeys(detect_outputs))

ignored_layers = list(
    dict.fromkeys(attention_convs + detect_outputs)
)


print("\nIgnored Attention Conv:")
for m in attention_convs:
    print(" ", module_names[m])

print("\nProtected Detect outputs:")
for m in detect_outputs:
    print(
        f"  {module_names[m]} "
        f"in={m.in_channels} "
        f"out={m.out_channels}"
    )

print("\nTotal ignored:", len(ignored_layers))


# ============================================================
# 4. Save channel structure BEFORE pruning
# ============================================================

before_channels = {
    name: (m.in_channels, m.out_channels, m.groups)
    for name, m in model.named_modules()
    if isinstance(m, nn.Conv2d)
}

protected_out_before = {
    module_names[m]: m.out_channels
    for m in detect_outputs
}


# ============================================================
# 5. Create pruner
# ============================================================

#importance = tp.importance.GroupMagnitudeImportance(p=2)

 # DEBUG IMPORTANCE
class DebugImportance:
    def __init__(self, model):
        self.base = tp.importance.GroupMagnitudeImportance(p=2)
        self.names = {
            m: name
            for name, m in model.named_modules()
        }
        self.call_count = 0

    def __call__(self, group):
        self.call_count += 1

        try:
            return self.base(group)

        except IndexError:
            print(
                f"\n{'=' * 70}\n"
                f"IMPORTANCE ERROR - GROUP #{self.call_count}\n"
                f"{'=' * 70}"
            )

            for i, (dep, idxs) in enumerate(group):
                layer = dep.target.module
                name = self.names.get(layer, "<unknown>")
                prune_fn = dep.handler

                if idxs is None:
                    idx_info = "None"
                else:
                    idx_list = list(idxs)
                    idx_info = (
                        f"len={len(idx_list)}, "
                        f"min={min(idx_list) if idx_list else None}, "
                        f"max={max(idx_list) if idx_list else None}"
                    )

                print(
                    f"\n[{i}]"
                    f"\n  layer    = {name}"
                    f"\n  type     = {type(layer).__name__}"
                    f"\n  prune_fn = {prune_fn}"
                    f"\n  idxs     = {idx_info}"
                )

                if hasattr(layer, "weight"):
                    print(
                        f"  weight   = {tuple(layer.weight.shape)}"
                    )

                if isinstance(layer, nn.Conv2d):
                    print(
                        f"  Conv     = "
                        f"in={layer.in_channels}, "
                        f"out={layer.out_channels}, "
                        f"groups={layer.groups}"
                    )

            raise

importance = DebugImportance(model)

original_detect_forwards = {}

for detect_module in model.modules():
    if isinstance(detect_module, Detect):
        original_detect_forwards[detect_module] = detect_module.forward
        detect_module.forward = types.MethodType(
            detect_forward_for_pruning,
            detect_module,
        )

try:
    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=dummy_input,
        importance=importance,

        pruning_ratio=0.1,
        iterative_steps=1,

        ignored_layers=ignored_layers,
        root_module_types=[nn.Conv2d],
    )
finally:
    for detect_module, original_forward in original_detect_forwards.items():
        detect_module.forward = original_forward


# ============================================================
# 6. Baseline
# ============================================================

base_macs, base_params = tp.utils.count_ops_and_params(
    model,
    dummy_input
)

print("\n========== BEFORE ==========")
print(f"MACs   : {base_macs / 1e9:.4f} G")
print(f"Params : {base_params / 1e6:.4f} M")


# ============================================================
# 7. PRUNE 10%
# ============================================================

print("\n>>> CALLING pruner.step() <<<")

pruner.step()

print("pruner.step(): PASS")


# ============================================================
# 8. Verify Detect output dimensions
# ============================================================

print("\n========== DETECT CHECK ==========")

for m in detect_outputs:
    name = module_names[m]

    before = protected_out_before[name]
    after = m.out_channels

    print(f"{name}: {before} -> {after}")

    assert before == after, (
        f"ERROR: Detect output changed: "
        f"{name}: {before} -> {after}"
    )

print("Detect output dimensions: PASS")


# ============================================================
# 9. Test forward
# ============================================================

print("\n========== FORWARD CHECK ==========")

with torch.no_grad():
    output = model(dummy_input)

print("Forward after pruning: PASS")


# ============================================================
# 10. Compare MACs / Params
# ============================================================

pruned_macs, pruned_params = tp.utils.count_ops_and_params(
    model,
    dummy_input
)

print("\n========== RESULT ==========")

print(
    f"MACs   : {base_macs / 1e9:.4f} G"
    f" -> {pruned_macs / 1e9:.4f} G"
)

print(
    f"Params : {base_params / 1e6:.4f} M"
    f" -> {pruned_params / 1e6:.4f} M"
)

print(
    f"MAC reduction: "
    f"{100 * (1 - pruned_macs / base_macs):.2f}%"
)

print(
    f"Param reduction: "
    f"{100 * (1 - pruned_params / base_params):.2f}%"
)


# ============================================================
# 11. Show which Conv changed
# ============================================================

print("\n========== CHANGED CONVS ==========")

changed = 0

for name, m in model.named_modules():
    if not isinstance(m, nn.Conv2d):
        continue

    old = before_channels.get(name)

    if old is None:
        continue

    new = (m.in_channels, m.out_channels, m.groups)

    if old != new:
        changed += 1
        print(
            f"{name}: "
            f"{old} -> {new}"
        )

print("\nNumber of changed Conv2d:", changed)