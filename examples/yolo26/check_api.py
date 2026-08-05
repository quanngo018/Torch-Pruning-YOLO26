def get_ultralytics_version():
    import ultralytics
    return ultralytics.__version__

def get_torch_pruning_version():
    import torch_pruning
    return torch_pruning.__version__

def get_pytorch_version():
    import torch
    return torch.__version__

def main():
    version = get_ultralytics_version()
    assert version == "8.4.69", f"Expected Ultralytics version: 8.4.69, got {version}"

    version = get_torch_pruning_version()
    assert version == "1.6.0", f"Expected Torch-Pruning version: 1.6.0, got {version}"

    version = get_pytorch_version()
    assert version == "2.12.1+cu130", f"Expected PyTorch version: 2.12.1, got {version}"

    from ultralytics import YOLO
    from ultralytics.nn.modules import Conv, Bottleneck, C3k2, C2PSA, Attention, Detect

if __name__ == "__main__":
    main()
    

