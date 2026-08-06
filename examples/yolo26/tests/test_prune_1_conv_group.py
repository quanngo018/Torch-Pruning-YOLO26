import torch_pruning as tp
from ultralytics import YOLO
import torch

yolo = YOLO("/home/edabk/Workspace/Quan/third_party/Torch-Pruning/examples/yolo26/yolo26n.pt")
raw_model = yolo.model
raw_model.eval() # Set the model to evaluation mode

target_conv = raw_model.model[0].conv
next_conv = raw_model.model[1].conv

print("type(target_conv):", type(target_conv))
print("target_conv:", target_conv)
print("target_conv.weight.shape:", target_conv.weight.shape)
print("target_conv.in_channels:", target_conv.in_channels)
print("target_conv.out_channels:", target_conv.out_channels)

print("type(next_conv):", type(next_conv))
print("next_conv:", next_conv)
print("next_conv.weight.shape:", next_conv.weight.shape)
print("next_conv.in_channels:", next_conv.in_channels)
print("next_conv.out_channels:", next_conv.out_channels)

pruning_indexes = [0, 1] # Prune the first two output channels of the target convolutional layer (testing only)

dummy_input = torch.randn(1, 3, 640, 640) # Create a dummy input tensor with shape (1, 3, 640, 640) representing a batch of 1 image with 3 color channels and size 640x640

raw_model.requires_grad_(True) # Set requires_grad to True for all parameters in the model to enable gradient computation
print("target required_grad: ", target_conv.weight.requires_grad) # Check if the target convolutional layer's weights require gradients

DG = tp.DependencyGraph()
DG.build_dependency(
    raw_model, 
    dummy_input
)
group = DG.get_pruning_group(
    target_conv,
    tp.prune_conv_out_channels,
    idxs=pruning_indexes
)
print(group)

print("Group valid: ", DG.check_pruning_group(group))

print("Before pruning:")
print("target_conv.weight.shape:", target_conv.weight.shape)
print("next_conv.weight.shape:", next_conv.weight.shape)

group.prune()

print("After pruning:")
print("target_conv.weight.shape:", target_conv.weight.shape)
print("next_conv.weight.shape:", next_conv.weight.shape)

with torch.no_grad():
    output = raw_model(dummy_input)
