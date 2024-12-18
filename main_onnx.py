import os, json
os.environ['TORCH_HOME'] = './.cache' 

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from time import time

# Load the pre-trained EfficientNet model (e.g., efficientnet_b0)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.eval()  # Set the model to evaluation mode

torch_model = model
torch_dummy_input = torch.randn(1, 3, 224, 224)
# onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)

# torch.onnx.export(torch_model, torch_dummy_input, "efficientnet_pretrained.onnx", export_params=True)


# Export the model
# src: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
torch.onnx.export(torch_model,               # model being run
                  torch_dummy_input,         # model input (or a tuple for multiple inputs)
                  "efficientnet_pretrained.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})