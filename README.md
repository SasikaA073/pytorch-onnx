# Setup

conda env create -f env.yaml

conda activate pytorch-onnx


To download the pytorch model weights to .cache folder 

For Windows 

set TORCH_HOME=.\.cache

For Linux/MacOS 

export TORCH_HOME=./.cache

To download the json file

wget https://gist.githubusercontent.com/SasikaA073/16135e8bb7c254b9f7a97e7701b18024/raw/6a3e91ad40699edce2f79c981e9d696d6f36ea93/imagenet_1000_cls_id_to_label.json -O imagenet_1000_cls_id_to_label.json