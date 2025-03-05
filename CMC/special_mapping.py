import os, json

json_file = os.path.join(os.path.dirname(__file__),"pytorch_to_paddlepaddle.json")
try:
    with open(json_file, "r") as file:
        API_MAPPING = json.load(file)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Please check if the file contains valid JSON.")

SUPPORT_PACKAGE_LIST = {
    "pytorch": [
        "torch",
        "torchvision",

    ],
}

TENSOR_MAPPING = {
    "pytorch":{}
}

SPECIALMODULE_MAPPING = {
    "pytorch": {
        "nn":{
            "tensorflow": "layer"
        },
        "Module":{
            "paddlepaddle": "Layer"
        },
        "utils.data":{
            "paddlepaddle": "io"
        },
        "optim": {
           "paddlepaddle": "optimizer"
        }
    }
}

# Different frameworks correspond to different main Modules name
FrameworkPackage = {
    "pytorch": ["torch", "torchvision"],
    "tensorflow": ["tensorflow.keras"],
    "paddlepaddle": ["paddle"],
}
omitSuffixCall = [
    "contiguous"
]

dataTypeMapping = {
    "int": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "short": "int8",
    "bool": "bool"
}

addOp = {
    "pytorch": "torch.add",
    "paddlepaddle": "paddle.add"
}

subOp = {
    "pytorch": "torch.sub",
    "paddlepaddle": "paddle.subtract"
}

divOp = {
    "pytorch": "torch.div",
    "paddlepaddle": "paddle.divide"
}

mulOp = {
    "pytorch": "torch.mul",
    "paddlepaddle": "paddle.multiply"
}

