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

vision_dataset = {
    "CIFAR10": {
        "num_class": 10,
        "mean_std": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    },
    "CIFAR100": {
        "num_class": 100,
        "mean_std": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    },
    "FashionMNIST": {
        "num_class": 10,
        "mean_std": ([0.5], [0.5])
    },
    "kinetics400":{
        "num_class": 27,
        "mean_std":([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    }
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

