# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Dropout3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input
    # Shape for 3D dropout is typically used for sequences of images or video frames
    test_cases = [np.random.rand(10, 3, 10, 24, 24).astype(np.float32) for _ in range(num_cases)]  # 10 distinct sets
    return test_cases


# Function to test Dropout3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a Dropout3d layer with a dropout probability
    dropout = nn.Dropout3d(p=0.5)  # 50% probability to zero out elements
    ret = []

    # Iterate over each test case
    for i, input_data in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(input_data)

        # Enable training mode to activate dropout behavior
        dropout.eval()

        # Apply Dropout2d
        output_tensor = dropout(input_tensor)
        ret.append(output_tensor[0, 0, 0, :5, :5].numpy())

    return ret
