# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.cosh using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([0, 1, -1, 0.5, -0.5], dtype=np.float32),  # Typical small values
        np.linspace(-3, 3, num=10, dtype=np.float32),  # Linearly spaced values within a normal range
        np.random.uniform(-10, 10, size=(10,)),        # Random values within a wider range
        np.zeros(10, dtype=np.float32),                # Zero values
        np.full((10,), 20, dtype=np.float32),          # Constant high values
        np.full((10,), -20, dtype=np.float32),         # Constant low values
        np.random.uniform(-5, 5, size=(5, 5)),         # 2D array of random floats
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),  # Special values: Infinity and NaN
        np.array([], dtype=np.float32),               # Empty array
        np.full((10, 10), 15, dtype=np.float32)       # 2D array with constant values
    ]
    return test_cases


# Function to test torch.cosh with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.cosh on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.cosh
        cosh_results = torch.cosh(tensor)
        ret.append(cosh_results.numpy())

    return ret
