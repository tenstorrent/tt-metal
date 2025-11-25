# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate Mean Squared Error between two images

    Args:
        img1: First PIL Image
        img2: Second PIL Image

    Returns:
        MSE value (lower is better, 0 = identical)
    """
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    # Ensure same size
    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size)
        arr2 = np.array(img2).astype(np.float32)

    return np.mean((arr1 - arr2) ** 2)


def calculate_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate Structural Similarity Index

    Args:
        img1: First PIL Image
        img2: Second PIL Image

    Returns:
        SSIM value (0-1, higher is better, 1 = identical)
    """
    arr1 = np.array(img1.convert("L"))  # Convert to grayscale
    arr2 = np.array(img2.convert("L"))

    if arr1.shape != arr2.shape:
        img2 = img2.resize(img1.size)
        arr2 = np.array(img2.convert("L"))

    return ssim(arr1, arr2)


def compare_images(img1: Image.Image, img2: Image.Image, ssim_threshold: float = 0.9):
    """
    Compare images and return metrics

    Args:
        img1: First PIL Image
        img2: Second PIL Image
        ssim_threshold: Threshold for similarity (default 0.9)

    Returns:
        Dictionary with mse, ssim, and similar flag
    """
    mse = calculate_mse(img1, img2)
    ssim_score = calculate_ssim(img1, img2)

    return {"mse": float(mse), "ssim": float(ssim_score), "similar": ssim_score >= ssim_threshold}
