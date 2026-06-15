# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
from io import BytesIO
from PIL import Image
import torch


def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL image to base64 string

    Args:
        image: PIL Image
        format: Output format (JPEG, PNG)

    Returns:
        Base64-encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def base64_to_pil(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL image

    Args:
        base64_str: Base64-encoded image string

    Returns:
        PIL Image
    """
    img_bytes = base64.b64decode(base64_str)
    return Image.open(BytesIO(img_bytes))


def tensor_to_pil(tensor: torch.Tensor, image_processor) -> Image.Image:
    """
    Convert tensor to PIL image using pipeline's processor

    Args:
        tensor: PyTorch tensor from SDXL pipeline
        image_processor: Image processor from DiffusionPipeline

    Returns:
        PIL Image
    """
    tensor = tensor.unsqueeze(0)
    return image_processor.postprocess(tensor, output_type="pil")[0]
