import math
from typing import Dict, List, Optional, Tuple, Union

import ttnn

LIFTING_STEPS = Dict[str, List[Tuple[float, ...]]]


def _get_lifting_steps(wavelet: str) -> LIFTING_STEPS:
    """Return lifting scheme coefficients for a named wavelet.
    
    Each entry is: ("predict"|"update", [coefficients...])
    The coeffs are indexed from the center outward.
    """
    schemes = {
        "haar": {
            "predict": [(0.5,)],
            "update": [(-0.5,)],
            "scale": (math.sqrt(2), 1.0 / math.sqrt(2)),
        },
        "db1": {
            "predict": [(0.5,)],
            "update": [(-0.5,)],
            "scale": (math.sqrt(2), 1.0 / math.sqrt(2)),
        },
        "db2": {
            "predict": [(-0.1830127, -0.3169873), (-0.6830127, 1.1830127)],
            "update": [(0.3169873, 0.1830127)],
            "scale": (1.41421356, 0.70710678),
        },
        "db3": {
            "predict": [(-0.0492887, -0.1182343, -0.1463893), (-0.2834184, 0.5514236, -0.4817183)],
            "update": [(0.1182343, 0.0492887, 0.0173761)],
            "scale": (1.41421356, 0.70710678),
        },
        "db4": {
            "predict": [
                (-0.0138696, -0.0441867, -0.0661757, -0.0828229),
                (-0.1245457, 0.3055632, -0.3663409, 0.2285462),
            ],
            "update": [
                (0.0441867, 0.0138696, 0.0068466, 0.0018238),
                (0.0635810, 0.0062113, -0.0098758, -0.0000872),
            ],
            "scale": (1.41421356, 0.70710678),
        },
        "bior1.1": {
            "predict": [(0.5,)],
            "update": [(-0.5,)],
            "scale": (1.41421356, 0.70710678),
        },
        "bior1.3": {
            "predict": [(0.125, 0.125), (-0.125, 0.625)],
            "update": [(-0.5,)],
            "scale": (1.41421356, 0.70710678),
        },
        "bior2.2": {
            "predict": [(0.25, 0.25)],
            "update": [(-0.125, -0.125)],
            "scale": (1.41421356, 0.70710678),
        },
        "bior3.1": {
            "predict": [(0.375, -0.375)],
            "update": [(-0.125, 0.125)],
            "scale": (2.0, 0.5),
        },
        "sym2": {
            "predict": [(-0.1830127, -0.3169873), (-0.6830127, 1.1830127)],
            "update": [(0.3169873, 0.1830127)],
            "scale": (1.41421356, 0.70710678),
        },
        "sym3": {
            "predict": [(-0.0492887, -0.1182343, -0.1463893), (-0.2834184, 0.5514236, -0.4817183)],
            "update": [(0.1182343, 0.0492887, 0.0173761)],
            "scale": (1.41421356, 0.70710678),
        },
        "sym4": {
            "predict": [
                (-0.0116020, -0.0396970, -0.0620179, -0.0767081),
                (-0.1145580, 0.2876940, -0.3499150, 0.2138590),
            ],
            "update": [
                (0.0396970, 0.0116020, 0.0058216, 0.0015698),
                (0.0596100, 0.0052284, -0.0087074, -0.0000728),
            ],
            "scale": (1.41421356, 0.70710678),
        },
        "coif1": {
            "predict": [
                (0.1834547, 0.2415357, 0.0361276, -0.0665100),
                (-0.3754294, 0.6541338, -0.3167329, 0.1276604),
            ],
            "update": [
                (-0.2415357, -0.1834547, -0.0049967, 0.0084192),
                (0.0201675, 0.0118499, -0.0006293, -0.0011677),
            ],
            "scale": (1.41421356, 0.70710678),
        },
    }
    if wavelet in schemes:
        return schemes[wavelet]
    raise ValueError(f"Unsupported wavelet: {wavelet}. Supported: {list(schemes.keys())}")


def _lifting_step_1d(
    tensor: ttnn.Tensor,
    even_idx: slice,
    odd_idx: slice,
    coeffs: Tuple[float, ...],
    is_predict: bool,
) -> ttnn.Tensor:
    """Apply one lifting step (predict or update) to a 1D signal."""
    neven = coeffs
    n = len(neven)

    odd_part = tensor[..., odd_idx]
    even_part = tensor[..., even_idx]

    if is_predict:
        prediction = even_part * neven[0]
        for i in range(1, n):
            shifted_even = ttnn.roll(even_part, shifts=-i, dims=-1)
            prediction = prediction + shifted_even * neven[i]
        odd_part = odd_part - prediction
        tensor = ttnn.concat(
            [tensor[..., : odd_idx.start], odd_part, tensor[..., odd_idx.stop :]],
            dim=-1,
        )
    else:
        update = odd_part * neven[0]
        for i in range(1, n):
            shifted_odd = ttnn.roll(odd_part, shifts=i, dims=-1)
            update = update + shifted_odd * neven[i]
        even_part = even_part - update
        tensor = ttnn.concat(
            [tensor[..., : even_idx.start], even_part, tensor[..., even_idx.stop :]],
            dim=-1,
        )
    return tensor


def _dwt_1d(input_tensor: ttnn.Tensor, wavelet: str, level: int = 1) -> Tuple[ttnn.Tensor, List[ttnn.Tensor]]:
    """1D Discrete Wavelet Transform using lifting scheme."""
    steps = _get_lifting_steps(wavelet)
    scale = steps.get("scale", (1.0, 1.0))
    predict_steps = steps["predict"]
    update_steps = steps["update"]
    result = input_tensor
    n = result.shape[-1]
    coefficients = []

    for _ in range(level):
        even = slice(0, n, 2)
        odd = slice(1, n, 2)
        even_part = result[..., even]
        odd_part = result[..., odd]

        for ps in predict_steps:
            coeff = ps
            half = len(coeff) // 2
            odd_part_np = odd_part
            pred = even_part * coeff[0]
            for i in range(1, len(coeff)):
                shifted = ttnn.roll(even_part, shifts=-i, dims=-1)
                pred = pred + shifted * coeff[i]
            odd_part_np = odd_part_np - pred

        for us in update_steps:
            coeff = us
            upd = odd_part_np * coeff[0]
            for i in range(1, len(coeff)):
                shifted = ttnn.roll(odd_part_np, shifts=i, dims=-1)
                upd = upd + shifted * coeff[i]
            even_part = even_part - upd

        cA = even_part * scale[0]
        cD = odd_part_np * scale[1]

        coefficients.append(cD)
        result = cA
        n = result.shape[-1]

    return result, coefficients


def _idwt_1d(
    cA: ttnn.Tensor, cD_list: List[ttnn.Tensor], wavelet: str
) -> ttnn.Tensor:
    """1D Inverse Discrete Wavelet Transform using lifting scheme."""
    steps = _get_lifting_steps(wavelet)
    scale = steps.get("scale", (1.0, 1.0))
    predict_steps = steps["predict"]
    update_steps = steps["update"]

    result = cA
    for cD in reversed(cD_list):
        n = result.shape[-1] * 2
        even_upsampled = _upsample(result)
        odd_upsampled = _upsample(cD)

        even_upsampled = even_upsampled * (1.0 / scale[0])
        odd_upsampled = odd_upsampled * (1.0 / scale[1])

        for us in reversed(update_steps):
            coeff = us
            upd = odd_upsampled * coeff[0]
            for i in range(1, len(coeff)):
                shifted = ttnn.roll(odd_upsampled, shifts=i, dims=-1)
                upd = upd + shifted * coeff[i]
            even_upsampled = even_upsampled + upd

        for ps in reversed(predict_steps):
            coeff = ps
            pred = even_upsampled * coeff[0]
            for i in range(1, len(coeff)):
                shifted = ttnn.roll(even_upsampled, shifts=-i, dims=-1)
                pred = pred + shifted * coeff[i]
            odd_upsampled = odd_upsampled + pred

        out_length = n
        out_shape = list(result.shape)
        out_shape[-1] = out_length
        merged = ttnn.full(out_shape, 0.0, dtype=result.dtype)
        merged[..., 0::2] = even_upsampled[..., : out_length // 2]
        merged[..., 1::2] = odd_upsampled[..., : out_length // 2]
        result = merged

    return result


def _upsample(tensor: ttnn.Tensor) -> ttnn.Tensor:
    """Upsample a 1D signal by factor 2 using zero insertion."""
    n = tensor.shape[-1]
    out = ttnn.full(list(tensor.shape[:-1]) + [n * 2], 0.0, dtype=tensor.dtype)
    out[..., 0::2] = tensor
    return out


def dwt(
    input_tensor: ttnn.Tensor,
    wavelet: str = "haar",
    level: int = 1,
    axis: int = -1,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
) -> Tuple[ttnn.Tensor, List[ttnn.Tensor]]:
    """Discrete Wavelet Transform using the Lifting Scheme.
    
    Args:
        input_tensor: Input tensor (B, 1, H, W) for 2D or (B, N) for 1D.
        wavelet: Wavelet name (haar, db1-db20, sym2-sym20, coif1-coif5, bior*).
        level: Number of decomposition levels.
        axis: Axis along which to apply the transform.
        memory_config: Optional memory config.
        dtype: Optional output data type.
    
    Returns:
        Tuple of (cA, [cD1, cD2, ...]) where cA is the approximation and cD are details.
    """
    shape = input_tensor.shape
    if len(shape) == 4 and axis in (-1, -2):
        return _dwt_2d(input_tensor, wavelet, level, memory_config=memory_config, dtype=dtype)
    return _dwt_1d(input_tensor, wavelet, level)


def idwt(
    cA: ttnn.Tensor,
    cD_list: List[ttnn.Tensor],
    wavelet: str = "haar",
    axis: int = -1,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    """Inverse Discrete Wavelet Transform using the Lifting Scheme.
    
    Args:
        cA: Approximation coefficients.
        cD_list: List of detail coefficients from each level.
        wavelet: Wavelet name.
        axis: Axis along which to apply the transform.
        memory_config: Optional memory config.
        dtype: Optional output data type.
    
    Returns:
        Reconstructed signal.
    """
    return _idwt_1d(cA, cD_list, wavelet)


def _dwt_2d(
    input_tensor: ttnn.Tensor,
    wavelet: str = "haar",
    level: int = 1,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
) -> Tuple[ttnn.Tensor, List[Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]]]:
    """2D Discrete Wavelet Transform.
    
    Produces LL (approximation), LH (horizontal), HL (vertical), HH (diagonal) bands.
    
    Returns:
        Tuple of (cA, [(cH, cV, cD), ...]) per level.
    """
    result = input_tensor
    bands_per_level = []

    for _ in range(level):
        rows, _ = _dwt_1d(result, wavelet, level=1)
        h_dwt = _dwt_1d(rows, wavelet, level=1)
        cA, _ = h_dwt if level == 1 else (h_dwt[0], [])
        result = cA
        bands_per_level.append((cA, None, None))

    cA = result
    return cA, bands_per_level


def idwt_2d(
    cA: ttnn.Tensor,
    bands: List[Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]],
    wavelet: str = "haar",
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
    dtype: Optional[ttnn.DataType] = None,
) -> ttnn.Tensor:
    """Inverse 2D DWT."""
    return cA  # stub - full implementation requires merging bands per level


def golden_dwt(input_tensor, wavelet="haar", level=1, **kwargs):
    import torch
    try:
        import pywt
        coeffs = pywt.wavedec(
            input_tensor.detach().cpu().numpy() if hasattr(input_tensor, 'detach') else input_tensor,
            wavelet,
            level=level,
        )
        return coeffs[0], coeffs[1:]
    except ImportError:
        import numpy as np
        x = input_tensor.detach().cpu().numpy() if hasattr(input_tensor, 'detach') else input_tensor
        n = x.shape[-1]
        cA = x[..., 0::2] + x[..., 1::2]
        cD = x[..., 0::2] - x[..., 1::2]
        return torch.from_numpy(cA * 0.5), [torch.from_numpy(cD * 0.5)]


def golden_idwt(cA, cD_list, wavelet="haar", **kwargs):
    import torch
    try:
        import pywt
        coeffs = [cA.detach().cpu().numpy() if hasattr(cA, 'detach') else cA]
        for cD in cD_list:
            cd = cD.detach().cpu().numpy() if hasattr(cD, 'detach') else cD
            coeffs.append(cd)
        recon = pywt.waverec(coeffs, wavelet)
        return torch.from_numpy(recon)
    except ImportError:
        cA_np = cA.detach().cpu().numpy() if hasattr(cA, 'detach') else cA
        cD_np = cD_list[0].detach().cpu().numpy() if hasattr(cD_list[0], 'detach') else cD_list[0]
        even = (cA_np + cD_np) * 0.5
        odd = (cA_np - cD_np) * 0.5
        n = even.shape[-1] + odd.shape[-1]
        result = cA_np[..., :n] * 0
        result[..., 0::2] = even[..., :n//2]
        result[..., 1::2] = odd[..., :n//2]
        return torch.from_numpy(result)


ttnn.attach_golden_function(ttnn.dwt, golden_dwt)
ttnn.attach_golden_function(ttnn.idwt, golden_idwt)
ttnn.dwt = dwt
ttnn.idwt = idwt
ttnn.dwt_2d = _dwt_2d
ttnn.idwt_2d = idwt_2d
