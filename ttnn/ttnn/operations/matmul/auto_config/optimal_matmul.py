# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Top-level API: ``optimal_matmul`` — a torch.matmul-like interface that
automatically selects the most performant matmul configuration.
"""

import math
import os
from enum import Enum
from typing import Optional, Tuple, Union

from .config_space import (
    DeviceConstraints,
    MatmulConfig,
    MatmulShape,
    TILE_SIZE,
    get_valid_subblock,
)
from .heuristic_model import (
    DNNConfigPredictor,
    HeuristicConfigPredictor,
    ScoringWeights,
)


class MatmulBackend(Enum):
    TTNN_MATMUL = "ttnn_matmul"
    MINIMAL_MATMUL = "minimal_matmul"
    AUTO = "auto"


class PredictorType(Enum):
    HEURISTIC = "heuristic"
    DNN = "dnn"
    AUTO = "auto"


_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
_DEFAULT_MODEL_PATH = os.path.join(_DEFAULT_MODEL_DIR, "matmul_config_model.json")
_predictor = None
_predictor_type = PredictorType.AUTO


def set_predictor(predictor_type, model_path=None, weights=None):
    global _predictor, _predictor_type
    _predictor_type = predictor_type
    if predictor_type == PredictorType.HEURISTIC:
        _predictor = HeuristicConfigPredictor(weights)
    elif predictor_type == PredictorType.DNN:
        _predictor = DNNConfigPredictor(model_path or _DEFAULT_MODEL_PATH)
        _predictor.load()
    elif predictor_type == PredictorType.AUTO:
        dnn = DNNConfigPredictor(model_path or _DEFAULT_MODEL_PATH)
        _predictor = dnn if dnn.load() else HeuristicConfigPredictor(weights)


def _get_predictor():
    global _predictor
    if _predictor is None: set_predictor(PredictorType.AUTO)
    return _predictor


def _extract_shape_from_tensors(a, b, transpose_a=False, transpose_b=False):
    a_s = list(a.padded_shape) if hasattr(a, 'padded_shape') else list(a.shape)
    b_s = list(b.padded_shape) if hasattr(b, 'padded_shape') else list(b.shape)
    if transpose_a: a_s[-2], a_s[-1] = a_s[-1], a_s[-2]
    if transpose_b: b_s[-2], b_s[-1] = b_s[-1], b_s[-2]
    batch = 1
    for d in a_s[:-2]: batch *= d
    return MatmulShape(M=a_s[-2], K=a_s[-1], N=b_s[-1], batch_size=batch)


def _extract_device_constraints(tensor):
    if hasattr(tensor, 'device') and callable(tensor.device):
        device = tensor.device()
        if hasattr(device, 'compute_with_storage_grid_size'):
            grid = device.compute_with_storage_grid_size()
            return DeviceConstraints(grid_x=getattr(grid, 'x', 8), grid_y=getattr(grid, 'y', 8))
    return DeviceConstraints()


def _is_multi_device(tensor):
    if hasattr(tensor, 'device') and callable(tensor.device):
        device = tensor.device()
        if hasattr(device, 'get_num_devices'): return device.get_num_devices() > 1
    return False


def _select_backend(shape):
    total_tiles = shape.M_tiles * shape.K_tiles * shape.N_tiles
    return MatmulBackend.MINIMAL_MATMUL if total_tiles > 1000 else MatmulBackend.TTNN_MATMUL


def get_optimal_config(M, K, N, grid_x=8, grid_y=8, batch_size=1, fp32_dest_acc_en=False, top_k=1):
    shape = MatmulShape(M=M, K=K, N=N, batch_size=batch_size)
    constraints = DeviceConstraints(grid_x=grid_x, grid_y=grid_y, fp32_dest_acc_en=fp32_dest_acc_en)
    results = _get_predictor().predict(shape, constraints, top_k=top_k)
    return results[0] if top_k == 1 and results else (results if top_k > 1 else None)


def optimal_matmul(input_tensor_a, input_tensor_b, transpose_a=False, transpose_b=False, memory_config=None, dtype=None, compute_kernel_config=None, backend=MatmulBackend.AUTO):
    import ttnn as _ttnn
    shape = _extract_shape_from_tensors(input_tensor_a, input_tensor_b, transpose_a, transpose_b)
    constraints = _extract_device_constraints(input_tensor_a)
    configs = _get_predictor().predict(shape, constraints, top_k=1)
    config = configs[0] if configs else None
    if backend == MatmulBackend.AUTO: backend = _select_backend(shape)
    if backend == MatmulBackend.MINIMAL_MATMUL and config:
        mm_config = _ttnn.MinimalMatmulConfig(M_block_size=config.M_block_size, K_block_size=config.K_block_size, N_block_size=config.N_block_size, subblock_h=config.subblock_h, subblock_w=config.subblock_w, compute_with_storage_grid_size=_ttnn.CoreCoord(config.grid_x, config.grid_y))
        return _ttnn.experimental.minimal_matmul(input_tensor=input_tensor_a, weight_tensor=input_tensor_b, bias_tensor=None, config=mm_config, compute_kernel_config=compute_kernel_config, memory_config=memory_config, dtype=dtype)
    else:
        return _ttnn.matmul(input_tensor_a=input_tensor_a, input_tensor_b=input_tensor_b, transpose_a=transpose_a, transpose_b=transpose_b, memory_config=memory_config, dtype=dtype, compute_kernel_config=compute_kernel_config)


def optimal_linear(input_tensor, weight_tensor, bias_tensor=None, transpose_a=False, transpose_b=False, activation=None, memory_config=None, dtype=None, compute_kernel_config=None, backend=MatmulBackend.AUTO):
    import ttnn as _ttnn
    shape = _extract_shape_from_tensors(input_tensor, weight_tensor, transpose_a, transpose_b)
    constraints = _extract_device_constraints(input_tensor)
    configs = _get_predictor().predict(shape, constraints, top_k=1)
    config = configs[0] if configs else None
    if backend == MatmulBackend.AUTO: backend = _select_backend(shape)
    if backend == MatmulBackend.MINIMAL_MATMUL and config:
        mm_config = _ttnn.MinimalMatmulConfig(M_block_size=config.M_block_size, K_block_size=config.K_block_size, N_block_size=config.N_block_size, subblock_h=config.subblock_h, subblock_w=config.subblock_w, compute_with_storage_grid_size=_ttnn.CoreCoord(config.grid_x, config.grid_y))
        return _ttnn.experimental.minimal_matmul(input_tensor=input_tensor, weight_tensor=weight_tensor, bias_tensor=bias_tensor, fused_activation=activation, config=mm_config, compute_kernel_config=compute_kernel_config, memory_config=memory_config, dtype=dtype)
    else:
        return _ttnn.linear(input_tensor_a=input_tensor, input_tensor_b=weight_tensor, bias=bias_tensor, transpose_a=transpose_a, transpose_b=transpose_b, activation=activation, memory_config=memory_config, dtype=dtype, compute_kernel_config=compute_kernel_config)
