# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pathlib
import sys
import types

from contextlib import contextmanager
from functools import wraps
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from typing import Callable

from loguru import logger

import ttnn
import ttnn.operation_tracer
from ttnn.trace_allocation_config import TRACE_ALLOC_DIAGNOSTICS, TRACE_ALLOC_TRACKING


@dataclasses.dataclass(frozen=True)
class GoldenComparisonConfig:
    """Defines a per-tensor comparison policy for golden outputs.
    Controls the metric, applicability, tolerances, and compared elements.
    """

    method: str  # Comparison metric: "ulp", "allclose", or "skip".
    scope: str = "degenerate"  # Apply only to degenerate PCC cases or to "all" outputs.
    ulp_threshold: float | None = None  # Maximum ULP distance accepted by the ULP metric.
    rtol: float = 1e-5  # Relative tolerance used by the allclose metric.
    atol: float = 1e-4  # Absolute tolerance used by the allclose metric.
    equal_nan: bool = True  # Whether matching NaN positions compare as equal.
    nonfinite: str = "strict"  # Policy for nonfinite values: "strict" or "mask".
    mask: object | None = None  # Optional boolean tensor selecting elements to compare.


def set_golden_comparison_config(
    tensor,
    *,
    method,
    scope="degenerate",
    ulp_threshold=None,
    rtol=1e-5,
    atol=1e-4,
    equal_nan=True,
    nonfinite="strict",
    mask=None,
):
    """Attach a validated comparison policy to a Torch golden tensor.
    Returns the same tensor so golden implementations can configure results inline.
    """

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    if method not in ("ulp", "allclose", "skip"):
        raise ValueError(f"Unsupported golden comparison method: {method}")
    if scope not in ("degenerate", "all"):
        raise ValueError(f"Unsupported golden comparison scope: {scope}")
    if nonfinite not in ("strict", "mask"):
        raise ValueError(f"Unsupported golden nonfinite policy: {nonfinite}")
    if method == "ulp" and ulp_threshold is None:
        raise ValueError("ULP golden comparison requires ulp_threshold")
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected comparison mask to be a torch.Tensor, got {type(mask)}")

    tensor._ttnn_comparison_config = GoldenComparisonConfig(
        method=method,
        scope=scope,
        ulp_threshold=ulp_threshold,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        nonfinite=nonfinite,
        mask=mask,
    )
    return tensor


def _copy_golden_comparison_config(source, destination):
    """Copy recognized golden comparison metadata between tensors.
    Leaves each destination attribute unchanged when the source does not define it.
    """

    comparison_config = getattr(source, "_ttnn_comparison_config", None)
    if comparison_config is not None:
        destination._ttnn_comparison_config = comparison_config
    return destination


def compare_tensors_using_pcc(
    python_fully_qualified_name, golden_outputs, outputs, desired_pcc, level, fail_on_bad_comparison
):
    import numbers
    import torch

    from models.common.utility_functions import comp_pcc, comp_ulp

    if isinstance(golden_outputs, (list, tuple, dict)) or isinstance(outputs, (list, tuple, dict)):
        comparison_records = []
        for golden_output, output in _structured_output_pairs(golden_outputs, outputs):
            comparison_records.extend(
                compare_tensors_using_pcc(
                    python_fully_qualified_name,
                    golden_output,
                    output,
                    desired_pcc,
                    level,
                    fail_on_bad_comparison,
                )
            )
        return comparison_records

    if golden_outputs is None or outputs is None:
        return list(_structured_output_pairs(golden_outputs, outputs))

    if isinstance(golden_outputs, ttnn.DistributedGolden):
        comparison_records = []
        for golden_shard, output_shard in _distributed_comparison_pairs(golden_outputs, outputs):
            comparison_records.extend(
                compare_tensors_using_pcc(
                    python_fully_qualified_name,
                    golden_shard,
                    output_shard,
                    desired_pcc,
                    level,
                    fail_on_bad_comparison,
                )
            )
        return comparison_records

    if isinstance(golden_outputs, numbers.Number) or isinstance(outputs, numbers.Number):
        if not isinstance(golden_outputs, numbers.Number) or not isinstance(outputs, numbers.Number):
            raise TypeError(
                f"Output structure mismatch: golden type {type(golden_outputs)} does not match output type {type(outputs)}"
            )
        return compare_scalar_outputs(
            python_fully_qualified_name,
            golden_outputs,
            outputs,
            desired_pcc,
            level,
            fail_on_bad_comparison,
        )

    if not isinstance(outputs, (ttnn.Tensor, torch.Tensor)):
        raise TypeError(f"Expected a tensor output, got {type(outputs)}")
    if not isinstance(golden_outputs, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor golden output, got {type(golden_outputs)}")

    output = outputs
    golden_output = golden_outputs
    torch_output = output if isinstance(output, torch.Tensor) else to_torch_for_comparison(output, golden_output)
    same_shape = golden_output.shape == torch_output.shape
    comparison_config = getattr(golden_output, "_ttnn_comparison_config", None)
    comparison_golden = golden_output
    comparison_output = torch_output
    if comparison_config is not None and comparison_config.mask is not None and same_shape:
        comparison_mask = comparison_config.mask.to(dtype=torch.bool, device=golden_output.device)
        while comparison_mask.ndim < golden_output.ndim:
            comparison_mask = comparison_mask.unsqueeze(-1)
        try:
            comparison_mask = torch.broadcast_to(comparison_mask, golden_output.shape)
        except RuntimeError as error:
            raise ValueError(
                f"Golden comparison mask shape {tuple(comparison_config.mask.shape)} cannot be broadcast "
                f"to output shape {tuple(golden_output.shape)}"
            ) from error
        comparison_golden = golden_output[comparison_mask]
        comparison_output = torch_output[comparison_mask]

    flattened_golden = comparison_golden.reshape(-1)
    flattened_output = comparison_output.reshape(-1)

    def is_constant(flattened_tensor):
        """Return whether a flattened tensor contains one repeated value.
        Treats empty and all-NaN tensors as constant for PCC routing.
        """

        if flattened_tensor.numel() == 0:
            return True
        first_value = flattened_tensor[0]
        # NaN never compares equal to itself, so direct equality misses all-NaN constants.
        # Recognize that case explicitly before checking ordinary constant values.
        if flattened_tensor.dtype.is_floating_point or flattened_tensor.dtype.is_complex:
            if bool(torch.isnan(first_value)):
                return bool(torch.all(torch.isnan(flattened_tensor)))
        return bool(torch.all(flattened_tensor == first_value))

    pcc_is_degenerate = (
        flattened_golden.numel() < 2
        or flattened_output.numel() < 2
        or is_constant(flattened_golden)
        or is_constant(flattened_output)
    )
    use_comparison_config = comparison_config is not None and (comparison_config.scope == "all" or pcc_is_degenerate)
    # Operation goldens opt into non-PCC metrics only where their numerical contract requires it.
    # Unmarked outputs retain the existing PCC and degenerate allclose behavior without relaxation.
    if use_comparison_config and comparison_config.method == "skip":
        return []

    if use_comparison_config and same_shape:
        nonfinite_masks_match = True
        if comparison_config.nonfinite == "mask" and (
            comparison_golden.dtype.is_floating_point
            or comparison_golden.dtype.is_complex
            or comparison_output.dtype.is_floating_point
            or comparison_output.dtype.is_complex
        ):
            golden_finite = torch.isfinite(comparison_golden)
            output_finite = torch.isfinite(comparison_output)
            nonfinite_masks_match = bool(torch.equal(golden_finite, output_finite))
            if nonfinite_masks_match and not bool(golden_finite.all()):
                comparison_golden = comparison_golden.clone()
                comparison_output = comparison_output.clone()
                comparison_golden[~golden_finite] = 0
                comparison_output[~output_finite] = 0

        if not nonfinite_masks_match:
            matches = False
        elif comparison_config.method == "ulp":
            matches, _ = comp_ulp(
                comparison_golden,
                comparison_output,
                ulp_threshold=comparison_config.ulp_threshold,
                allow_nonfinite=True,
            )
            matches = bool(matches)
        else:
            if comparison_golden.dtype != comparison_output.dtype:
                comparison_output = comparison_output.to(comparison_golden.dtype)
            matches = bool(
                torch.allclose(
                    comparison_golden,
                    comparison_output,
                    rtol=comparison_config.rtol,
                    atol=comparison_config.atol,
                    equal_nan=comparison_config.equal_nan,
                )
            )
        actual_pcc = 1.0 if matches else 0.0
    elif use_comparison_config:
        matches = False
        actual_pcc = 0.0
    elif pcc_is_degenerate:
        if golden_output.dtype != torch_output.dtype:
            torch_output = torch_output.to(golden_output.dtype)
        matches = same_shape and bool(torch.allclose(golden_output, torch_output, rtol=1e-5, atol=1e-4, equal_nan=True))
        actual_pcc = 1.0 if matches else 0.0
    else:
        matches, actual_pcc = comp_pcc(comparison_golden, comparison_output, desired_pcc)

    mesh_coord = getattr(output, "_ttnn_mesh_coord", None)
    comparison_record = {
        "tensor_id": int(output.tensor_id),
        "golden_tensor_id": int(golden_output.tensor_id),
        "mesh_coord": mesh_coord,
        "matches": bool(matches),
        "desired_pcc": float(desired_pcc),
        "actual_pcc": float(actual_pcc),
    }
    if not matches:
        mesh_coord_message = f" at mesh coordinate {mesh_coord}" if mesh_coord is not None else ""
        error_message = f"{python_fully_qualified_name}: Comparing output tensor 0{mesh_coord_message} against CPU {level} failed: pcc is {actual_pcc} but should be >={desired_pcc}"
        if fail_on_bad_comparison:
            raise RuntimeError(error_message)
        logger.error(error_message)
    return [comparison_record]


PRE_OPERATION_HOOKS = []
POST_OPERATION_HOOKS = []

push_current_command_queue_id_for_thread = ttnn._ttnn.core.push_current_command_queue_id_for_thread
pop_current_command_queue_id_for_thread = ttnn._ttnn.core.pop_current_command_queue_id_for_thread
get_current_command_queue_id_for_thread = ttnn._ttnn.core.get_current_command_queue_id_for_thread


@contextmanager
def register_pre_operation_hook(hook):
    """

    register_pre_operation_hook is a context manager that registers a pre-operation hook. The hook can be used to run custom code before the operation is executed.

    Args:
        operation: The operation that is being called.
        args: The arguments that are passed to the operation.
        kwargs: The keyword arguments that are passed to the operation.

    Returns:
        `None`: the hook is executed.

    """

    global PRE_OPERATION_HOOKS
    PRE_OPERATION_HOOKS.append(hook)
    yield
    PRE_OPERATION_HOOKS.pop()


@contextmanager
def command_queue(cq_id: int):
    """Context manager to set a default command queue for all TTNN operations within this context.

    Operations within this context will use the specified cq_id unless they explicitly
    provide their own cq_id parameter, which takes precedence.

    Args:
        cq_id: The command queue ID to use for operations in this context

    Example:
        with ttnn.command_queue(1):
            result = ttnn.some_operation(tensor)  # Will use cq_id 1
            result2 = ttnn.other_operation(tensor, queue_id=0)  # Will use cq_id 0 (overrides context)
    """
    if cq_id is None:
        raise ValueError("cq_id cannot be None in command_queue context")

    push_current_command_queue_id_for_thread(cq_id)
    try:
        yield
    finally:
        # Check if command queue is in expected state when exiting context
        current_cq_id = get_current_command_queue_id_for_thread()
        if current_cq_id != cq_id:
            logger.warning(
                f"command_queue({cq_id}) context exiting with unexpected command queue ID: {current_cq_id}. "
                f"This might indicate an operation didn't properly restore the command queue state. "
                f"Restoring to original value {cq_id}."
            )
        pop_current_command_queue_id_for_thread()


@contextmanager
def register_post_operation_hook(hook):
    """

    register_post_operation_hook is a context manager that registers a post-operation hook. The hook can be used to run custom code after the operation is executed.

    Args:
        operation: The operation that is being called.
        args: The arguments that are passed to the operation.
        kwargs: The keyword arguments that are passed to the operation.
        output: The output of the operation.

    Returns:
        `None`: the hook is executed.

    """

    global POST_OPERATION_HOOKS
    POST_OPERATION_HOOKS.append(hook)
    yield
    POST_OPERATION_HOOKS.pop()


def get_devices(object_value):
    devices = set()
    if isinstance(object_value, ttnn.Tensor):
        if ttnn.is_tensor_storage_on_device(object_value) and object_value.is_allocated():
            devices.update(object_value.devices())
    elif isinstance(object_value, ttnn.Device):
        devices.add(object_value)
    elif isinstance(object_value, (list, tuple)):
        for element in object_value:
            devices |= get_devices(element)
    elif isinstance(object_value, dict):
        for value in object_value.values():
            devices |= get_devices(value)
    return devices


def get_tensors(object_value, tensor_type):
    tensors = []
    if isinstance(object_value, tensor_type):
        tensors.append(object_value)
    elif isinstance(object_value, ttnn.DistributedGolden):
        if object_value.global_value is not None:
            tensors += get_tensors(object_value.global_value, tensor_type)
        if object_value.shards is not None:
            tensors += get_tensors(object_value.shards, tensor_type)
    elif isinstance(object_value, (list, tuple)):
        for element in object_value:
            tensors += get_tensors(element, tensor_type)
    elif isinstance(object_value, dict):
        for value in object_value.values():
            tensors += get_tensors(value, tensor_type)
    return tensors


def get_ttnn_tensors(object_value):
    return get_tensors(object_value, ttnn.Tensor)


def get_all_tensors(object_value):
    import torch

    return get_tensors(object_value, (ttnn.Tensor, torch.Tensor))


def should_compare_tensor_outputs(golden_outputs, outputs):
    """Return whether either output structure contains tensor values.
    Selects the tensor comparison path when one side exposes tensors.
    """

    # Keep tensor and scalar comparison paths separate so each can preserve its report contract.
    return bool(get_all_tensors(golden_outputs)) or bool(get_all_tensors(outputs))


def should_compare_scalar_outputs(golden_outputs, outputs):
    """Return whether corresponding leaves in two output structures are all numeric scalars."""

    import numbers

    try:
        output_pairs = _structured_output_pairs(golden_outputs, outputs)
    except (TypeError, ValueError):
        return False
    return bool(output_pairs) and all(
        isinstance(golden_output, numbers.Number) and isinstance(output, numbers.Number)
        for golden_output, output in output_pairs
    )


def compare_scalar_outputs(
    python_fully_qualified_name, golden_output, output, desired_pcc, level, fail_on_bad_comparison
):
    """Compare scalar operation output and record tensor-compatible metadata.
    Uses exact equality for integers and allclose semantics for other numbers.
    """

    import numbers
    import torch

    golden_tensor = torch.as_tensor(golden_output)
    output_tensor = torch.as_tensor(output)
    set_tensor_id(golden_tensor)
    set_tensor_id(output_tensor)

    if isinstance(golden_output, numbers.Integral) and isinstance(output, numbers.Integral):
        matches = golden_output == output
    else:
        common_dtype = torch.promote_types(golden_tensor.dtype, output_tensor.dtype)
        matches = bool(
            torch.allclose(
                golden_tensor.to(common_dtype),
                output_tensor.to(common_dtype),
                rtol=1e-5,
                atol=1e-4,
                equal_nan=True,
            )
        )

    comparison_record = {
        "tensor_id": int(output_tensor.tensor_id),
        "golden_tensor_id": int(golden_tensor.tensor_id),
        "matches": bool(matches),
        "desired_pcc": float(desired_pcc),
        "actual_pcc": 1.0 if matches else 0.0,
    }
    ttnn.graph.record_tensor_comparison_data(
        tensors=[get_tensor_report_record(output_tensor), get_tensor_report_record(golden_tensor)]
    )

    if not matches:
        error_message = (
            f"{python_fully_qualified_name}: Comparing scalar output against CPU {level} failed: "
            f"expected {golden_output!r}, got {output!r}"
        )
        if fail_on_bad_comparison:
            raise RuntimeError(error_message)
        logger.error(error_message)

    return [comparison_record]


def set_tensor_id(tensor, force=False):
    import torch

    if isinstance(tensor, (ttnn.Tensor, torch.Tensor)):
        if not force and hasattr(tensor, "tensor_id") and tensor.tensor_id is not None:
            return
        tensor.tensor_id = ttnn._ttnn.fetch_and_increment_tensor_id()
    elif isinstance(tensor, ttnn.DistributedGolden):
        if tensor.global_value is not None:
            set_tensor_id(tensor.global_value, force)
        if tensor.shards is not None:
            for shard in tensor.shards.values():
                set_tensor_id(shard, force)
    elif isinstance(tensor, (list, tuple)):
        for element in tensor:
            set_tensor_id(element, force)
    else:
        raise RuntimeError(f"Unsupported input to set_tensor_id: {type(tensor)}")


def get_output_tensor_ids(output):
    """Return the list of tensor_id ints from all tensors in *output*.

    Tensor IDs must already be assigned (via ``set_tensor_id``).
    """
    ids = []
    for t in get_all_tensors(output):
        tid = getattr(t, "tensor_id", None)
        if tid is not None:
            ids.append(int(tid))
    return ids


def _convert_ttnn_to_torch_for_comparison(tensor):
    if tensor.dtype == ttnn.DataType.FP8_E4M3:
        # Torch 2.7 cannot import FP8 DLPack tensors; compare through host FLOAT32 instead.
        # This matches the FP8 golden's dequantized torch.float32 representation.
        if ttnn.is_tensor_storage_on_device(tensor):
            tensor = ttnn.from_device(tensor)
        tensor = ttnn.to_dtype(tensor, ttnn.float32)
    return ttnn.to_torch(tensor)


def distributed_golden_for_comparison(tensor, golden_value=None):
    """Represent a mesh tensor as a global value plus coordinate-keyed Torch shards."""

    try:
        device_tensors = list(ttnn.get_device_tensors(tensor))
        topology = tensor.tensor_topology()
    except (RuntimeError, TypeError) as error:
        raise ValueError("Distributed golden conversion requires a tensor with mesh topology") from error

    if not device_tensors:
        raise ValueError("Distributed golden conversion requires at least one mesh shard")
    topology = ttnn.TensorTopologySnapshot.from_topology(topology)
    mesh_coords = ttnn.mesh_coords_for_shards(topology, len(device_tensors), tensor.device())
    shard_shapes_by_mesh_coord = {
        mesh_coord: tuple(int(dimension) for dimension in device_tensor.shape)
        for mesh_coord, device_tensor in zip(mesh_coords, device_tensors)
    }

    if isinstance(golden_value, ttnn.DistributedGolden):
        return golden_value
    if golden_value is not None:
        shards = ttnn.decompose_mesh_value(
            golden_value,
            topology=topology,
            shard_shapes_by_mesh_coord=shard_shapes_by_mesh_coord,
        )
        return ttnn.DistributedGolden(
            topology=topology,
            global_value=golden_value,
            shards=shards,
            compare_coords=frozenset(mesh_coords),
        )

    shards = {
        mesh_coord: _convert_ttnn_to_torch_for_comparison(device_tensor)
        for mesh_coord, device_tensor in zip(mesh_coords, device_tensors)
    }
    try:
        global_value = ttnn.compose_mesh_value(shards_by_mesh_coord=shards, topology=topology)
    except ttnn.MeshValueIncompleteError:
        global_value = None
    return ttnn.DistributedGolden(
        topology=topology,
        global_value=global_value,
        shards=shards,
        compare_coords=frozenset(mesh_coords),
    )


def _distributed_golden_global_value(distributed_golden):
    if distributed_golden.global_value is not None:
        return distributed_golden.global_value
    if distributed_golden.shards is None:
        raise ValueError("Distributed golden does not contain a global value or shards")
    return ttnn.compose_mesh_value(
        shards_by_mesh_coord=distributed_golden.shards,
        topology=distributed_golden.topology,
    )


def to_torch_for_comparison(tensor, golden_tensor=None):
    import math
    import torch

    if isinstance(tensor, torch.Tensor):
        return tensor
    if not isinstance(tensor, ttnn.Tensor):
        raise RuntimeError(f"Unsupported tensor type for comparison: {type(tensor)}")

    topology = ttnn.TensorTopologySnapshot.from_topology(tensor.tensor_topology())
    if math.prod(topology.distribution_shape) == 1:
        return _convert_ttnn_to_torch_for_comparison(tensor)
    return _distributed_golden_global_value(distributed_golden_for_comparison(tensor))


def _distributed_comparison_pairs(golden, output):
    import torch

    if not isinstance(output, ttnn.Tensor):
        if golden.shards is not None or golden.compare_coords is not None:
            raise TypeError("Per-coordinate distributed comparison requires a TTNN mesh tensor output")
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected a tensor output for DistributedGolden, got {type(output)}")
        return [(_distributed_golden_global_value(golden), output)]

    device_tensors = list(ttnn.get_device_tensors(output))
    output_topology = output.tensor_topology()
    output_coords = ttnn.mesh_coords_for_shards(output_topology, len(device_tensors), output.device())
    actual_shards_by_key = {
        tuple(int(value) for value in mesh_coord): _convert_ttnn_to_torch_for_comparison(device_tensor)
        for mesh_coord, device_tensor in zip(output_coords, device_tensors)
    }
    shard_shapes = {
        mesh_coord: tuple(actual_shards_by_key[tuple(int(value) for value in mesh_coord)].shape)
        for mesh_coord in golden.topology.mesh_coords
        if tuple(int(value) for value in mesh_coord) in actual_shards_by_key
    }
    expected_shards = golden.shards
    if expected_shards is None:
        expected_shards = ttnn.decompose_mesh_value(
            golden.global_value,
            topology=golden.topology,
            shard_shapes_by_mesh_coord=shard_shapes,
        )

    compare_coords_are_explicit = golden.compare_coords is not None
    compare_coords = golden.compare_coords if compare_coords_are_explicit else frozenset(expected_shards)
    pairs = []
    for mesh_coord in golden.topology.mesh_coords:
        if mesh_coord not in compare_coords:
            continue
        mesh_coord_key = tuple(int(value) for value in mesh_coord)
        if mesh_coord not in expected_shards:
            raise ValueError(f"Distributed golden has no shard for comparison coordinate {mesh_coord_key}")
        if mesh_coord_key not in actual_shards_by_key:
            if compare_coords_are_explicit:
                raise ValueError(f"Runtime output has no shard for comparison coordinate {mesh_coord_key}")
            continue
        actual_shard = actual_shards_by_key[mesh_coord_key]
        actual_shard.tensor_id = output.tensor_id
        actual_shard._ttnn_mesh_coord = mesh_coord_key
        pairs.append((expected_shards[mesh_coord], actual_shard))
    return pairs


def _structured_output_pairs(golden_outputs, outputs):
    import torch

    if isinstance(outputs, (ttnn.Tensor, torch.Tensor)) and isinstance(golden_outputs, (list, tuple)):
        # Backward goldens commonly return a one-element list even when the runtime returns one tensor.
        # Unwrap only this unambiguous singleton shape; preserve list handling for true multi-output operations.
        if len(golden_outputs) != 1:
            raise ValueError(
                f"Output structure mismatch: tensor output cannot be paired with {len(golden_outputs)} golden outputs"
            )
        golden_outputs = golden_outputs[0]

    def pairs(golden_output, output):
        if golden_output is None or output is None:
            if golden_output is None and output is None:
                return
            raise TypeError(
                f"Output structure mismatch: golden type {type(golden_output)} does not match output type {type(output)}"
            )
        if isinstance(golden_output, (list, tuple)) or isinstance(output, (list, tuple)):
            if not isinstance(golden_output, (list, tuple)) or not isinstance(output, (list, tuple)):
                raise TypeError(
                    f"Output structure mismatch: golden type {type(golden_output)} "
                    f"does not match output type {type(output)}"
                )
            if len(golden_output) != len(output):
                raise ValueError(
                    f"Output structure mismatch: golden has {len(golden_output)} elements "
                    f"but output has {len(output)}"
                )
            for nested_golden, nested_output in zip(golden_output, output):
                yield from pairs(nested_golden, nested_output)
            return
        if isinstance(golden_output, dict) or isinstance(output, dict):
            if not isinstance(golden_output, dict) or not isinstance(output, dict):
                raise TypeError(
                    f"Output structure mismatch: golden type {type(golden_output)} "
                    f"does not match output type {type(output)}"
                )
            if golden_output.keys() != output.keys():
                raise ValueError("Output structure mismatch: golden and output dictionaries have different keys")
            for key in golden_output:
                yield from pairs(golden_output[key], output[key])
            return
        yield golden_output, output

    return tuple(pairs(golden_outputs, outputs))


def get_tensor_report_record(tensor):
    import torch

    if isinstance(tensor, ttnn.Tensor):
        device_id = None
        address = None
        memory_config = None
        buffer_type = None
        if ttnn.has_storage_type_of(tensor, ttnn.DEVICE_STORAGE_TYPE) and tensor.is_allocated():
            memory_config = ttnn.get_memory_config(tensor)
            device_id = tensor.device().id()
            address = tensor.buffer_address()
            buffer_type = memory_config.buffer_type.value

        return {
            "tensor_id": int(tensor.tensor_id),
            "shape": str(tensor.shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "memory_config": str(memory_config) if memory_config is not None else None,
            "device_id": device_id,
            "address": address,
            "buffer_type": buffer_type,
        }

    if isinstance(tensor, torch.Tensor):
        return {
            "tensor_id": int(tensor.tensor_id),
            "shape": str(tensor.shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "memory_config": None,
            "device_id": None,
            "address": None,
            "buffer_type": None,
        }

    raise RuntimeError(f"Unsupported tensor report record type: {type(tensor)}")


def set_output_tensor_id_decorator(function):
    @wraps(function)
    def call_wrapper(*function_args, **function_kwargs):
        output = function(*function_args, **function_kwargs)
        output_tensors = get_all_tensors(output)
        set_tensor_id(output_tensors, force=True)
        return output

    return call_wrapper


OPERATION_CALL_STACK = []


def default_preprocess_golden_function_inputs(function_args, function_kwargs):
    def recursive_preprocess_golden_function_inputs(object_value):
        if isinstance(object_value, ttnn.Tensor):
            return to_torch_for_comparison(object_value)
        elif isinstance(object_value, (list, tuple)):
            new_object_value = [recursive_preprocess_golden_function_inputs(element) for element in object_value]
            return type(object_value)(new_object_value)
        else:
            return object_value

    new_args = []
    for arg in function_args:
        new_arg = recursive_preprocess_golden_function_inputs(arg)
        new_args.append(new_arg)
    new_kwargs = {}
    for key, value in function_kwargs.items():
        new_value = recursive_preprocess_golden_function_inputs(value)
        new_kwargs[key] = new_value
    return tuple(new_args), new_kwargs


def prepare_backward_golden_inputs(function_args_and_kwargs):
    """Prepare nested Torch inputs for comparison-mode backward goldens.
    Clears gradients and enables autograd on floating-point and complex tensors.
    """

    import torch

    def prepare(object_value):
        """Recursively enable autograd on eligible tensors in an input structure.
        Preserves list, tuple, and dictionary containers while replacing their values.
        """

        if isinstance(object_value, torch.Tensor) and (
            object_value.dtype.is_floating_point or object_value.dtype.is_complex
        ):
            object_value.grad = None
            object_value.requires_grad_(True)
        elif isinstance(object_value, (list, tuple)):
            object_value = type(object_value)(prepare(element) for element in object_value)
        elif isinstance(object_value, dict):
            object_value = {key: prepare(value) for key, value in object_value.items()}
        return object_value

    args, kwargs = function_args_and_kwargs
    # Backward goldens need autograd-enabled references even in comparison mode.
    return prepare(args), prepare(kwargs)


def default_postprocess_golden_function_outputs(output, function_args, function_kwargs):
    input_tensors = get_ttnn_tensors((function_args, function_kwargs))

    input_dtype = None
    input_layout = None
    input_device = None
    if input_tensors:
        input_tensor, *_ = input_tensors
        input_dtype = input_tensor.dtype
        input_layout = input_tensor.layout
        if ttnn.is_tensor_storage_on_device(input_tensor):
            input_device = input_tensor.device()

    def recursive_postprocess_golden_function_outputs(output):
        import torch

        if isinstance(output, torch.Tensor):
            return ttnn.from_torch(output, dtype=input_dtype, layout=input_layout, device=input_device)
        elif isinstance(output, (list, tuple)):
            new_output = [recursive_postprocess_golden_function_outputs(element) for element in output]
            return type(output)(new_output)
        else:
            raise RuntimeError(f"Unsupported output type: {type(output)}")

    output = recursive_postprocess_golden_function_outputs(output)
    return output


TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR = {}
TENSOR_IDS_PRODUCED_BY_OPERATION = set()


def preprocess_global_golden_function_inputs(function_args, function_kwargs, *, local_golden_inputs=None):
    """Resolve stored global goldens while preserving each operation's local preprocessing shape.

    A matching ``local_value`` is the locally preprocessed argument, not a device-local shard.
    Its type determines whether a cached mesh golden stays distributed for the global golden.
    """

    if ttnn.CONFIG.report_path is None:
        return None
    input_index = 0
    local_args, local_kwargs = local_golden_inputs or ((), {})

    def preprocess(object_value, local_value=None):
        nonlocal input_index
        if isinstance(object_value, ttnn.Tensor):
            if object_value.tensor_id is None:
                raise RuntimeError("Input tensor does not have a tensor_id")
            if object_value.tensor_id not in TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR:
                if object_value.tensor_id in TENSOR_IDS_PRODUCED_BY_OPERATION:
                    logger.warning(
                        f"Intermediate tensor with tensor_id {object_value.tensor_id} (input index: {input_index}) is not found in the global golden tensors. Global golden will be skipped"
                    )
                    raise RuntimeError("Intermediate tensor is not found in the global golden tensors")
                else:
                    logger.warning(
                        f"Input tensor with tensor_id {object_value.tensor_id} (input index: {input_index})  is not found in the global golden tensors. Creating it from ttnn tensor."
                    )
                    golden_tensor = to_torch_for_comparison(object_value)
            else:
                golden_tensor = TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[object_value.tensor_id]
            input_index += 1
            if isinstance(local_value, ttnn.DistributedGolden):
                return distributed_golden_for_comparison(object_value, golden_tensor)
            if isinstance(golden_tensor, ttnn.DistributedGolden):
                return _distributed_golden_global_value(golden_tensor)
            return golden_tensor
        if isinstance(object_value, ttnn.Shape):
            return tuple(object_value)
        if isinstance(object_value, (list, tuple)):
            local_values = local_value if isinstance(local_value, (list, tuple)) else ()
            return type(object_value)(
                preprocess(element, local_values[index] if index < len(local_values) else None)
                for index, element in enumerate(object_value)
            )
        return object_value

    try:
        return (
            [
                preprocess(arg, local_args[index] if index < len(local_args) else None)
                for index, arg in enumerate(function_args)
            ],
            {key: preprocess(value, local_kwargs.get(key)) for key, value in function_kwargs.items()},
        )
    except Exception as error:
        logger.warning(f"Failed to preprocess global golden function inputs: {error}")
        return None


def postprocess_global_golden_function_outputs(outputs, golden_outputs):
    import numbers
    import torch

    for golden_output, output in _structured_output_pairs(golden_outputs, outputs):
        if isinstance(golden_output, numbers.Number) or isinstance(output, numbers.Number):
            if isinstance(golden_output, numbers.Number) and isinstance(output, numbers.Number):
                continue
            raise TypeError(
                f"Output structure mismatch: golden type {type(golden_output)} does not match output type {type(output)}"
            )
        if not isinstance(output, (ttnn.Tensor, torch.Tensor)):
            raise TypeError(f"Expected a tensor output, got {type(output)}")
        if not isinstance(golden_output, (torch.Tensor, ttnn.DistributedGolden)):
            raise TypeError(f"Expected torch.Tensor or DistributedGolden, got {type(golden_output)}")
        if output.tensor_id is None:
            raise RuntimeError("Output tensor does not have a tensor_id")
        if isinstance(output, ttnn.Tensor) and not isinstance(golden_output, ttnn.DistributedGolden):
            device_tensor_count = len(ttnn.get_device_tensors(output))
            if device_tensor_count > 1:
                golden_output = distributed_golden_for_comparison(output, golden_output)
        TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[output.tensor_id] = _clone_golden_value(golden_output)


def _clone_golden_value(value):
    """Clone a golden storage boundary while retaining recognized comparison metadata."""

    import torch

    if isinstance(value, torch.Tensor):
        return _copy_golden_comparison_config(value, value.clone())
    if isinstance(value, ttnn.DistributedGolden):
        global_value = _clone_golden_value(value.global_value) if value.global_value is not None else None
        shards = (
            {mesh_coord: _clone_golden_value(shard) for mesh_coord, shard in value.shards.items()}
            if value.shards is not None
            else None
        )
        return ttnn.DistributedGolden(
            topology=value.topology,
            global_value=global_value,
            shards=shards,
            compare_coords=value.compare_coords,
        )
    raise TypeError(f"Unsupported global golden value type: {type(value)}")


if TRACE_ALLOC_DIAGNOSTICS:

    def _drain_traceback_ids(source="op_end", op_name=None):
        """Drain allocation IDs and capture their Python call stacks."""
        from ttnn._ttnn.operations.trace import drain_pending_traceback_ids, drain_retired_traceback_ids
        from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker

        pending = drain_pending_traceback_ids()
        retired = set(drain_retired_traceback_ids())
        for buf_id in retired:
            UnsafeAllocationTracker._tracebacks.pop(buf_id, None)
        pending = [buf_id for buf_id in pending if buf_id not in retired]
        if not pending:
            return
        import traceback as _tb

        # Drop the tracker wrapper frames so the traceback ends at the model call site.
        stack = "".join(_tb.format_stack()[:-2])
        if source == "op_start":
            marker = (
                "[trace alloc tracker] pending traceback IDs were flushed at op entry; "
                "allocation likely happened outside a wrapped op"
            )
            if op_name:
                marker += f" before '{op_name}'"
            marker += ".\n"
            stack = marker + stack
        for buf_id in pending:
            UnsafeAllocationTracker._tracebacks[buf_id] = stack


# Keyword argument names through which an operation writes into a caller-supplied tensor in
# place; the tensor's contents are overwritten so any pre-existing global golden becomes stale.
INPLACE_OUTPUT_KWARG_NAMES = (
    "output_tensor",
    "optional_tensor",
    "optional_output_tensor",
    "out",
    "output",
    "output_tensors",
    "optional_output_tensors",
)


def get_inplace_output_tensors(function_kwargs):
    tensors = []
    for name in INPLACE_OUTPUT_KWARG_NAMES:
        if name in function_kwargs:
            tensors += get_ttnn_tensors(function_kwargs[name])
    return tensors


def refresh_or_invalidate_global_goldens(inplace_tensors, global_golden_output):
    # Re-key the fresh golden onto a caller tensor an op wrote in place, or drop its stale
    # entry when no golden exists so the next read rebuilds the golden from device data.
    import torch

    if not inplace_tensors:
        return
    golden_tensors = get_tensors(global_golden_output, torch.Tensor) if global_golden_output is not None else []
    for index, tensor in enumerate(inplace_tensors):
        tensor_id = getattr(tensor, "tensor_id", None)
        if tensor_id is None:
            continue
        if len(golden_tensors) == len(inplace_tensors):
            # Keep caller-buffer goldens behind the same storage boundary as ordinary outputs.
            # Copy the operation-owned comparison contract so a later read retains its tolerance.
            golden_clone = golden_tensors[index].clone()
            TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR[tensor_id] = _copy_golden_comparison_config(
                golden_tensors[index], golden_clone
            )
        else:
            TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(tensor_id, None)


def _merge_local_golden_metadata_into_global_inputs(local_inputs, global_inputs):
    """Copy operation-specific preprocessing metadata into cached global inputs.
    Also normalizes keyword aliases before the global golden is evaluated.
    """

    _, local_kwargs = local_inputs
    _, global_kwargs = global_inputs
    argument_aliases = local_kwargs.get("_ttnn_golden_argument_aliases", {})
    for alias, canonical_name in argument_aliases.items():
        if alias in global_kwargs and canonical_name not in global_kwargs:
            global_kwargs[canonical_name] = global_kwargs.pop(alias)
    global_kwargs.update({key: value for key, value in local_kwargs.items() if key.startswith("_ttnn_")})
    return global_inputs


@dataclasses.dataclass
class FastOperation:
    python_fully_qualified_name: str
    function: Callable
    preprocess_golden_function_inputs: Callable
    golden_function: Callable
    postprocess_golden_function_outputs: Callable
    is_cpp_operation: bool
    is_experimental: bool
    _slow_operation: "Operation | None" = dataclasses.field(default=None, init=False, repr=False)

    @property
    def __name__(self):
        return self.python_fully_qualified_name

    def _requires_slow_runtime(self) -> bool:
        return (
            not ttnn.CONFIG.enable_fast_runtime_mode or ttnn.CONFIG.enable_comparison_mode or ttnn.CONFIG.enable_logging
        )

    def _slow_operation_instance(self) -> "Operation":
        if self._slow_operation is None:
            self._slow_operation = Operation(
                python_fully_qualified_name=self.python_fully_qualified_name,
                function=self.function,
                golden_function=self.golden_function,
                preprocess_golden_function_inputs=self.preprocess_golden_function_inputs,
                postprocess_golden_function_outputs=self.postprocess_golden_function_outputs,
                is_cpp_operation=self.is_cpp_operation,
                is_experimental=self.is_experimental,
            )
            self._slow_operation.__post_init__()
        return self._slow_operation

    def __gt__(self, other):
        return self.python_fully_qualified_name < other.python_fully_qualified_name

    def __hash__(self):
        return hash(self.python_fully_qualified_name)

    def _enhance_type_error_message(self, original_error, function_args, function_kwargs):
        """
        Parse pybind11/nanobind TypeError and create a concise error message showing
        how the function was called vs available signatures.
        Returns None if this is not a pybind11/nanobind type error that we can enhance.
        """
        import re

        # Only enhance pybind11/nanobind type errors
        if not (
            "incompatible function arguments" in original_error
            and ("Invoked with:" in original_error or "Invoked with types:" in original_error)
        ):
            return None

        def clean(s):
            return s.replace("ttnn._ttnn.tensor.", "ttnn.").replace("ttnn._ttnn.operations.", "ttnn.")

        def pretty_type(v):
            t = type(v)
            if t.__module__ == "builtins":
                return t.__name__
            return clean(f"{t.__module__}.{t.__name__}")

        name = self.python_fully_qualified_name.split(".")[-1]

        # Parse signatures from the original error message
        sigs = []
        lines = original_error.split("\n")
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if re.match(r"^\d+\.", stripped):  # Lines starting with "1.", "2.", etc.
                # Found a signature, combine lines until we find the return type
                sig_lines = []
                j = i
                while j < len(lines):
                    sig_lines.append(lines[j].strip())
                    if "-> " in lines[j]:  # Found the return type, signature is complete
                        break
                    j += 1
                signature = " ".join(sig_lines)

                # Extract just the parameters from the signature
                match = re.search(r"__call__\(self,\s*(.+?)\)\s*->", signature)
                if not match:
                    match = re.search(r"\(self:[^,]+,\s*(.+?)\)\s*->", signature)

                if match:
                    params_str = clean(match.group(1).strip())
                    sigs.append(f"{name}({params_str})")

                i = j + 1
            else:
                i += 1

        # Format the call with actual argument types
        args = [pretty_type(a) for a in function_args]
        kwargs = [
            f"{k}={v!r}" if isinstance(v, (bool, str, type(None))) else f"{k}={v}" for k, v in function_kwargs.items()
        ]
        called = f"{name}({', '.join(args + kwargs)})"

        return (
            f"\n{self.python_fully_qualified_name}(): incompatible function arguments.\n\n"
            f"Called with:\n  {called}\n\n"
            f"Available signatures:\n  " + "\n  ".join(sigs or ["<unknown>"])
        )

    def __call__(self, *function_args, **function_kwargs):
        if self._requires_slow_runtime():
            return self._slow_operation_instance()(*function_args, **function_kwargs)

        cq_id = None
        if "queue_id" in function_kwargs:
            cq_id = function_kwargs.pop("queue_id")
        elif "cq_id" in function_kwargs:
            cq_id = function_kwargs.pop("cq_id")

        recording = ttnn.graph.is_python_io_recording_enabled()
        started = False
        python_io_record = None
        try:
            if recording:
                ttnn.graph.track_function_start(self.python_fully_qualified_name)
                started = True
                python_io_record = ttnn.graph.append_python_io_record(self.python_fully_qualified_name)
                ttnn.graph.record_python_operation(
                    self.python_fully_qualified_name,
                    function_args,
                    function_kwargs,
                    record=python_io_record,
                )
                input_tensors = get_all_tensors((function_args, function_kwargs))
                set_tensor_id(input_tensors)

            if cq_id is None:
                result = self.function(*function_args, **function_kwargs)
            else:
                with command_queue(cq_id):
                    result = self.function(*function_args, **function_kwargs)
        except TypeError as e:
            enhanced_msg = self._enhance_type_error_message(str(e), function_args, function_kwargs)
            ttnn.graph.record_python_operation_error(python_io_record, "TypeError", enhanced_msg or str(e))
            if enhanced_msg:
                raise TypeError(enhanced_msg) from e
            raise
        except Exception as exception:
            ttnn.graph.record_python_operation_error(python_io_record, type(exception).__name__, str(exception))
            raise
        finally:
            if started:
                ttnn.graph.track_function_end()

        if recording:
            set_tensor_id(get_all_tensors(result), force=True)
            ttnn.graph.store_output_tensor_ids(get_output_tensor_ids(result))

        return result

    def __post_init__(self):
        if self.function.__doc__ is None:
            return

        # Delete the signature line created by nanobind
        docstring_lines = self.function.__doc__.split("\n")
        op_name = self.python_fully_qualified_name.split(".")[-1]
        if f"{op_name}(" in docstring_lines[0]:
            docstring_lines.pop(0)
        self.__doc__ = "\n".join(docstring_lines)

        # # TEMP HACK to read docstring from the file
        # doc_folder = pathlib.Path(__file__).parent.parent.parent / "ops_docs"
        # doc_file = doc_folder / f"{self.python_fully_qualified_name}.md"
        # if doc_file.exists():
        #     with open(doc_file, "r") as f:
        #         self.__doc__ = f.read()


if TRACE_ALLOC_TRACKING:
    from ttnn._ttnn.operations.trace import pop_allocation_context, push_allocation_context

    _untracked_fast_operation_call = FastOperation.__call__

    if TRACE_ALLOC_DIAGNOSTICS:

        @wraps(_untracked_fast_operation_call)
        def _tracked_fast_operation_call(self, *function_args, **function_kwargs):
            if self._requires_slow_runtime():
                return _untracked_fast_operation_call(self, *function_args, **function_kwargs)
            _drain_traceback_ids(source="op_start", op_name=self.python_fully_qualified_name)
            push_allocation_context(self.python_fully_qualified_name)
            try:
                result = _untracked_fast_operation_call(self, *function_args, **function_kwargs)
                _drain_traceback_ids(source="op_end", op_name=self.python_fully_qualified_name)
                return result
            finally:
                pop_allocation_context()

    else:

        @wraps(_untracked_fast_operation_call)
        def _tracked_fast_operation_call(self, *function_args, **function_kwargs):
            if self._requires_slow_runtime():
                return _untracked_fast_operation_call(self, *function_args, **function_kwargs)
            push_allocation_context(self.python_fully_qualified_name)
            try:
                return _untracked_fast_operation_call(self, *function_args, **function_kwargs)
            finally:
                pop_allocation_context()

    FastOperation.__call__ = _tracked_fast_operation_call


@dataclasses.dataclass
class Operation:
    python_fully_qualified_name: str
    function: Callable
    preprocess_golden_function_inputs: Callable
    golden_function: Callable
    postprocess_golden_function_outputs: Callable
    is_cpp_operation: bool
    is_experimental: bool

    @property
    def __name__(self):
        return self.python_fully_qualified_name

    def __gt__(self, other):
        return self.python_fully_qualified_name < other.python_fully_qualified_name

    def __hash__(self):
        return hash(self.python_fully_qualified_name)

    def __post_init__(self):
        # Wrap function for parameter tracing (if tracing enabled)
        function = ttnn.operation_tracer.wrap_function_for_tracing(self.function, self.python_fully_qualified_name)

        self.preprocess_golden_function_inputs = (
            self.preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
        )
        self.postprocess_golden_function_outputs = (
            self.postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
        )

        def comparison_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                import torch

                local_golden_function_args_and_kwargs = None
                global_golden_function_args_and_kwargs = None
                if self.golden_function is not None:
                    try:
                        local_golden_function_args_and_kwargs = self.preprocess_golden_function_inputs(
                            function_args, function_kwargs
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed to preprocess local golden inputs: {e}. "
                            "Local comparison will be skipped"
                        )
                    global_golden_function_args_and_kwargs = preprocess_global_golden_function_inputs(
                        function_args,
                        function_kwargs,
                        local_golden_inputs=local_golden_function_args_and_kwargs,
                    )
                    if (
                        local_golden_function_args_and_kwargs is not None
                        and global_golden_function_args_and_kwargs is not None
                    ):
                        global_golden_function_args_and_kwargs = _merge_local_golden_metadata_into_global_inputs(
                            local_golden_function_args_and_kwargs,
                            global_golden_function_args_and_kwargs,
                        )

                function_return_value = function(*function_args, **function_kwargs)

                local_tensor_comparison_records = []
                global_tensor_comparison_records = []

                if self.golden_function is None:
                    logger.debug(
                        f"{self.python_fully_qualified_name}: Skipping comparison against CPU because golden_function is not provided"
                    )
                    # An op without a golden (e.g. dropout) can still mutate a caller tensor in
                    # place; invalidate its stale global golden so later reads don't mismatch.
                    if ttnn.CONFIG.report_path is not None:
                        refresh_or_invalidate_global_goldens(get_inplace_output_tensors(function_kwargs), None)
                    TENSOR_IDS_PRODUCED_BY_OPERATION.update(get_output_tensor_ids(function_return_value))
                    return function_return_value, (
                        local_tensor_comparison_records,
                        [],
                        global_tensor_comparison_records,
                        [],
                    )

                output = function_return_value

                logger.debug(f"{self.python_fully_qualified_name}: Comparing against CPU")
                local_golden_function_output = None
                if local_golden_function_args_and_kwargs is not None:
                    local_golden_function_args, local_golden_function_kwargs = local_golden_function_args_and_kwargs
                    # Backward goldens call backward() and read input gradients.
                    # Comparison preprocessing otherwise supplies detached Torch inputs.
                    if self.python_fully_qualified_name.endswith("_bw"):
                        local_golden_function_args, local_golden_function_kwargs = prepare_backward_golden_inputs(
                            (local_golden_function_args, local_golden_function_kwargs)
                        )
                    try:
                        local_golden_function_output = self.golden_function(
                            *local_golden_function_args, **local_golden_function_kwargs
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed to compute local golden output: {e}. "
                            "Local comparison will be skipped"
                        )

                global_golden_function_output = None
                if global_golden_function_args_and_kwargs is not None:
                    global_golden_function_args, global_golden_function_kwargs = global_golden_function_args_and_kwargs
                    if getattr(self.golden_function, "_ttnn_mutates_global_inputs", False):
                        # Only marked global goldens may update stored inputs for positional in-place operations.
                        # Local and directly requested goldens remain ordinary out-of-place references.
                        global_golden_function_kwargs["_ttnn_global_golden"] = True
                    # Global backward goldens need the same autograd inputs as local goldens.
                    if self.python_fully_qualified_name.endswith("_bw"):
                        global_golden_function_args, global_golden_function_kwargs = prepare_backward_golden_inputs(
                            (global_golden_function_args, global_golden_function_kwargs)
                        )
                    try:
                        global_golden_function_output = self.golden_function(
                            *global_golden_function_args, **global_golden_function_kwargs
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed to compute global golden output: {e}. "
                            "Global comparison will be skipped"
                        )

                if local_golden_function_output is not None and should_compare_tensor_outputs(
                    local_golden_function_output, output
                ):
                    try:
                        for golden_tensor in get_all_tensors(local_golden_function_output):
                            set_tensor_id(golden_tensor)
                        local_tensor_comparison_records = compare_tensors_using_pcc(
                            self.python_fully_qualified_name,
                            local_golden_function_output,
                            output,
                            desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                            level="locally",
                            fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        local_golden_function_output = None
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed local tensor comparison: {e}. "
                            "Local comparison will be skipped"
                        )
                elif local_golden_function_output is not None and should_compare_scalar_outputs(
                    local_golden_function_output, output
                ):
                    try:
                        local_tensor_comparison_records = compare_tensors_using_pcc(
                            self.python_fully_qualified_name,
                            local_golden_function_output,
                            output,
                            desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                            level="locally",
                            fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        local_golden_function_output = None
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed local scalar comparison: {e}. "
                            "Local comparison will be skipped"
                        )

                if global_golden_function_output is not None and should_compare_tensor_outputs(
                    global_golden_function_output, output
                ):
                    try:
                        for golden_tensor in get_all_tensors(global_golden_function_output):
                            set_tensor_id(golden_tensor)
                        postprocess_global_golden_function_outputs(output, global_golden_function_output)
                        global_tensor_comparison_records = compare_tensors_using_pcc(
                            self.python_fully_qualified_name,
                            global_golden_function_output,
                            output,
                            desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                            level="globally",
                            fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        global_golden_function_output = None
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed global tensor comparison: {e}. "
                            "Global comparison will be skipped"
                        )
                elif global_golden_function_output is not None and should_compare_scalar_outputs(
                    global_golden_function_output, output
                ):
                    try:
                        global_tensor_comparison_records = compare_tensors_using_pcc(
                            self.python_fully_qualified_name,
                            global_golden_function_output,
                            output,
                            desired_pcc=ttnn.CONFIG.comparison_mode_pcc,
                            level="globally",
                            fail_on_bad_comparison=ttnn.CONFIG.comparison_mode_should_raise_exception,
                        )
                    except Exception as e:
                        if ttnn.CONFIG.comparison_mode_should_raise_exception:
                            raise
                        global_golden_function_output = None
                        logger.warning(
                            f"{self.python_fully_qualified_name}: Failed global scalar comparison: {e}. "
                            "Global comparison will be skipped"
                        )

                # An in-place op (optional_tensor/output_tensor) returns a new wrapper, so re-key
                # the fresh global golden onto the caller's tensor to keep later reads consistent.
                if ttnn.CONFIG.report_path is not None:
                    refresh_or_invalidate_global_goldens(
                        get_inplace_output_tensors(function_kwargs), global_golden_function_output
                    )

                if isinstance(local_golden_function_output, torch.Tensor):
                    local_golden_function_output = [local_golden_function_output]
                if isinstance(global_golden_function_output, torch.Tensor):
                    global_golden_function_output = [global_golden_function_output]

                TENSOR_IDS_PRODUCED_BY_OPERATION.update(get_output_tensor_ids(output))

                return function_return_value, (
                    local_tensor_comparison_records,
                    local_golden_function_output,
                    global_tensor_comparison_records,
                    global_golden_function_output,
                )

            return call_wrapper

        def runtime_decorator(function):
            @wraps(function)
            def call_wrapper(*function_args, **function_kwargs):
                operation_id = ttnn._ttnn.get_python_operation_id()
                is_top_level_operation = len(OPERATION_CALL_STACK) == 1

                decorated_function = function

                if not is_top_level_operation:
                    return decorated_function(*function_args, **function_kwargs)

                cq_id = None
                if "queue_id" in function_kwargs:
                    cq_id = function_kwargs.pop("queue_id")
                elif "cq_id" in function_kwargs:
                    cq_id = function_kwargs.pop("cq_id")

                for hook in PRE_OPERATION_HOOKS:
                    hook_return_value = hook(self, function_args, function_kwargs)
                    if hook_return_value is not None:
                        raise RuntimeError(
                            f"Pre-operation hook {hook} returned {hook_return_value} but must return None"
                        )

                if ttnn.CONFIG.enable_logging and ttnn.CONFIG.enable_graph_report:
                    if not ttnn.tracer.is_tracing_enabled():
                        ttnn.tracer.enable_tracing()

                if ttnn.tracer.ENABLE_TRACER:
                    decorated_function = ttnn.tracer.trace_ttnn_operation(
                        self.python_fully_qualified_name, decorated_function
                    )

                python_io_record = None
                if ttnn.graph.is_python_io_recording_enabled():
                    python_io_record = ttnn.graph.append_python_io_record(self.python_fully_qualified_name)
                    try:
                        ttnn.graph.record_python_operation(
                            self.python_fully_qualified_name,
                            function_args,
                            function_kwargs,
                            record=python_io_record,
                        )
                    except Exception as exception:
                        ttnn.graph.record_python_operation_error(
                            python_io_record, type(exception).__name__, str(exception)
                        )
                        raise

                if ttnn.CONFIG.enable_logging or ttnn.CONFIG.enable_comparison_mode:
                    input_tensors = get_all_tensors((function_args, function_kwargs))
                    set_tensor_id(input_tensors)
                    decorated_function = set_output_tensor_id_decorator(decorated_function)

                if ttnn.CONFIG.enable_logging:
                    devices = get_devices((function_args, function_kwargs))
                    for device in devices:
                        ttnn.synchronize_device(device)

                    logger.debug(f"Started {self.python_fully_qualified_name:50}")

                if ttnn.CONFIG.enable_comparison_mode:
                    decorated_function = comparison_decorator(decorated_function)

                # Initialize variables for comparison mode
                local_tensor_comparison_records = []
                local_golden_function_output = []
                global_tensor_comparison_records = []
                global_golden_function_output = []

                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL, _internal=True)

                operation_error = None
                try:
                    if cq_id is None:
                        output = decorated_function(*function_args, **function_kwargs)
                    else:
                        with command_queue(cq_id):
                            output = decorated_function(*function_args, **function_kwargs)

                    # Success path - only runs if no exception
                    if ttnn.CONFIG.enable_comparison_mode:
                        (
                            output,
                            (
                                local_tensor_comparison_records,
                                local_golden_function_output,
                                global_tensor_comparison_records,
                                global_golden_function_output,
                            ),
                        ) = output

                    if ttnn.CONFIG.enable_logging:
                        for device in devices:
                            ttnn.synchronize_device(device)
                        logger.debug(f"Finished {self.python_fully_qualified_name:50}")

                    # Comparison mode: record Python-specific golden comparison data
                    # for offline graph_report import.
                    if ttnn.CONFIG.enable_comparison_mode:
                        golden_tensors = get_all_tensors((local_golden_function_output, global_golden_function_output))
                        ttnn.graph.record_tensor_comparison_data(
                            local_tensor_comparison_records=local_tensor_comparison_records,
                            global_tensor_comparison_records=global_tensor_comparison_records,
                            tensors=[get_tensor_report_record(tensor) for tensor in golden_tensors],
                        )

                except Exception as exception:
                    operation_error = (type(exception).__name__, str(exception))
                    raise

                finally:
                    captured_graph = ttnn.graph.end_graph_capture()
                    # Failure path: the success block below is skipped, so persist here instead.
                    if operation_error is not None and ttnn.graph.is_graph_capture_active():
                        ttnn.graph.record_python_operation_error(python_io_record, *operation_error)
                        ttnn.graph.store_captured_graph(captured_graph)

                if ttnn.graph.is_graph_capture_active() and ttnn.graph.is_python_io_recording_enabled():
                    ttnn.graph.store_output_tensor_ids(get_output_tensor_ids(output))
                    ttnn.graph.store_captured_graph(captured_graph)

                for hook in POST_OPERATION_HOOKS:
                    hook_return_value = hook(self, function_args, function_kwargs, output)
                    if hook_return_value is not None:
                        raise RuntimeError(
                            f"Post-operation hook {hook} returned {hook_return_value} but must return None"
                        )

                return output

            return call_wrapper

        function = runtime_decorator(function)
        self.decorated_function = function

        if self.function.__doc__ is None:
            return

        # Delete the signature line created by nanobind
        docstring_lines = self.function.__doc__.split("\n")
        op_name = self.python_fully_qualified_name.split(".")[-1]
        if f"{op_name}(" in docstring_lines[0]:
            docstring_lines.pop(0)
        self.__doc__ = "\n".join(docstring_lines)

    def __call__(self, *function_args, **function_kwargs):
        recording = ttnn.graph.is_python_io_recording_enabled()
        if recording:
            ttnn.graph.track_function_start(self.python_fully_qualified_name)
        try:
            if not OPERATION_CALL_STACK:
                ttnn._ttnn.fetch_and_increment_python_operation_id()
            OPERATION_CALL_STACK.append(self.python_fully_qualified_name)
            output = self.decorated_function(*function_args, **function_kwargs)
        finally:
            OPERATION_CALL_STACK.pop()
            if recording:
                ttnn.graph.track_function_end()
        return output


if TRACE_ALLOC_TRACKING:
    _untracked_operation_call = Operation.__call__

    if TRACE_ALLOC_DIAGNOSTICS:

        @wraps(_untracked_operation_call)
        def _tracked_operation_call(self, *function_args, **function_kwargs):
            _drain_traceback_ids(source="op_start", op_name=self.python_fully_qualified_name)
            push_allocation_context(self.python_fully_qualified_name)
            try:
                result = _untracked_operation_call(self, *function_args, **function_kwargs)
                _drain_traceback_ids(source="op_end", op_name=self.python_fully_qualified_name)
                return result
            finally:
                pop_allocation_context()

    else:

        @wraps(_untracked_operation_call)
        def _tracked_operation_call(self, *function_args, **function_kwargs):
            push_allocation_context(self.python_fully_qualified_name)
            try:
                return _untracked_operation_call(self, *function_args, **function_kwargs)
            finally:
                pop_allocation_context()

    Operation.__call__ = _tracked_operation_call


class RegisteredOperations:
    def __init__(self):
        self.operations = set()

    def __iter__(self):
        return iter(self.operations)

    def __contains__(self, operation):
        return operation in self.operations

    def add(self, operation, name):
        if operation in self.operations:
            raise RuntimeError(f'Operation with name "{name}" is already registered')
        self.operations.add(operation)


REGISTERED_OPERATIONS = RegisteredOperations()


def query_registered_operations(include_experimental=False):
    sorted_operations = sorted(REGISTERED_OPERATIONS)

    ttnn_operations = [
        operation
        for operation in sorted_operations
        if operation.python_fully_qualified_name.startswith("ttnn.")
        and not operation.python_fully_qualified_name.startswith("ttnn.experimental.")
    ]
    ttl_operations = [
        operation
        for operation in sorted_operations
        if operation.python_fully_qualified_name.startswith("ttnn.experimental.")
    ]
    if include_experimental:
        return ttnn_operations + ttl_operations
    else:
        return ttnn_operations


def dump_operations(csv_file, include_experimental=False):
    import pandas as pd

    apis = query_registered_operations(include_experimental)

    def to_dict(obj):
        return {
            "python_fully_qualified_name": obj.python_fully_qualified_name,
            "function": str(obj.function),
            "preprocess_golden_function_inputs": str(obj.preprocess_golden_function_inputs),
            "golden_function": str(obj.golden_function),
            "postprocess_golden_function_outputs": str(obj.postprocess_golden_function_outputs),
            "has_golden_function": obj.golden_function is not None,
            "is_cpp_operation": obj.is_cpp_operation,
            "is_experimental": obj.is_experimental,
        }

    df = pd.DataFrame([to_dict(obj) for obj in apis])
    df.sort_values(by=["is_experimental", "is_cpp_operation", "python_fully_qualified_name"], inplace=True)
    df = df[
        [
            "python_fully_qualified_name",
            "is_cpp_operation",
            "has_golden_function",
            "is_experimental",
        ]
    ]
    df.to_csv(csv_file, index=False)


def get_golden_function(operation):
    if operation.golden_function is None:
        raise RuntimeError(f"{operation} does not have a golden function")
    return operation.golden_function


def get_fallback_function(operation):
    golden_function = get_golden_function(operation)

    def fallback_function(*function_args, **function_kwargs):
        preprocess_inputs = operation.preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
        postprocess_outputs = (
            operation.postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
        )
        updated_function_args, updated_function_kwargs = preprocess_inputs(function_args, function_kwargs)
        output = golden_function(*updated_function_args, **updated_function_kwargs)
        output = postprocess_outputs(output, function_args, function_kwargs)

        return output

    return fallback_function


def attach_golden_function(
    operation,
    golden_function,
    *,
    preprocess_golden_function_inputs=None,
    postprocess_golden_function_outputs=None,
):
    operation.golden_function = golden_function
    operation.preprocess_golden_function_inputs = (
        preprocess_golden_function_inputs or default_preprocess_golden_function_inputs
    )
    operation.postprocess_golden_function_outputs = (
        postprocess_golden_function_outputs or default_postprocess_golden_function_outputs
    )


def create_module_if_not_exists(module_name):
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Recursively create parent modules if they don't exist
    parent_module_name, _, child_module_name = module_name.rpartition(".")
    if parent_module_name:
        parent_module = create_module_if_not_exists(parent_module_name)
    else:
        parent_module = None

    # Check if parent already has this as a package attribute
    if parent_module and hasattr(parent_module, child_module_name):
        existing_attr = getattr(parent_module, child_module_name)
        if hasattr(existing_attr, "__path__"):
            # It's a package, register it and return
            sys.modules[module_name] = existing_attr
            return existing_attr

    # Create the module
    new_module = module_from_spec(ModuleSpec(module_name, None))

    if parent_module:
        setattr(parent_module, child_module_name, new_module)
    sys.modules[module_name] = new_module
    return new_module


def register_cpp_operation(target_module: types.ModuleType, func_name: str, function: Callable):
    operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode else Operation

    operation = operation_class(
        python_fully_qualified_name=function.python_fully_qualified_name,
        function=function,
        golden_function=None,
        preprocess_golden_function_inputs=None,
        postprocess_golden_function_outputs=None,
        is_cpp_operation=True,
        is_experimental=False,
    )

    REGISTERED_OPERATIONS.add(operation, func_name)
    setattr(target_module, func_name, operation)

    return operation


def register_python_operation(
    *,
    name,
    is_experimental=False,
    is_method=False,
    golden_function=None,
    preprocess_golden_function_inputs=None,
    postprocess_golden_function_outputs=None,
    doc=None,
):
    python_fully_qualified_name = name

    def operation_decorator(function: Callable):
        is_cpp_operation = hasattr(function, "__ttnn_operation__")

        if is_cpp_operation:
            raise RuntimeError(f"{function} is a C++ operation, but it is being registered as a Python operation")
        # Disabling for now (See GH issue #18386)
        # elif not is_experimental and not is_method:
        #     logger.debug(f"Should {python_fully_qualified_name} be migrated to C++?")

        operation_class = FastOperation if ttnn.CONFIG.enable_fast_runtime_mode else Operation

        if not ttnn.CONFIG.enable_fast_runtime_mode:
            # Wrap function before attaching documentation to avoid errors
            if doc is not None:

                def doc_decorator(function):
                    @wraps(function)
                    def wrapper(*args, **kwargs):
                        return function(*args, **kwargs)

                    return wrapper

                function = doc_decorator(function)
                function.__doc__ = doc

        operation = operation_class(
            python_fully_qualified_name=python_fully_qualified_name,
            function=function,
            golden_function=golden_function,
            preprocess_golden_function_inputs=preprocess_golden_function_inputs,
            postprocess_golden_function_outputs=postprocess_golden_function_outputs,
            is_cpp_operation=False,
            is_experimental=is_experimental,
        )

        attach_golden_function(
            operation,
            golden_function,
            preprocess_golden_function_inputs=preprocess_golden_function_inputs,
            postprocess_golden_function_outputs=postprocess_golden_function_outputs,
        )

        if not is_method:  # Do not export methods
            module_path, _, func_name = python_fully_qualified_name.rpartition(".")
            if not module_path:
                raise RuntimeError("Module path have to have at least 2 tokens!")
            if not module_path.startswith("ttnn"):
                raise RuntimeError('Module path must start with "ttnn."')

            target_module = create_module_if_not_exists(module_path)

            REGISTERED_OPERATIONS.add(operation, python_fully_qualified_name)
            setattr(target_module, func_name, operation)

        # Wrap method appropriately in order to avoid errors
        if is_method:

            @wraps(operation)
            def method_call(self, *function_args, **function_kwargs):
                return operation(self, *function_args, **function_kwargs)

            return method_call

        return operation

    return operation_decorator


def register_ttl_operation_as_ttnn_operation(python_fully_qualified_name, function):
    function = register_python_operation(
        name=python_fully_qualified_name,
        is_experimental=True,
    )(function)
    return function
