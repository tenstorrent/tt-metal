# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools

from typing import List, Dict, Optional, Callable, Tuple, Optional, Callable, Union, List

import ttnn


def get_mesh_device_core_grid(mesh_device):
    compute_with_storage_grid_size = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


MeshDevice = ttnn._ttnn.multi_device.MeshDevice
MeshDevice.core_grid = property(get_mesh_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType


def _get_rich_table(
    mesh_device: "ttnn.MeshDevice", style_cell: Optional[Callable] = None, annotate_cell: Optional[Callable] = None
):
    from rich import box, padding
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    from loguru import logger

    CELL_SIZE = 30

    # Setup rich table
    try:
        rows, cols = mesh_device.shape
    except AttributeError as e:
        logger.error("Error getting device mesh shape: {}.", e)
        rows, cols = 0, 0

    mesh_table = Table(
        title=f"MeshDevice(rows={rows}, cols={cols}):",
        show_header=False,
        show_footer=False,
        box=box.SQUARE,
        expand=False,
        show_lines=True,
        padding=(0, 0),
    )

    for _ in range(cols):
        mesh_table.add_column(justify="center", vertical="middle", width=CELL_SIZE)

    # Populate table
    for row_idx in range(rows):
        row_cells = []
        for col_idx in range(cols):
            try:
                device = mesh_device.get_device(row_idx, col_idx)
            except Exception as e:
                logger.error("Error fetching device from MeshDevice at row {}, col {}: {}.", row_idx, col_idx, e)
                device = None

            try:
                device_id = f"Dev. ID: {device.id()}" if device else "Empty"
                coords = f"({row_idx}, {col_idx})"
                annotation = annotate_cell(device) if annotate_cell and device else ""

                cell_content = Text(f"{device_id}\n{coords}\n{annotation}", justify="center")
                cell_content.truncate(CELL_SIZE * 3, overflow="ellipsis")  # 3 lines max
            except AttributeError as e:
                logger.error("Error formatting cell content at row {}, col {}: {}.", row_idx, col_idx, e)
                cell_content = Text("Error", justify="center")

            cell_style = style_cell(device) if style_cell and device else None
            cell = Align(cell_content, "center", vertical="middle")
            if cell_style:
                cell.style = cell_style
            row_cells.append(cell)
        mesh_table.add_row(*row_cells)
    return mesh_table


def visualize_mesh_device(mesh_device: "ttnn.MeshDevice", tensor: "ttnn.Tensor" = None):
    """
    Visualize the device mesh and the given tensor (if specified).
    """
    from rich.console import Console
    from rich.style import Style
    from loguru import logger

    style_cell, annotate_cell = None, None
    if tensor is not None:
        try:
            mapped_devices = set(device.id() for device in tensor.devices())
        except Exception as e:
            logger.error(f"Error getting devices for tensor: {e}")
            mapped_devices = set()

        def color_mapped_devices(device):
            try:
                return Style(bgcolor="dark_green") if device.id() in mapped_devices else None
            except Exception as e:
                logger.error(f"Error getting device ID: {e}")
                return None

        def annotate_with_tensor_shape(device):
            return f"{tensor.shape}" if device.id() in mapped_devices else ""

        style_cell = color_mapped_devices
        annotate_cell = annotate_with_tensor_shape

    mesh_table = _get_rich_table(mesh_device, style_cell=style_cell, annotate_cell=annotate_cell)
    Console().print(mesh_table)


def get_num_devices() -> List[int]:
    return ttnn._ttnn.device.GetNumAvailableDevices()


def get_num_pcie_devices() -> int:
    return ttnn._ttnn.device.GetNumPCIeDevices()


def get_pcie_device_ids() -> List[int]:
    num_pcie_devices = get_num_pcie_devices()
    return list(range(num_pcie_devices))


def get_device_ids() -> List[int]:
    num_devices = get_num_devices()
    return list(range(num_devices))


def open_mesh_device(
    mesh_shape: ttnn.MeshShape,
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    num_command_queues: int = 1,
    dispatch_core_config: ttnn.DispatchCoreConfig = ttnn.DispatchCoreConfig(),
    offset: Optional[ttnn.MeshCoordinate] = None,
    physical_device_ids: List[int] = [],
):
    """
    Open a mesh device with the specified configuration.

    Args:
        mesh_shape (ttnn.MeshShape): The shape of the mesh device.
        l1_small_size (int, optional): Size of the L1 small memory. Defaults to ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE.
        trace_region_size (int, optional): Size of the trace region. Defaults to ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE.
        num_command_queues (int, optional): Number of command queues. Defaults to 1.
        dispatch_core_type (int, optional): Type of dispatch core. Defaults to DispatchCoreType.WORKER.
        offset (ttnn.MeshCoordinate, optional): Offset in logical mesh coordinates for the mesh device. Defaults to None.
        physical_device_ids (List[int], optional): List of physical device IDs to use. Defaults to [].

    Returns:
        ttnn._ttnn.multi_device.MeshDevice: The opened mesh device.

    """
    return ttnn._ttnn.multi_device.MeshDevice(
        mesh_shape=mesh_shape,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
        dispatch_core_config=dispatch_core_config,
        offset=offset,
        physical_device_ids=physical_device_ids,
    )


def close_mesh_device(mesh_device):
    """
    close_mesh_device(multi_device: ttnn.Multi) -> None:

    Close the device and remove it from the device cache.
    """
    return ttnn._ttnn.multi_device.close_mesh_device(mesh_device)


@contextlib.contextmanager
def create_mesh_device(*args, **kwargs):
    """
    create_mesh_device(*args, **kwargs) -> ttnn.MeshDevice

    Context manager for opening and closing a device.
    """
    mesh_device = open_mesh_device(*args, **kwargs)
    try:
        yield mesh_device
    finally:
        close_mesh_device(mesh_device)


def synchronize_devices(
    devices: Union["ttnn.Device", "ttnn.MeshDevice"],
    queue_id: Optional[int] = ttnn.DefaultQueueId,
    sub_device_ids: List[ttnn.SubDeviceId] = [],
) -> None:
    """
    synchronize_devices(devices: Union[ttnn.Device, ttnn.MeshDevice], queue_id: Optional[int] = None, sub_device_ids: List[ttnn.SubDeviceId] = []) -> None:

    Synchronize the devices with host by waiting for all operations to complete.
    If queue_id is provided then only the operations associated with that queue_id are waited for,
    otherwise operations for all command queues are waited on.
    """
    if isinstance(devices, ttnn.Device):
        ttnn._ttnn.device.synchronize_device(devices, queue_id, sub_device_ids)
    else:
        for device in devices.get_device_ids():
            ttnn._ttnn.device.synchronize_device(devices.get_device(device), queue_id, sub_device_ids)


@contextlib.contextmanager
def distribute(default: Union[ttnn.TensorToMesh, ttnn.MeshToTensor]):
    """
    Context manager to temporarily modify the behavior of ttnn.from_torch and ttnn.to_torch to use the specified
    mesh_mapper or mesh_composer for tensor distribution and composition to/from MeshDevice.
    Invocations of ttnn.from_torch(..) will use the mesh_mapper as defined by the default in ttnn.distribute.
    Invocations of ttnn.to_torch(..) will use the mesh_composer as defined by the default in ttnn.distribute.

    Args:
        mesh_mapper_or_composer (Union[TensorToMesh, MeshToTensor]): An instance of either TensorToMesh or MeshToTensor
            used to map tensors to a mesh or compose tensors from a mesh.

    Example:
        with distribute(ShardTensorToMesh(mesh_device, dim=3)):
            # Code here will use the default mapper
            result = ttnn.from_torch(torch_tensor)

        is equivalent to:
        result = ttnn.from_torch(torch_tensor, mesh_mapper=ShardTensorToMesh(mesh_device, dim=3))
    """
    _original_to_torch = ttnn.to_torch
    _original_from_torch = ttnn.from_torch

    try:
        if isinstance(default, ttnn.TensorToMesh):
            ttnn.from_torch = functools.partial(_original_from_torch, mesh_mapper=default)
        elif isinstance(default, ttnn.MeshToTensor):
            ttnn.to_torch = functools.partial(_original_to_torch, mesh_composer=default)
        else:
            raise ValueError("Argument must be an instance of either TensorToMesh or MeshToTensor.")
        yield

    finally:
        # Restore the original functions
        ttnn.from_torch = _original_from_torch
        ttnn.to_torch = _original_to_torch


__all__ = []
