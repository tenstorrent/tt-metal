# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

from typing import List, Dict, Optional, Callable, Tuple, Optional, Callable, Union

import ttnn


def get_device_mesh_core_grid(device_mesh):
    compute_with_storage_grid_size = device_mesh.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)


DeviceMesh = ttnn._ttnn.multi_device.DeviceMesh
DeviceMesh.core_grid = property(get_device_mesh_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType


def _get_rich_table(
    device_mesh: "ttnn.DeviceMesh", style_cell: Optional[Callable] = None, annotate_cell: Optional[Callable] = None
):
    from rich import box, padding
    from rich.align import Align
    from rich.table import Table
    from rich.text import Text
    from loguru import logger

    CELL_SIZE = 30

    # Setup rich table
    try:
        rows, cols = device_mesh.shape
    except AttributeError as e:
        logger.error("Error getting device mesh shape: {}.", e)
        rows, cols = 0, 0

    mesh_table = Table(
        title=f"DeviceMesh(rows={rows}, cols={cols}):",
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
                device = device_mesh.get_device(row_idx, col_idx)
            except Exception as e:
                logger.error("Error fetching device from DeviceMesh at row {}, col {}: {}.", row_idx, col_idx, e)
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


def visualize_device_mesh(device_mesh: "ttnn.DeviceMesh", tensor: "ttnn.Tensor" = None):
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

    mesh_table = _get_rich_table(device_mesh, style_cell=style_cell, annotate_cell=annotate_cell)
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


def open_device_mesh(
    device_grid: ttnn.DeviceGrid,
    device_ids: List[int],
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    num_command_queues: int = 1,
    dispatch_core_type: int = DispatchCoreType.WORKER,
):
    """
    open_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: int) -> ttnn.DeviceMesh:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    assert len(device_ids) > 0

    return ttnn._ttnn.multi_device.DeviceMesh(
        device_grid=device_grid.as_tuple(),
        device_ids=device_ids,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
        dispatch_core_type=dispatch_core_type,
    )


def close_device_mesh(device_mesh):
    """
    close_device_mesh(multi_device: ttnn.Multi) -> None:

    Close the device and remove it from the device cache.
    """
    return ttnn._ttnn.multi_device.close_device_mesh(device_mesh)


@contextlib.contextmanager
def create_device_mesh(
    device_grid: ttnn.DeviceGrid,
    device_ids: List[int],
    l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
    trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
    num_command_queues: int = 1,
    dispatch_core_type: int = DispatchCoreType.WORKER,
):
    """
    create_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]) -> ttnn.DeviceMesh

    Context manager for opening and closing a device.
    """
    device_mesh = open_device_mesh(
        device_grid=device_grid,
        device_ids=device_ids,
        l1_small_size=l1_small_size,
        trace_region_size=trace_region_size,
        num_command_queues=num_command_queues,
        dispatch_core_type=dispatch_core_type,
    )
    try:
        yield device_mesh
    finally:
        close_device_mesh(device_mesh)


def synchronize_devices(devices: Union["ttnn.Device", "ttnn.DeviceMesh"], queue_id: Optional[int] = None) -> None:
    """
    synchronize_devices(devices: Union[ttnn.Device, ttnn.DeviceMesh], queue_id: Optional[int] = None) -> None:

    Synchronize the devices with host by waiting for all operations to complete.
    If queue_id is provided then only the operations associated with that queue_id are waited for,
    otherwise operations for all command queues are waited on.
    """
    if isinstance(devices, ttnn.Device):
        ttnn._ttnn.device.Synchronize(devices, queue_id)
    else:
        for device in devices.get_device_ids():
            ttnn._ttnn.device.Synchronize(devices.get_device(device), queue_id)


class TensorToMesh:
    """
    Defines the mapping of a torch.Tensor to a device mesh: e.g. Shard/Replicate.
    You can also "Bring your own TensorToMesh" based on your custom mapping.
    """

    def __init__(self, device_mesh):
        self.device_mesh = device_mesh

    def map(self, tensor: "torch.Tensor"):
        raise NotImplementedError("Subclasses must implement this method")

    def config(self):
        raise NotImplementedError("Subclasses must implement this method")


class MeshToTensor:
    """
    Defines the inverse operation of TensorToMesh. Given a set of per-device
    ttnn.Tensor objects (aggregated into a single ttnn.Tensor), this class defines
    the mapping back to one or many torch.Tensor objects.

    You can also "Bring your own MeshToTensor" based on your custom mapping.
    """

    def compose(self, tensor: ttnn.Tensor):
        raise NotImplementedError("Subclasses must implement this method")


class ShardTensorToMesh(TensorToMesh):
    def __init__(self, device_mesh, dim):
        super().__init__(device_mesh)
        self.shard_dim = dim

    def map(self, tensor: "torch.Tensor") -> Dict[int, ttnn.Tensor]:
        import torch

        sliced_tensors = torch.chunk(tensor, self.device_mesh.get_num_devices(), dim=self.shard_dim)
        return list(sliced_tensors)

    def config(self):
        return {
            "strategy": "shard",
            "shard_dim": f"{self.shard_dim}",
        }


class ShardTensor2dMesh(TensorToMesh):
    """
    Shard a tensor across a 2D mesh of devices.

    This class implements a strategy for distributing a tensor across a 2D grid of devices,
    allowing for efficient parallel processing in distributed computing environments.
    """

    def __init__(self, device_mesh: DeviceMesh, mesh_shape: Tuple[int, int], dims: Tuple[Optional[int], Optional[int]]):
        """
        Initialize the ShardTensor2dMesh.

        Args:
            device_mesh: The target device mesh for distributing the tensor.
            mesh_shape: The shape of the 2D mesh as (rows, cols).
            dims: The dimensions to shard along, specified as (row_dim, col_dim).

        The `dims` tuple determines how the tensor is sharded across the 2D mesh:
        - row_dim: The dimension to shard across mesh rows (or None for replication).
        - col_dim: The dimension to shard across mesh columns (or None for replication).

        Examples:
        1. dims=(2, 3) for a tensor of shape (A, B, C, D):
           - Shard along dimension 2 (C) across mesh rows
           - Shard along dimension 3 (D) across mesh columns

        2. dims=(None, 3):
           - Replicate across mesh rows
           - Shard along dimension 3 (D) across mesh columns

        3. dims=(None, None):
           - Fully replicate the tensor across all devices
        """
        super().__init__(device_mesh)
        self.mesh_shape: Tuple[int, int] = mesh_shape
        self.dims: Tuple[Optional[int], Optional[int]] = dims

        device_mesh_rows, device_mesh_cols = self.device_mesh.shape
        if mesh_shape[0] > device_mesh_rows or mesh_shape[1] > device_mesh_cols:
            raise ValueError("ShardTensor2dMesh: Device mesh shape does not match the provided mesh shape.")

    def map(self, tensor: "torch.Tensor") -> List["torch.Tensor"]:
        """
        Map the input tensor to a list of sharded tensors.

        Args:
            tensor: The input tensor to be sharded.

        Returns:
            A list of sharded tensors, one for each device in the mesh.

        Raises:
            ValueError: If the number of sharding dimensions is not 2.
        """
        import torch

        if len(self.dims) != 2:
            raise ValueError("ShardTensor2dMesh only supports 2D shard dimensions")

        rows, cols = self.mesh_shape
        row_dim, col_dim = self.dims

        # Shard along rows
        row_tensors = (
            [tensor.clone() for _ in range(rows)] if row_dim is None else torch.chunk(tensor, rows, dim=row_dim)
        )

        # Shard along columns
        if col_dim is None:
            return [t.clone() for t in row_tensors for _ in range(cols)]
        tensor_shards = [tt for t in row_tensors for tt in torch.chunk(t, cols, dim=col_dim)]

        if len(tensor_shards) != rows * cols:
            raise ValueError(
                "ShardTensor2dMesh: Sharding failed. Number of shards should match the product of the mesh dimensions."
            )

        return tensor_shards

    def config(self) -> Dict[str, str]:
        """
        Provide the configuration of the sharding strategy.

        Returns:
            A dictionary containing the sharding strategy and dimensions.
        """
        return {
            "strategy": "shard_2d",
            "mesh_shape_y": str(self.mesh_shape[0]),
            "mesh_shape_x": str(self.mesh_shape[1]),
        }


class ConcatMesh2dToTensor(MeshToTensor):
    """
    Concatenate tensors from a 2D mesh back into a single tensor.

    This class implements the inverse operation of ShardTensor2dMesh, combining
    sharded tensors from a 2D device mesh back into a single tensor.
    """

    def __init__(self, device_mesh: DeviceMesh, mesh_shape: Tuple[int, int], dims: Tuple[int, int]):
        """
        Initialize the ConcatMesh2dToTensor.

        Args:
            device_mesh: The source device mesh containing the sharded tensors.
            mesh_shape: The shape of the 2D mesh as (rows, cols).
            dims: A tuple of two integers specifying the dimensions along which to concatenate the tensors.
                  The first element (row_dim) indicates the dimension for concatenating tensors from different rows.
                  The second element (col_dim) indicates the dimension for concatenating tensors from different columns.
                  Both dimensions must be specified and different from each other.
                  These dimensions correspond to the tensor dimensions, not the mesh dimensions.
                  For example, if the original tensor was 4D with shape (batch, channel, height, width),
                  and it was sharded across height and width, dims might be (-2, -1) or (2, 3).

        Raises:
            ValueError: If either dimension in 'dims' is None or if both dimensions are the same.
        """
        self.device_mesh = device_mesh
        self.mesh_shape = mesh_shape
        self.dims = dims
        if self.dims[0] == self.dims[1]:
            raise ValueError("Both dimensions in 'dims' must be different")

    def compose(self, tensor: ttnn.Tensor) -> "torch.Tensor":
        """
        Compose the sharded tensors back into a single tensor.

        Args:
            tensor: A ttnn.Tensor object containing the sharded tensors distributed across multiple devices.

        Returns:
            A single torch.Tensor that combines all the sharded tensors from all devices.

        This method first concatenates the shards along the column dimension within each row,
        then concatenates the resulting tensors along the row dimension to form the final tensor.
        """
        import torch

        device_shards = [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]

        rows, cols = self.mesh_shape
        row_dim, col_dim = self.dims

        # Reshape the list of shards into a 2D list representing the device mesh
        device_grid = [device_shards[i : i + cols] for i in range(0, len(device_shards), cols)]

        # Concatenate along columns first (within each row)
        row_concatenated = [torch.cat(row, dim=col_dim) for row in device_grid]

        # Then concatenate the resulting tensors along rows
        return torch.cat(row_concatenated, dim=row_dim)


class ReplicateTensorToMesh(TensorToMesh):
    def __init__(self, device_mesh: DeviceMesh):
        super().__init__(device_mesh)

    def map(self, tensor: "torch.Tensor"):
        return [tensor for i in range(self.device_mesh.get_num_devices())]

    def config(self):
        return {
            "strategy": "replicate",
            "replication_factor": str(self.device_mesh.get_num_devices()),
        }


class ConcatMeshToTensor(MeshToTensor):
    def __init__(self, device_mesh: DeviceMesh, dim: int):
        self.concat_dim = dim
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> "torch.Tensor":
        import torch

        device_shards_converted_to_torch = [
            ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)
        ]
        return torch.cat(device_shards_converted_to_torch, dim=self.concat_dim)


class ListMeshToTensor(MeshToTensor):
    def __init__(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> List["torch.Tensor"]:
        return [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]


__all__ = []
