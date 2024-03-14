# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections import abc

from typing import List, Dict
import tt_lib as ttl

import ttnn
import torch


DeviceMesh = ttnn._ttnn.multi_device.DeviceMesh


def get_num_devices() -> List[int]:
    return ttl.device.GetNumAvailableDevices()


def get_num_pcie_devices() -> int:
    return ttl.device.GetNumPCIeDevices()


def get_pcie_device_ids() -> List[int]:
    num_pcie_devices = get_num_pcie_devices()
    return list(range(num_pcie_devices))


def get_device_ids() -> List[int]:
    num_devices = get_num_devices()
    return list(range(num_devices))


def open_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]):
    """
    open_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: int) -> ttnn.DeviceMesh:

    Open a device with the given device_id. If the device is already open, return the existing device.
    """
    assert len(device_ids) > 0
    return ttnn._ttnn.multi_device.DeviceMesh(device_grid=device_grid.as_tuple(), device_ids=device_ids)


def close_device_mesh(device_mesh):
    """
    close_device(multi_device: ttnn.Multi) -> None:

    Close the device and remove it from the device cache.
    """
    return ttnn._ttnn.multi_device.close_device_mesh(device_mesh)


@contextlib.contextmanager
def create_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]):
    """
    create_device_mesh(device_grid: ttnn.DeviceGrid, device_ids: List[int]) -> ttnn.DeviceMesh

    Context manager for opening and closing a device.
    """
    device_mesh = open_device_mesh(device_grid=device_grid, device_ids=device_ids)
    try:
        yield device_mesh
    finally:
        close_device_mesh(device_mesh)


class TensorToMesh:
    """
    Defines the mapping of a torch.Tensor to a device mesh: e.g. Shard/Replicate.
    You can also "Bring your own TensorToMesh" based on your custom mapping.
    """

    def __init__(self, device_mesh):
        self.device_mesh = device_mesh
        self.device_id_to_tensor = {}

    def map(self, tensor: torch.tensor):
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

    def map(self, tensor: torch.tensor) -> Dict[int, ttnn.Tensor]:
        sliced_tensors = torch.chunk(tensor, self.device_mesh.get_num_devices(), dim=self.shard_dim)
        self.device_id_to_tensor = {i: input_tensor for i, input_tensor in enumerate(sliced_tensors)}
        return self.device_id_to_tensor


class ReplicateTensorToMesh(TensorToMesh):
    def __init__(self, device_mesh: DeviceMesh):
        super().__init__(device_mesh)

    def map(self, tensor: torch.tensor):
        self.device_id_to_tensor = {i: tensor for i in range(self.device_mesh.get_num_devices())}
        return self.device_id_to_tensor


class ConcatMeshToTensor(MeshToTensor):
    def __init__(self, device_mesh: DeviceMesh, dim: int):
        self.concat_dim = dim
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> torch.Tensor:
        device_shards_converted_to_torch = [
            ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)
        ]
        return torch.cat(device_shards_converted_to_torch, dim=self.concat_dim)


class ListMeshToTensor(MeshToTensor):
    def __init__(self, device_mesh: DeviceMesh):
        self.device_mesh = device_mesh

    def compose(self, tensor: ttnn.Tensor) -> List[torch.Tensor]:
        return [ttnn.to_torch(tt_input_tensor) for tt_input_tensor in ttnn.get_device_tensors(tensor)]


__all__ = []
