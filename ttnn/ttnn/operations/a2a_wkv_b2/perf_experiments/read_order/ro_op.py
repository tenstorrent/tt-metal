"""read_order experiment: a2a_wkv_b2 with this dir's descriptor + kernels (variant via RI_* env knobs)."""
import os
import sys

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttnn.operations.a2a_wkv_b2.a2a_wkv_b2 import validate, default_compute_kernel_config  # noqa: E402
import ro_descriptor  # noqa: E402


def a2a_wkv_b2(input_tensor, weight, *, cluster_axis=1, num_links=2, compute_kernel_config=None):
    validate(
        input_tensor,
        weight,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=compute_kernel_config,
    )
    cfg = compute_kernel_config or default_compute_kernel_config()
    mesh_device = input_tensor.device()
    n = tuple(mesh_device.shape)[cluster_axis]
    _, H, T, _ = tuple(input_tensor.shape)
    V = tuple(weight.shape)[-1]
    sem_ready, sem_done = ro_descriptor.get_cross_chip_semaphores(mesh_device)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H // n, T * n, V]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    mesh_pd = ro_descriptor.create_mesh_program_descriptor(
        input_tensor,
        weight,
        output_tensor,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=cfg,
        sem_ready=sem_ready,
        sem_done=sem_done,
    )
    return ttnn.generic_op([input_tensor, weight, output_tensor], mesh_pd)
