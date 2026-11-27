"""launch_gap experiment: the op's entry point, but building its program from lg_descriptor (knobs in KN)."""
import ttnn
from ttnn.operations.a2a_wkv_b2.a2a_wkv_b2 import validate, default_compute_kernel_config
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap import lg_descriptor as D


def lg_op(input_tensor, weight, *, cluster_axis=1, num_links=2, compute_kernel_config=None):
    validate(input_tensor, weight, cluster_axis=cluster_axis, num_links=num_links)
    cfg = compute_kernel_config or default_compute_kernel_config()
    mesh_device = input_tensor.device()
    n = tuple(mesh_device.shape)[cluster_axis]
    _, H, T, _ = tuple(input_tensor.shape)
    V = tuple(weight.shape)[-1]
    sem_ready, sem_done = D.get_cross_chip_semaphores(mesh_device)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, H // n, T * n, V]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    mpd = D.create_mesh_program_descriptor(
        input_tensor,
        weight,
        out,
        cluster_axis=cluster_axis,
        num_links=num_links,
        compute_kernel_config=cfg,
        sem_ready=sem_ready,
        sem_done=sem_done,
    )
    return ttnn.generic_op([input_tensor, weight, out], mpd)
