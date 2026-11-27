"""launch_gap2: the op's entry point, building its program from v_descriptor (knobs in KN)."""
import ttnn
from ttnn.operations.a2a_wkv_b2.a2a_wkv_b2 import validate, default_compute_kernel_config
from ttnn.operations.a2a_wkv_b2.perf_experiments.launch_gap2 import v_descriptor as D

VARIANTS = {
    "base": {},
    "merge": dict(merge=True),
    "tx": dict(tx_common=True),
    "cb": dict(cb_split=True),
    "merge_tx": dict(merge=True, tx_common=True),
    "merge_tx_cb": dict(merge=True, tx_common=True, cb_split=True),
    "merge_cb": dict(merge=True, cb_split=True),
    "merge_box": dict(merge=True, cb_box=True),
    "merge_tx_box": dict(merge=True, tx_common=True, cb_box=True),
}
BASE = dict(D.KN)


def apply(name, **extra):
    D.KN.clear()
    D.KN.update(BASE)
    D.KN.update(VARIANTS[name])
    D.KN.update(extra)


def v_op(input_tensor, weight, *, cluster_axis=1, num_links=2, compute_kernel_config=None):
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
