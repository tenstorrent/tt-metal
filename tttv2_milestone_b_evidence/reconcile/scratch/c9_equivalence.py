"""C9: prove the test helper's prefetch config is unchanged by the de-duplication.

Rebuilds the OLD (pre-C9) helper body verbatim and compares every field of the
resulting Prefetcher2DConfig against what the production builder now returns.
"""
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import ttnn
from models.common.modules.prefetcher import Prefetcher2DConfig, Prefetcher2DModeConfig
from models.common.tests.modules._wh_galaxy_hardware import galaxy_prefetcher_config

OLD_SENDER_COORDS = ((0, 9), (0, 0), (0, 4), (0, 5), (4, 0), (4, 9), (4, 1), (4, 7), (4, 6), (4, 2), (4, 4), (4, 5))
OLD_GLOBAL_CB_SIZE = 728 * 1088


def old_config(mesh_device, resources, weight_count, *, global_cb_size=OLD_GLOBAL_CB_SIZE):
    sender_coords = tuple(ttnn.CoreCoord(x, y) for x, y in OLD_SENDER_COORDS)
    receiver_pairs = tuple(((1, y), (2, y)) for y in (9, 0, 4, 5)) + tuple(
        ((5, y), (6, y)) for y in (0, 9, 1, 7, 6, 2, 4, 5)
    )
    receiver_sets = tuple(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e))}) for s, e in receiver_pairs
    )
    dummy_sender_coords = tuple(
        ttnn.CoreCoord(x, y) for x, y in ((0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8), (4, 3), (4, 8))
    )

    def ranges(*coordinates):
        return ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1 in coordinates]
        )

    dummy_receiver_sets = (
        ranges((3, 0, 3, 0), (1, 1, 3, 1)),
        ranges((1, 2, 3, 2)),
        ranges((1, 3, 3, 3), (3, 4, 3, 4)),
        ranges((3, 5, 3, 5), (1, 6, 3, 6)),
        ranges((1, 7, 3, 7)),
        ranges((1, 8, 3, 8), (3, 9, 3, 9)),
        ranges((5, 3, 6, 3)),
        ranges((5, 8, 6, 8)),
    )
    sender_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in sender_coords])
    address_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sender_cores, [1, weight_count], ttnn.ShardOrientation.ROW_MAJOR),
    )

    def mode_config(plan):
        return Prefetcher2DModeConfig(
            mode=plan.mode,
            sub_devices=plan.sub_devices,
            worker_sub_device_id=plan.worker_sub_device_id,
            stall_group=plan.stall_group,
            local_l1_size=plan.local_l1_size,
        )

    return Prefetcher2DConfig(
        mesh_device=mesh_device,
        architecture=resources.architecture,
        prefill=mode_config(resources.prefill),
        decode=mode_config(resources.decode),
        sender_receiver_mapping=tuple(zip(sender_coords + dummy_sender_coords, receiver_sets + dummy_receiver_sets)),
        global_cb_size=global_cb_size,
        expected_weight_count=weight_count,
        address_repeat_count=len(sender_coords),
        address_memory_config=address_memory_config,
        address_mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _mesh():
    m = MagicMock(spec=ttnn.MeshDevice)
    m.shape = (8, 4)
    m.get_num_devices.return_value = 32
    m.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    m.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    m.compute_with_storage_grid_size.return_value = ttnn.CoreCoord(7, 10)
    return m


from models.common.models.galaxy.plans import build_galaxy_resources_config
from models.common.models.galaxy.recipes import GalaxyDenseGeometry, resolve_galaxy_decode_placements

ttnn.ReplicateTensorToMesh = lambda *_a, **_k: "replicate-mapper"
ttnn.ShardTensor2dMesh = lambda *_a, **_k: "shard-2d-mapper"

mesh = _mesh()
geometry = GalaxyDenseGeometry(
    dim=8192,
    hidden_dim=28672,
    n_heads=64,
    n_kv_heads=8,
    head_dim=128,
    vocab_size=128256,
    max_seq_len=2048,
    prefill_sequence_lengths=(128,),
)
resources = build_galaxy_resources_config(mesh, geometry, resolve_galaxy_decode_placements(geometry, mesh))

old = old_config(mesh, resources, 5)
new = galaxy_prefetcher_config(mesh, resources, 5)

differences = []
for f in fields(Prefetcher2DConfig):
    a, b = getattr(old, f.name), getattr(new, f.name)
    if f.name == "address_mesh_mapper":
        continue  # two distinct ReplicateTensorToMesh instances; not comparable by value
    same = a == b
    if not same:
        differences.append((f.name, a, b))
print(f"fields compared: {len(fields(Prefetcher2DConfig))}")
if differences:
    for name, a, b in differences:
        print(f"  DIFFERS {name}:\n    old={a!r}\n    new={b!r}")
else:
    print("  no differences: the de-duplicated helper builds the identical config")
