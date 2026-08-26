"""C1 neighbour check: do the decode core sets overlap the norm shard grid?

`_subgrid_cores` anchors the attention decode core sets at CoreCoord(1, 0) and
walks the worker subdevice; the distributed norm origin has moved to (2, 0).
Enumerate both and report the intersection explicitly.
"""
import ttnn
from models.common.models.galaxy.recipes import (
    GalaxyDenseGeometry,
    distributed_norm_decode_memory_config,
    distributed_norm_stats_memory_config,
    resolve_galaxy_decode_placements,
    worker_cores,
)
from unittest.mock import MagicMock
from types import SimpleNamespace


def _mesh():
    m = MagicMock(spec=ttnn.MeshDevice)
    m.shape = (8, 4)
    m.get_num_devices.return_value = 32
    m.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    m.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    m.compute_with_storage_grid_size.return_value = SimpleNamespace(x=7, y=10)
    return m


def cores(crs):
    out = set()
    for r in crs.ranges():
        for x in range(r.start.x, r.end.x + 1):
            for y in range(r.start.y, r.end.y + 1):
                out.add((x, y))
    return out


def grid_of(memcfg):
    return cores(memcfg.shard_spec.grid)


MODELS = {
    "llama-3.3-70b": dict(dim=8192, hidden_dim=28672, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=128256),
    "qwen3-32b": dict(dim=5120, hidden_dim=25600, n_heads=64, n_kv_heads=8, head_dim=128, vocab_size=151936),
}

print(f"worker subdevice: {sorted(cores(worker_cores()))[:3]} ... ({len(cores(worker_cores()))} cores)")
for name, spec in MODELS.items():
    g = GalaxyDenseGeometry(**spec, max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    p = resolve_galaxy_decode_placements(g, _mesh())
    norm = grid_of(p.residual_memcfg)
    stats = grid_of(distributed_norm_stats_memory_config(p.residual_memcfg))
    print(f"\n=== {name} ===")
    print(f"  norm/residual grid   : {sorted(norm)}")
    print(f"  fused-stats shard    : {sorted(stats)}")
    print(f"  stats inside worker  : {stats <= cores(worker_cores())}")
    print(f"  stats == norm origin : {sorted(stats) == [min(sorted(norm))]}")
    for label, memcfg in (
        ("attention_heads", p.attention_heads_memcfg),
        ("attention_kv", p.attention_kv_memcfg),
        ("attention_sdpa_output", p.attention_sdpa_output_memcfg),
        ("attention_gather_users", p.attention_gather_users_memcfg),
        ("attention_qkv_reduced", p.attention_qkv_reduced_memcfg),
        ("mlp_reduce_scatter", p.mlp_reduce_scatter_memcfg),
    ):
        s = grid_of(memcfg)
        print(f"  {label:24s} {len(s):3d} cores  overlap(norm)={len(s & norm):3d}  overlap(stats)={len(s & stats)}")
