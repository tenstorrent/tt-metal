# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.update_padded_kv_cache: the ROW_MAJOR copy path with per-chip stripes that are not whole 32-row tiles
(the fork's one change; the stock op refuses them). A TP-deduped cache (tp_axis = 1, cluster_axis = 0) on the box's
whole mesh, as DeepSeek-V4.1's ratio-2 compressed caches at chunk 5120: 640 tokens a mesh row, 4 columns -> 80
compressed rows a chip, a cache of 11 chunks = 880 rows a chip (also not a tile multiple), ND-sharded one 16-row
record per DRAM bank round robin (the served cache's layout). Torch reference: the sp x tp linearized block-cyclic
order (global row g on chip (g / stripe) mod (SP TP), local row (g / (stripe SP TP)) stripe + g mod stripe), every chip
compared exactly, a sentinel fill catches rows written that should not be."""

import pytest
import torch

import ttnn

WIDTH = 512  # one compressed row (DeepSeek-V4.1 ckv)
REC = 16  # rows per 32-token record at ratio 2: the ND shard
STRIPE = 80  # rows a chip writes per chunk (not a multiple of 32)
CHUNKS = 11  # cache depth in chunks: 880 rows a chip
LAYERS, LAYER, SLOT = 2, 1, 0
SENTINEL = -7.0


def _mesh_cases():
    # The case runs on the box's own mesh (a smaller mesh on a bigger box fails the 2D fabric handshake)
    return [pytest.param(m, id=f"{m[0]}x{m[1]}") for m in ((8, 4), (2, 4), (2, 2))]


def _cache(mesh, sp, tp):
    rows = CHUNKS * STRIPE
    banks = mesh.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, REC, WIDTH],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)
    host = torch.full([LAYERS, 1, rows * sp * tp, WIDTH], SENTINEL)
    t = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),  # chip-linear r tp + c
    )
    phys = ttnn.MeshShape(sp, tp)
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(phys)]
    t.update_tensor_topology(ttnn.TensorTopology(phys, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords))
    return t


def _input(mesh, x):
    """x [SP, tp STRIPE, WIDTH]: mesh row s gets x[s], replicated over the columns."""
    return ttnn.from_torch(
        x.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
    )


def _u32(mesh, v):
    return ttnn.from_torch(
        torch.tensor([[v]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _expect(ref, x, start, valid, sp, tp):
    """ref [SP TP, rows, WIDTH] (chip-linear r tp + c): global rows [start, valid) of the chunk at ``start`` (a
    multiple of the chunk's sp tp STRIPE rows), input row j of mesh row s = global row start + s tp STRIPE + j."""
    n = sp * tp
    for s in range(sp):
        for j in range(tp * STRIPE):
            g = start + s * tp * STRIPE + j
            if g >= valid:
                continue
            chip = (g // STRIPE) % n
            ref[chip, (g // (STRIPE * n)) * STRIPE + g % STRIPE] = x[s, j]


@pytest.mark.parametrize("mesh_device", _mesh_cases(), indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "require_exact_physical_num_devices": True}],
    indirect=True,
)
def test_row_stripes_tp_dedup(mesh_device):
    mesh = mesh_device
    sp, tp = tuple(mesh.shape)
    if tp < 2:
        pytest.skip("TP dedup needs at least 2 mesh columns")
    n, chunk = sp * tp, sp * tp * STRIPE
    torch.manual_seed(0)
    cache = _cache(mesh, sp, tp)
    ref = torch.full([n, CHUNKS * STRIPE, WIDTH], SENTINEL)
    args = dict(layer_idx=LAYER, num_layers=LAYERS, cluster_axis=0, tp_axis=1)

    # scalar path, whole chunks 0 and 3
    for k in (0, 3):
        x = torch.randn(sp, tp * STRIPE, WIDTH).bfloat16().float()
        ttnn.bringup.update_padded_kv_cache(
            cache, _input(mesh, x), slot_idx=SLOT, kv_actual_global=k * chunk, valid_global=(k + 1) * chunk, **args
        )
        _expect(ref, x, k * chunk, (k + 1) * chunk, sp, tp)
    # metadata path (the model's ratio-2 path), chunk 10 (the cache's last) clamped to its first rows: valid a
    # multiple of 32 that ends inside a chip's stripe
    k, valid_rows = 10, 32 * (chunk // 32 // 2) + 32
    x = torch.randn(sp, tp * STRIPE, WIDTH).bfloat16().float()
    slot_t, kv_t, valid_t = _u32(mesh, SLOT), _u32(mesh, k * chunk), _u32(mesh, k * chunk + valid_rows)
    ttnn.bringup.update_padded_kv_cache(cache, _input(mesh, x), slot_t, kv_t, valid_global=valid_t, **args)
    _expect(ref, x, k * chunk, k * chunk + valid_rows, sp, tp)

    got = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(cache)]
    for d in range(n):
        assert torch.equal(got[d][1 - LAYER, 0].float(), torch.full_like(ref[d], SENTINEL)), f"chip {d}: other layer"
        mism = (got[d][LAYER, 0].float() != ref[d]).any(dim=-1).nonzero().flatten().tolist()
        assert not mism, f"chip {d}: {len(mism)} rows differ, first {mism[:8]}"


# Only the 4-column meshes: the 10-tile input splits evenly over 2 columns.
@pytest.mark.parametrize("mesh_device", [p for p in _mesh_cases() if p.values[0][1] == 4], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D, "require_exact_physical_num_devices": True}],
    indirect=True,
)
def test_row_stripes_tile_layout_still_refused(mesh_device, expect_error):
    """The TILE path keeps its 32-row unit: a 10-tile input over 4 columns is refused as before."""
    mesh = mesh_device
    cache = ttnn.from_torch(
        torch.zeros(LAYERS, 1, 32 * 10 * 4, WIDTH),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    x = ttnn.from_torch(
        torch.zeros(1, 1, 320, WIDTH),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    with expect_error(RuntimeError, "must be divisible by tp_factor"):
        ttnn.bringup.update_padded_kv_cache(
            cache, x, slot_idx=0, kv_actual_global=0, layer_idx=0, num_layers=LAYERS, cluster_axis=0, tp_axis=1
        )
