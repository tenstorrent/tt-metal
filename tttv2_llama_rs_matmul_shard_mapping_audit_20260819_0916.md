# `llama_rs_matmul` Per-Device Shard Mapping Audit

## Checkpoint: 2026-08-19 09:16 UTC

### Scope

Audited the exact input and W1/W3 shard mapping used by
`models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py` for
`ttnn.experimental.llama_rs_matmul`. No TT hardware was used and no source files
were changed.

### Result

Let:

- `X` have global logical shape `[1, 1, 32, D]`.
- `W1` and `W3` have global logical shape `[D, H]`.
- `(r, c)` be the logical/tensor mesh coordinate, with `0 <= r < 8` and
  `0 <= c < 4`.
- `Kc = D / 4` and `Nr = H / 8`.

The exact logical shards at mesh coordinate `(r, c)` are:

```python
x_rc  = X[..., c * Kc : (c + 1) * Kc]
w1_rc = W1[c * Kc : (c + 1) * Kc, r * Nr : (r + 1) * Nr]
w3_rc = W3[c * Kc : (c + 1) * Kc, r * Nr : (r + 1) * Nr]
```

`x_rc` is replicated over `r`; changing `r` with fixed `c` does not change the
input shard. The two raw primitive outputs are:

```python
raw_output_0_rc = torch.matmul(x_rc, w1_rc)
raw_output_1_rc = torch.matmul(x_rc, w3_rc)
```

There is no transpose in either formula. Both `transpose_a` and `transpose_b`
default to `false` and are passed unchanged into matmul attribute creation.

These are partial products over one quarter of K. The fully reduced projection
block for mesh row `r` is therefore:

```python
w1_row_r = sum(raw_output_0_rc for c in range(4))
w3_row_r = sum(raw_output_1_rc for c in range(4))
```

Consequently, comparing a raw output directly with a slice of `X @ W1` or
`X @ W3` is incorrect. The sum across the four mesh columns must be compared
with the row's output-feature slice.

For the two test cases:

| Case | `D` | `H` | `Kc` | `Nr` | Raw logical output shape per device |
|---|---:|---:|---:|---:|---|
| Llama | 8192 | 28672 | 2048 | 3584 | `[1, 1, 32, 3584]` |
| Qwen | 5120 | 25600 | 1280 | 3200 | `[1, 1, 32, 3200]` |

DRAM/L1 shard padding is a storage detail and does not add logical columns to
the Torch references. Any padded tail beyond `Nr` must be excluded from PCC.

### Coordinate and ordering evidence

1. The hardware test creates W1/W3 with placements
   `[PlacementShard(-1), PlacementShard(-2)]` on `MeshShape(8, 4)`. Thus mesh
   axis 0 selects tensor dimension `-1` (H/N), and mesh axis 1 selects tensor
   dimension `-2` (D/K):
   `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:188-210`.
2. Decode input loading uses placements
   `[PlacementReplicate(), PlacementShard(-1)]`, so mesh axis 0 replicates X
   and mesh axis 1 selects its D/K slice:
   `models/common/modules/mlp/mlp_2d.py:920-940`.
3. Mapper construction gathers shard dimensions in mesh-axis order, chunks by
   those tensor dimensions, then assigns chunks while incrementing the final
   shard index fastest. This makes `(r, c)` select H chunk `r` and K chunk `c`:
   `ttnn/core/distributed/distributed_tensor.cpp:195-207`,
   `ttnn/core/distributed/distributed_tensor.cpp:234-264`.
4. `chunk_ndim` normalizes negative dimensions and advances its final chunk
   dimension fastest, confirming the same `(H chunk r, K chunk c)` ordering:
   `ttnn/core/tensor/xtensor/partition.cpp:46-61`,
   `ttnn/core/tensor/xtensor/partition.cpp:68-114`.
5. Because the mapper override `(8, 4)` equals and fits the device shape,
   distribution mode is `SUBMESH`; this preserves distribution coordinates
   exactly instead of reshaping them in row-major order:
   `ttnn/core/distributed/distribution_mode.cpp:11-29`,
   `ttnn/core/distributed/distributed_tensor.cpp:41-49`.
6. Mesh coordinates are row-major, outer row then inner column. The topology
   flattening uses distribution strides, and tests confirm that the first
   `num_cols` entries returned by `get_device_tensors` are mesh row 0:
   `tt_metal/api/tt-metalium/mesh_coord.hpp:21-29`,
   `tt_metal/impl/tensor/topology/tensor_topology.cpp:77-95`,
   `tests/ttnn/unit_tests/gtests/tensor/test_distributed_tensor.cpp:281-325`.
   Therefore list index `i` corresponds to `(r, c) = (i // 4, i % 4)`, or
   `i = r * 4 + c`.
7. This ordering is mesh-coordinate ordering, not a promise that physical chip
   ID equals `r * 4 + c`. The physical device for a logical mesh coordinate is
   obtained from the mesh view/device mapping. No additional permutation is
   introduced by this mapper because `SUBMESH` preserves `(r, c)`.

### Primitive output evidence

1. The common module calls the primitive as `(input, W1, ..., second_weight=W3)`
   and unpacks `(first_projection, w3_projection, w1_reduced)`:
   `models/common/modules/mlp/mlp_2d.py:337-374`.
2. With a second weight, the primitive creates matmul outputs in input-weight
   order, chooses output 0 as the reduce-scatter source, and returns both raw
   matmul specs before the reduced result:
   `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp:53-83`,
   `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp:161-195`.
3. The multi-weight matmul path builds its B tensor vector from input tensors
   beginning at index 1 and produces one output per B tensor in that order:
   `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:5689-5696`,
   `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2439-2449`.
4. Python binding defaults both transpose flags to false, and the primitive
   passes them directly to `create_matmul_attributes`:
   `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/rs_matmul_nanobind.cpp:59-74`,
   `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp:141-159`.

### Diagnostic implication

For raw-output diagnostics, iterate `ttnn.get_device_tensors(output)` using
`i = r * 4 + c` and compare each shard against the corresponding partial-product
formula above. For reduced-output diagnostics, sum the four raw column partials
or compare against the appropriate H/N row slice only after axis-1 reduction.
