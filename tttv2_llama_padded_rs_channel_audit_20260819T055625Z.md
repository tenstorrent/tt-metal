# Llama Padded Reduce-Scatter Channel Audit

Timestamp: 2026-08-19 05:56:25 UTC

## Scope

Source-only audit of logical channel ownership and ordering for:

- `ttnn.experimental.llama_reduce_scatter`
- `ttnn.experimental.llama_rs_matmul` output 2 when `second_weight_tensor` is supplied

Case audited: mesh `cluster_axis=1`, four devices per mesh row, input logical width 3584, width-sharded over 24 cores with a 160-channel (five-tile) shard, and padded reduce-scatter output width 960 per device. No TT hardware was used. No production or test file was edited.

## Conclusion

The primitive treats the input as one row-major channel stream padded on its global right edge:

```text
logical:  [0, 3584)
physical: [0, 3840) = [0, 3584) valid + [3584, 3840) padding
```

It then assigns four contiguous 960-channel slices by ring index. It does **not** split 3584 into four 896-channel logical chunks and insert 64 padding channels into every device output.

| Ring index / mesh column | Output positions | Source channel interval | Valid channels | Padding channels |
|---:|---:|---:|---:|---:|
| 0 | `[0, 960)` | `[0, 960)` | 960 | 0 |
| 1 | `[0, 960)` | `[960, 1920)` | 960 | 0 |
| 2 | `[0, 960)` | `[1920, 2880)` | 960 | 0 |
| 3 | `[0, 960)` | `[2880, 3840)` | 704 (`[2880, 3584)`) | 256 (`[3584, 3840)`) |

Therefore the often-inferred `960 - 896 = 64` padding channels per column do not exist as four separate padding regions. All four nominal 64-channel excesses are accumulated into the final 256 channels of ring-index 3. The padding is at the **end of the globally concatenated stream**, and at the **end of device 3's local output**. It is neither at the beginning nor interleaved.

## Source Derivation

1. `LlamaReduceScatterDeviceOperation::compute_output_specs` computes 112 logical width tiles for 3584 channels, rounds that to 120 tiles from the 24-core input grid, and divides by four devices. The result is 30 tiles, or 960 channels, per device. See `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp`, especially the explicit 3584-to-3840 comment and `padded_input_width / ring_devices` calculation around lines 64-84.

2. The program factory derives `input_shard_cores_per_device = 24 / 4 = 6`. With five tiles per input core, each device owns `6 * 5 = 30` consecutive pages/tiles. See `llama_reduce_scatter_program_factory.cpp` around lines 400-419.

3. For each target device, the local input page offset is:

   ```text
   offset_for_input = chip_id * input_shard_cores_per_device * input_tiles_per_core_width
                    = chip_id * 6 * 5
                    = chip_id * 30 tiles
   ```

   See `llama_reduce_scatter_program_factory.cpp` around lines 744-745 and 826-835. This is the decisive ownership rule: ring indices 0, 1, 2, and 3 start at tiles 0, 30, 60, and 90 respectively.

4. Output writeback is linear. Packet workers receive `local_page` values 0, packet-size, and so on, and the writer maps each linear page to `output_core = page / output_tiles_per_core_width` and `tile_offset = page % output_tiles_per_core_width`. See `writer_llama_reduce_scatter.cpp` and the runtime-argument setup in `llama_reduce_scatter_program_factory.cpp` around lines 826-835. There is no channel permutation or per-device pad insertion in this path.

5. Ring index is the target device's position in `mesh_view.get_devices_on_row(mesh_coordinate[0])` for `cluster_axis=1`; it is not the physical PCIe/device ID. See `llama_reduce_scatter_program_factory.cpp` around lines 359-389. On the normal rectangular `(8, 4)` logical mesh this corresponds to mesh columns 0 through 3 within each row.

6. Fused `llama_rs_matmul` output 2 uses the same `LlamaReduceScatterDeviceOperation`. With two weights, the first matmul output becomes `new_rs_tensor`, and the fused program passes return tensor index 2 to `create_at_program_processing`. See `llama_reduce_scatter_matmul/device/rs_matmul_op.cpp` around lines 53-83 and 161-186, plus `llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp` around lines 43-70. Consequently output 2 has exactly the same channel ownership and ordering as standalone `llama_reduce_scatter`.

## Correct Torch Reconstruction

For one fixed mesh row, let `partial[c]` be the 3584-channel pre-reduction result resident on mesh column `c`. For fused matmul, this is the per-column partial product over that column's K shard. The unpadded Torch reduction is:

```python
reduced = sum(partial[c] for c in range(4))  # [..., 3584]
```

Construct the physical reference by right-padding once, then slicing by ring index:

```python
import torch.nn.functional as F

padded = F.pad(reduced, (0, 3840 - 3584))   # [..., 3840]
expected_by_ring = [
    padded[..., ring_index * 960 : (ring_index + 1) * 960]
    for ring_index in range(4)
]
```

When extracting device tensors, map each tensor to its logical mesh coordinate and then to its ring index. Do not rely on a flat device list being ordered by physical device ID. Compare as follows:

```python
# actual_by_ring[i] is the local [..., 960] Torch tensor for ring index i.
for i in (0, 1, 2):
    compare(actual_by_ring[i], expected_by_ring[i])

compare(actual_by_ring[3][..., :704], expected_by_ring[3][..., :704])
```

The final 256 values on ring index 3 are physical padding. Compare them to zero only if the producer contract explicitly guarantees zero-filled output padding. Otherwise exclude them from PCC/allclose because logical correctness does not assign values to those channels.

An equivalent whole-row check is:

```python
actual_padded = torch.cat(actual_by_ring, dim=-1)  # [..., 3840]
compare(actual_padded[..., :3584], reduced)
# actual_padded[..., 3584:] is the 256-channel padding tail.
```

## Incorrect Comparison Pattern

This reference is wrong for these primitives:

```python
logical_chunks = torch.chunk(reduced, 4, dim=-1)  # four [..., 896] tensors
expected = [F.pad(chunk, (0, 64)) for chunk in logical_chunks]
```

It assumes per-column padding and shifts valid channels after column 0. In particular, actual ring index 1 starts at logical channel 960, not 896; ring index 2 starts at 1920, not 1792; and ring index 3 starts at 2880, not 2688. Such a comparison can produce one comparatively better first-column PCC and near-zero PCC on later columns even when the reduce-scatter ordering itself is correct.

## Confidence And Caveat

Confidence is high because output shape computation, source-page offsets, writeback indexing, and fused-op wiring all agree. The one caveat is the value of the physical padding tail: the audited reduce-scatter code moves and reduces those pages but does not itself establish a logical zero-value contract for them. Placement is exact; padding values should be treated as don't-care unless the upstream matmul/input preparation proves they are zero.
