# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pack_fp8_rows: per-token fp8 rows -> Blaze decode's 192-byte indexer key cache rows
(deepseek_v41_flash_d_p P.39, INDEX_K_SERVED=fp8_row192).

    e4m3  [.., M, 128] FP8_E4M3 ROW_MAJOR  and  scale [.., M, 1] FLOAT32 ROW_MAJOR
          (ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False) of a 128-wide x)
    ->    out [.., M, 192] UINT8 ROW_MAJOR, row m = [e4m3 row m (128 B) | scale m as fp32 little endian (4 B) | 60 zero B]

The layout of tt-blaze's DsaIndexerDirectKCacheUpdate (blaze/ops/dsa_indexer_direct_k_cache_update, SCALED_ROW_BYTES
192; tests/blaze/micro_ops/dsa/test_dsa_indexer_direct_k_cache_update.py checks bytes 0..131 against
per_token_cast_to_fp8). A row whose 128 values are all +-0 (a zeroed pad row) is written all zero: the stock packer of
the GLM rows (ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache: 512 B latent | 4 scales | 64 bf16 RoPE)
copies the clamp-floor scale (1e-4 / 448) there. A data-movement program only (one kernel, NoC 0): rows split over
the cores by common runtime args (no per-core args, DS41-145). pack_fp8_rows is the native C++ operation
ttnn.pack_fp8_rows (program cached by the framework; a cache hit rewrites the three addresses).
"""

from __future__ import annotations

import ttnn

VALUES, ROW_BYTES = 128, 192


def _grid(device, max_grid=None) -> tuple[int, int]:
    """The compute grid the op uses: the device's, cut to ``max_grid`` (a Galaxy's 110-core cap, agent rule 18)."""
    g = device.compute_with_storage_grid_size()
    if max_grid is None:
        return g.x, g.y
    mx, my = (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)
    return max(1, min(g.x, mx)), max(1, min(g.y, my))


def _mg_key(max_grid):
    if max_grid is None:
        return None
    return (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)


def _rows(t: ttnn.Tensor) -> int:
    s = t.shape
    r = 1
    for i in range(len(s) - 1):
        r *= s[i]
    return r


def _interleaved_dram(t: ttnn.Tensor, what: str) -> None:
    mc = t.memory_config()
    assert (
        mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"pack_fp8_rows: {what} must be DRAM interleaved, got {mc}"


# pack_fp8_rows(e4m3, scale, *, output_tensor=None, memory_config=None, max_grid=None): the registered C++ operation
# (ttnn/cpp/ttnn/operations/pack_fp8_rows), which checks the same asserts (raised as AssertionError).
pack_fp8_rows = ttnn.pack_fp8_rows


# ---------------------------------------------------------------------------------------------------------------------
# ttnn.bringup.unpack_fp8_rows: the inverse reader (deepseek_v41_flash_d_p P.76, INDEX_K_SOURCE=served)


def __getattr__(name):
    """``_PREPARED``: the keys of the programs ttnn.unpack_fp8_rows has built, as the Python op recorded them
    (("unpack", mesh id, rows shape, out shape, str(rows mc), str(out mc), ncores, max_grid, common_values)),
    built from the C++ op's record only when read."""
    if name == "_PREPARED":
        from ttnn._ttnn.operations.unpack_fp8_rows import unpack_fp8_rows_program_keys

        return {key: None for key in unpack_fp8_rows_program_keys()}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# unpack_fp8_rows(rows, out, *, row_lo, row_hi, batch=0, out_row0=None, max_grid=None, whole_grid=False,
# common_values=False): the registered C++ operation (ttnn/cpp/ttnn/operations/unpack_fp8_rows), which checks the
# same asserts (raised as AssertionError) and returns ``out``.
unpack_fp8_rows = ttnn.unpack_fp8_rows
