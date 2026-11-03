# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Offline preparation for :func:`ttnn.experimental.dequantize_mxfp8`."""

E4M3_NAN_CODES = (0x7F, 0xFF)
E8M0_NAN_CODE = 0xFF
BLOCK_SIZE = 32


def pack(codes, scales, *, dtype=None):
    """Arrange E4M3 bytes and E8M0 block scales into the dequantize_mxfp8 contract.

    ``codes`` is a 2D host integer tensor of E4M3 bit patterns in [0, 255],
    excluding the NaN encodings 0x7F and 0xFF. Its height must be divisible by
    32 and its width by 32 (uint8) or 64 (uint16). ``scales`` has shape
    [height, width // 32]: one E8M0 code in [0, 254] per block of 32 horizontally
    adjacent values (OCP MX). The value is E4M3(code) * 2^(scale - 127).

    ``dtype`` selects the device container, ttnn.uint16 by default.
    For ttnn.uint8, the packed tensor equals ``codes``: the E4M3 bytes stay in
    their natural positions. For ttnn.uint16, output tile 2k+p of each packed
    tile row comes from byte p of each word of packed tile k (tile-plane packing,
    least significant byte first), mirroring nvfp4.pack.

    Returns a pair of host int32 tensors: the packed data, to upload with the
    selected container dtype in TILE or ROW_MAJOR layout, and the scales with
    shape [packed_tile_count, 32 * planes], to upload as ttnn.uint8 ROW_MAJOR.
    Scale rows follow packed tiles in row-major tile order; within a row the
    order is byte plane, then output tile row. Codes and scales are rearranged
    without numeric conversion.

    This is offline host preparation; keep it outside device tracing.
    """
    import torch
    import ttnn

    if dtype is None:
        dtype = ttnn.uint16
    if dtype not in (ttnn.uint8, ttnn.uint16):
        raise ValueError("MXFP8 container dtype must be ttnn.uint8 or ttnn.uint16")
    planes = 1 if dtype == ttnn.uint8 else 2
    if codes.ndim != 2:
        raise ValueError("MXFP8 pack expects a 2D tensor of E4M3 integer codes")
    if codes.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("MXFP8 pack expects host tensors")
    height, width = codes.shape
    if height == 0 or width == 0 or height % 32 or width % (32 * planes):
        raise ValueError(f"MXFP8 height must be a positive multiple of 32 and width of {32 * planes}")
    if codes.is_floating_point() or codes.is_complex() or bool(((codes < 0) | (codes > 255)).any()):
        raise ValueError("MXFP8 data must contain integer E4M3 codes in [0, 255]")
    if bool(((codes == E4M3_NAN_CODES[0]) | (codes == E4M3_NAN_CODES[1])).any()):
        raise ValueError("MXFP8 data must not contain E4M3 NaN codes 0x7F or 0xFF")
    if tuple(scales.shape) != (height, width // BLOCK_SIZE):
        raise ValueError("MXFP8 scales must have shape [height, width // 32]")
    if scales.is_floating_point() or scales.is_complex() or bool(((scales < 0) | (scales > 254)).any()):
        raise ValueError("MXFP8 scales must contain integer E8M0 codes in [0, 254] (0xFF is NaN)")
    tile_rows, packed_tile_cols = height // 32, width // (32 * planes)
    codes = codes.to(torch.int64)
    if planes == 1:
        packed = codes
    else:
        # [tile row, packed tile column, row, column, plane]
        tiles = codes.reshape(tile_rows, 32, packed_tile_cols, planes, 32).permute(0, 2, 1, 4, 3)
        packed = sum(tiles[..., plane] << (8 * plane) for plane in range(planes))
        packed = packed.permute(0, 2, 1, 3).reshape(height, width // planes)
    # Each 32-value block is one row of one output tile.
    # [tile row, packed tile column, plane, output tile row]
    ordered_scales = (
        scales.to(torch.int64)
        .reshape(tile_rows, 32, packed_tile_cols, planes)
        .permute(0, 2, 3, 1)
        .reshape(tile_rows * packed_tile_cols, planes * 32)
    )
    return packed.to(torch.int32).contiguous(), ordered_scales.to(torch.int32).contiguous()
