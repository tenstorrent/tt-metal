# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Offline preparation for :func:`ttnn.experimental.dequantize_nvfp4`."""


def pack(codes, scales, *, dtype=None, experimental_fast=False, scale_format=None):
    """Arrange integer E2M1 codes and block-scale bytes into tile planes.

    ``codes`` is a 2D host integer tensor with values in [0, 15]. Its height
    must be divisible by 32 and width by 64 (uint8) or 128 (uint16). ``scales``
    has shape [height, width // 16], with one encoded scale per horizontal
    block. For 16x16 scaling, repeat each block scale across its 16 rows first.

    ``dtype`` selects the device container, ttnn.uint16 by default. The returned
    pair of host int32 tensors can be passed to ``ttnn.from_torch`` with the
    selected container dtype and ttnn.uint8, respectively. The integer codes
    are rearranged without numeric conversion or quantization. Scale bytes
    are rearranged without decoding; their allowed range depends on the
    scale_format passed to dequantize_nvfp4.

    With ``experimental_fast=True``, use uint16 storage, transpose each output
    tile before packing, and return compact BF16 scale rows in transposed face
    order. E4M3/E2M1 scales are decoded exactly on the host; ``scale_format``
    defaults to Nvfp4ScaleFormat.E4M3. Upload the second return value as BF16
    ROW_MAJOR, and call dequantize_nvfp4 with experimental_fast=True.

    This is offline host preparation; keep it outside device tracing.
    """
    import torch
    import ttnn

    if dtype is None:
        dtype = ttnn.uint16
    if dtype not in (ttnn.uint8, ttnn.uint16):
        raise ValueError("NVFP4 container dtype must be ttnn.uint8 or ttnn.uint16")
    planes = 2 if dtype == ttnn.uint8 else 4
    if codes.ndim != 2:
        raise ValueError("NVFP4 pack expects a 2D tensor of E2M1 integer codes")
    height, width = codes.shape
    if height == 0 or width == 0 or height % 32 or width % (32 * planes):
        raise ValueError(f"NVFP4 height must be a positive multiple of 32 and width of {32 * planes}")
    if codes.is_floating_point() or codes.is_complex() or bool(((codes < 0) | (codes > 15)).any()):
        raise ValueError("NVFP4 data must contain integer E2M1 codes in [0, 15]")
    if tuple(scales.shape) != (height, width // 16):
        raise ValueError("NVFP4 scales must have shape [height, width // 16]")
    if scales.is_floating_point() or scales.is_complex() or bool(((scales < 0) | (scales > 255)).any()):
        raise ValueError("NVFP4 scales must contain integer byte codes in [0, 255]")
    if codes.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("NVFP4 pack expects host tensors")
    tile_rows, packed_tile_cols = height // 32, width // (32 * planes)
    tiles = codes.reshape(tile_rows, 32, packed_tile_cols, planes, 32).permute(0, 2, 1, 4, 3).to(torch.int64)
    if experimental_fast:
        if dtype != ttnn.uint16:
            raise ValueError("experimental_fast requires ttnn.uint16")
        # Transpose every output 32x32 tile before nibble packing. Restore with
        # the FPU in DEST after column-aligned group scaling.
        tiles = tiles.transpose(2, 3)
    packed = sum(tiles[..., plane] << (4 * plane) for plane in range(planes))
    packed = packed.permute(0, 2, 1, 3).reshape(height, width // planes).to(torch.int32)
    # [tile row, packed tile column, plane, vertical face, horizontal face, face row]
    ordered_scales = (
        scales.reshape(tile_rows, 2, 16, packed_tile_cols, planes, 2)
        .permute(0, 3, 4, 1, 5, 2)
        .contiguous()
        .reshape(tile_rows * packed_tile_cols, planes * 64)
        .to(torch.int32)
    )
    if experimental_fast:
        if scale_format is None:
            scale_format = ttnn.experimental.Nvfp4ScaleFormat.E4M3
        if scale_format == ttnn.experimental.Nvfp4ScaleFormat.E4M3:
            if bool((ordered_scales > 126).any()):
                raise ValueError("E4M3 scales must be finite nonnegative codes in [0, 126]")
            exponent = ordered_scales >> 3
            mantissa = ordered_scales & 7
            decoded = torch.where(
                exponent == 0, mantissa.float() * 2**-9, torch.ldexp(1 + mantissa.float() / 8, exponent - 7)
            )
        elif scale_format == ttnn.experimental.Nvfp4ScaleFormat.E2M1:
            if bool((ordered_scales > 7).any()):
                raise ValueError("E2M1 scales must be nonnegative codes in [0, 7]")
            decoded = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[ordered_scales.long()]
        else:
            raise ValueError("Unsupported NVFP4 scale format")
        # A full tile transpose exchanges the top-right and bottom-left faces.
        ordered_scales = decoded.reshape(-1, planes, 2, 2, 16).transpose(2, 3).reshape(-1, planes * 64).bfloat16()
    return packed, ordered_scales
