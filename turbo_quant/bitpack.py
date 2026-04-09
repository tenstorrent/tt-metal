"""Bit-packing utilities for TurboQuant index compression.

Packs b-bit quantization indices into uint8 tensors to achieve real memory savings:
  - 1-bit: 8 values per byte  (8x compression)
  - 2-bit: 4 values per byte  (4x compression)
  - 3-bit: 8 values per 3 bytes (2.67x compression)
  - 4-bit: 2 values per byte  (2x compression)

All functions operate on the last dimension and preserve all leading dimensions.
"""

from __future__ import annotations

import torch


def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack b-bit indices along the last dimension into uint8 bytes.

    Args:
        indices: Tensor of shape [..., N] with values in [0, 2^bits).
                 Must have N divisible by the packing group size.
        bits: Bit-width (1, 2, 3, or 4).

    Returns:
        Packed uint8 tensor of shape [..., packed_N].
    """
    idx = indices.to(torch.uint8)

    if bits == 1:
        # 8 values per byte
        N = idx.shape[-1]
        assert N % 8 == 0, f"Last dim ({N}) must be divisible by 8 for 1-bit packing"
        idx = idx.reshape(*idx.shape[:-1], N // 8, 8)
        packed = torch.zeros(*idx.shape[:-1], dtype=torch.uint8, device=idx.device)
        for i in range(8):
            packed |= (idx[..., i] & 0x1) << (7 - i)
        return packed

    elif bits == 2:
        # 4 values per byte
        N = idx.shape[-1]
        assert N % 4 == 0, f"Last dim ({N}) must be divisible by 4 for 2-bit packing"
        idx = idx.reshape(*idx.shape[:-1], N // 4, 4)
        packed = (
            ((idx[..., 0] & 0x3) << 6) | ((idx[..., 1] & 0x3) << 4) | ((idx[..., 2] & 0x3) << 2) | (idx[..., 3] & 0x3)
        )
        return packed

    elif bits == 3:
        # 8 values (24 bits) per 3 bytes
        N = idx.shape[-1]
        assert N % 8 == 0, f"Last dim ({N}) must be divisible by 8 for 3-bit packing"
        idx = idx.reshape(*idx.shape[:-1], N // 8, 8)

        # Pack 8 × 3-bit values into 3 bytes (24 bits)
        # Byte 0: [v0(3) v1(3) v2(2msb)]         = v0<<5 | v1<<2 | v2>>1
        # Byte 1: [v2(1lsb) v3(3) v4(3) v5(1msb)] = (v2&1)<<7 | v3<<4 | v4<<1 | v5>>2
        # Byte 2: [v5(2lsb) v6(3) v7(3)]          = (v5&3)<<6 | v6<<3 | v7
        b0 = ((idx[..., 0] & 0x7) << 5) | ((idx[..., 1] & 0x7) << 2) | ((idx[..., 2] & 0x7) >> 1)
        b1 = (
            (((idx[..., 2] & 0x1)) << 7)
            | ((idx[..., 3] & 0x7) << 4)
            | ((idx[..., 4] & 0x7) << 1)
            | ((idx[..., 5] & 0x7) >> 2)
        )
        b2 = ((idx[..., 5] & 0x3) << 6) | ((idx[..., 6] & 0x7) << 3) | (idx[..., 7] & 0x7)

        packed = torch.stack([b0, b1, b2], dim=-1)  # [..., N//8, 3]
        return packed.reshape(*packed.shape[:-2], -1)  # [..., N//8 * 3]

    elif bits == 4:
        # 2 values per byte
        N = idx.shape[-1]
        assert N % 2 == 0, f"Last dim ({N}) must be divisible by 2 for 4-bit packing"
        idx = idx.reshape(*idx.shape[:-1], N // 2, 2)
        packed = ((idx[..., 0] & 0xF) << 4) | (idx[..., 1] & 0xF)
        return packed

    else:
        raise ValueError(f"Unsupported bit-width: {bits}. Must be 1, 2, 3, or 4.")


def unpack(packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Unpack uint8 bytes back to b-bit indices along the last dimension.

    Args:
        packed: Packed uint8 tensor from pack().
        bits: Bit-width (1, 2, 3, or 4).
        num_elements: Original number of elements in the last dimension.

    Returns:
        Tensor of shape [..., num_elements] with values in [0, 2^bits).
    """
    if bits == 1:
        groups = packed.reshape(*packed.shape[:-1], -1, 1)  # [..., N//8, 1]
        groups = groups.expand(*groups.shape[:-1], 8)  # [..., N//8, 8]
        shifts = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device=packed.device, dtype=torch.uint8)
        unpacked = (groups >> shifts) & 0x1
        return unpacked.reshape(*packed.shape[:-1], -1)[..., :num_elements]

    elif bits == 2:
        groups = packed.reshape(*packed.shape[:-1], -1, 1)
        groups = groups.expand(*groups.shape[:-1], 4)
        shifts = torch.tensor([6, 4, 2, 0], device=packed.device, dtype=torch.uint8)
        unpacked = (groups >> shifts) & 0x3
        return unpacked.reshape(*packed.shape[:-1], -1)[..., :num_elements]

    elif bits == 3:
        # Reverse of 3-bit packing: 3 bytes → 8 values
        packed_3 = packed.reshape(*packed.shape[:-1], -1, 3)  # [..., N//8, 3]
        b0, b1, b2 = packed_3[..., 0], packed_3[..., 1], packed_3[..., 2]

        v0 = (b0 >> 5) & 0x7
        v1 = (b0 >> 2) & 0x7
        v2 = ((b0 & 0x3) << 1) | ((b1 >> 7) & 0x1)
        v3 = (b1 >> 4) & 0x7
        v4 = (b1 >> 1) & 0x7
        v5 = ((b1 & 0x1) << 2) | ((b2 >> 6) & 0x3)
        v6 = (b2 >> 3) & 0x7
        v7 = b2 & 0x7

        unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
        return unpacked.reshape(*packed.shape[:-1], -1)[..., :num_elements]

    elif bits == 4:
        groups = packed.reshape(*packed.shape[:-1], -1, 1)
        groups = groups.expand(*groups.shape[:-1], 2)
        shifts = torch.tensor([4, 0], device=packed.device, dtype=torch.uint8)
        unpacked = (groups >> shifts) & 0xF
        return unpacked.reshape(*packed.shape[:-1], -1)[..., :num_elements]

    else:
        raise ValueError(f"Unsupported bit-width: {bits}. Must be 1, 2, 3, or 4.")


def packed_size(num_elements: int, bits: int) -> int:
    """Compute the number of uint8 bytes needed to store num_elements at given bit-width."""
    if bits == 1:
        return (num_elements + 7) // 8
    elif bits == 2:
        return (num_elements + 3) // 4
    elif bits == 3:
        assert num_elements % 8 == 0, "3-bit packing requires num_elements divisible by 8"
        return (num_elements // 8) * 3
    elif bits == 4:
        return (num_elements + 1) // 2
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")


def compression_ratio(bits: int) -> float:
    """Theoretical compression ratio of packed b-bit indices vs uint8 storage."""
    return 8.0 / bits
