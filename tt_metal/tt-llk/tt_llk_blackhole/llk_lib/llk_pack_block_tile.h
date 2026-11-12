// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The per-tile pack of a CB output, llk_pack_block's fallback; the includer provides DST_SYNC_MODE and pack_dst_format.

#include <cstdint>

#include "llk_pack.h"

// Under full DEST sync the packer is the bound, so the default pack writes the destination by bytes; under half sync
// the earlier pack start costs the unpackers L1 bank grants in latency-bound kernels.
template <PackMode pack_mode>
inline constexpr bool llk_pack_rmw_tile = DST_SYNC_MODE == DstSync::SyncFull && pack_mode == PackMode::Default;

// Unified cores, shared by the CB-id API below and the LLKOperand API (experimental/). They take
// already-resolved scalar formats/geometry + the runtime write address; the per-source prologue
// (resolving these from a CB id, or from an MemDescriptor) lives in the callers.
template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, bool skip_packer_strides = false>
inline void llk_pack_init_impl(
    const std::uint32_t pack_src_reg_format,
    const std::uint32_t face_r_dim,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const std::uint32_t num_tiles,
    const bool is_input_8bit_format)
{
    // Blackhole's MOP does not take the format at all; set_packer_strides is its only consumer here and
    // reconfig_packer_data_format re-derives that, so a format reconfig between init and execute is legal
    // and must not read as drift.
    SAN_HOOK(init<OperationPack>(
        StateDiscard<std::uint32_t>(pack_src_reg_format),
        StateVal<Operand<Exu::Pack>::FaceHeight>(face_r_dim),
        StateVal<Operand<Exu::Pack>::TileWidth>(tile_c_dim),
        StateVal<Operand<Exu::Pack>::NumFaces>(num_faces),
        StateDiscard<bool>(is_input_8bit_format),
        StateVal<OperationPack::NumTiles>(num_tiles)));

    _llk_pack_init_<pack_mode, zero_output, skip_addrmod_config, skip_packer_strides>(
        pack_src_reg_format, face_r_dim, tile_c_dim, num_faces, num_tiles, is_input_8bit_format);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default, bool w_invalidate = true>
inline void llk_pack_impl(std::uint32_t tile_index, std::uint32_t pack_tile_addr)
{
    _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, llk_pack_rmw_tile<pack_mode>, w_invalidate>(tile_index, pack_tile_addr);
}

// Float32 from a 16 bit DEST keeps main's write under full sync: the long pack hides the stall the byte-wise write
// saves, so its two extra config stores (0.1 cycle per tile) only cost.
template <bool is_fp32_dest_acc_en, PackMode pack_mode, bool w_invalidate = true>
inline void llk_pack_tile_of_output(std::uint32_t tile_index, std::uint32_t pack_tile_addr, const std::uint8_t output_id)
{
    if constexpr (llk_pack_rmw_tile<pack_mode> && !is_fp32_dest_acc_en)
    {
        if (pack_dst_format[output_id] == static_cast<std::uint32_t>(DataFormat::Float32))
        {
            _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en, pack_mode, false, false, w_invalidate>(tile_index, pack_tile_addr);
            return;
        }
    }
    llk_pack_impl<is_fp32_dest_acc_en, pack_mode, w_invalidate>(tile_index, pack_tile_addr);
}
