// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the tilized-output embedding program factory: gathers each block's weight rows into the
// row-major staging buffer that the tilize compute kernel drains. Bound through Metal 2.0 named
// bindings (dfb:: / tensor:: / args::), so the buffer indices and argument slots are supplied by the
// host KernelSpec rather than positional compile-time and runtime args.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "embeddings_common_metal2.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    const uint32_t input_start_id = get_arg(args::input_start_id);
    const uint32_t input_start_offset = get_arg(args::input_start_offset);
    const uint32_t weight_offset = get_arg(args::weight_offset);
    const uint32_t num_blocks = get_arg(args::num_blocks);

    constexpr uint32_t input_page_size = get_arg(args::input_page_size);
    constexpr uint32_t weight_block_size = get_arg(args::weight_block_size);
    constexpr uint32_t tiles_per_chunk = get_arg(args::tiles_per_chunk);
    constexpr uint32_t input_block_size_bytes = get_arg(args::input_block_size_bytes);
    constexpr uint32_t num_chunks = get_arg(args::num_chunks);
    constexpr uint32_t last_chunk_tiles = get_arg(args::last_chunk_tiles);

    auto input = TensorAccessor(tensor::input);
    // `weight_offset` is this core's column slice of each weight row (non-zero only for width-/
    // block-sharded output). It is applied as a per-read offset below, not folded into the
    // accessor base, so both accessors sit on their buffers' clean base addresses.
    auto weights = TensorAccessor(tensor::weights);

    // The local weight cache exists only for the embeddings types that serve some weight rows out of
    // local SRAM, so it is bound (and named) only on those builds.
#if defined PADDED
    prepare_local_cache(
        noc,
        scratch::local_cache,
        weights,
        weight_block_size,
        get_arg(args::pad_token),
        /*weight_col_offset_bytes=*/weight_offset);
#elif defined BINARY
    prepare_local_cache(
        noc,
        scratch::local_cache,
        weights,
        weight_block_size,
        /*pad_token_value=*/0,
        /*weight_col_offset_bytes=*/weight_offset);
#endif

    // dfb_in0 stages the weight rows this reader gathers for one chunk; the compute kernel tilizes them
    // out of it.
    DataflowBuffer dfb_in0(dfb::in0);
    // The index scratch is this reader's private page for the block of indices it is working through:
    // the NoC reads into it via the scratchpad object and the RISC reads the indices back.
    Scratchpad<volatile input_token_t> index_scratch(scratch::in1);

    // Per-row byte counts for the full and (possibly partial) last chunk.
    constexpr uint32_t num_tiles_per_block = (num_chunks - 1) * tiles_per_chunk + last_chunk_tiles;
    constexpr uint32_t bytes_per_tile_row = weight_block_size / num_tiles_per_block;
    constexpr uint32_t full_chunk_bytes = tiles_per_chunk * bytes_per_tile_row;
    constexpr uint32_t last_chunk_bytes = last_chunk_tiles * bytes_per_tile_row;

#ifdef SERVER_ORDER
    // NC.9 (deepseek_v41_flash_d_p): the input is every mesh row's ids in the server's order ([SP, 1, W], a page a
    // row); this chip's block i is natural unit d (32 ids) of the chunk, read where tt/page_copy.IdsReorder's kernel
    // (chunk_program ids_reorder.cpp) reads it, positions >= valid and ids > clamp_max read as clamp_max.
    constexpr uint32_t WU = get_arg(args::server_units);  // 32-id units a mesh row (W / 32)
    constexpr uint32_t SPR = get_arg(args::server_rows);  // mesh rows
    constexpr uint32_t VMAX = get_arg(args::clamp_max);   // vocab - 1 (the clamp and the pad id)
    const uint32_t me = get_arg(args::server_me);         // this mesh row
    const uint32_t iu = get_arg(args::server_iu);         // (start mod W) / 32 (per call)
    const uint32_t cs = get_arg(args::server_cs);         // (start / W) mod SP (per call)
    const uint32_t valid = get_arg(args::server_valid);   // natural positions below it are ids (per call)
    constexpr uint32_t SU = WU * SPR;                     // units a chunk
    const uint32_t u0 = input_start_offset / input_block_size_bytes;
#endif
    uint32_t curr_row = input_start_id;
    uint32_t offset = input_start_offset;
    for (uint32_t i = 0; i < num_blocks; ++i) {
#ifdef SERVER_ORDER
        const uint32_t d = (me * WU) + u0 + i;  // the natural unit
        if (d + iu < WU) {                      // the start row's head
            curr_row = cs;
            offset = d * input_block_size_bytes;
        } else if (d + iu >= SU) {  // the start row's tail: the chunk's last iu units
            curr_row = cs;
            offset = (d + WU - SU) * input_block_size_bytes;
        } else {
            const uint32_t e = d + iu;
            curr_row = (e / WU + cs) % SPR;
            offset = (e % WU) * input_block_size_bytes;
        }
#endif
#ifdef ARCH_QUASAR
        noc.async_read(input, index_scratch, input_block_size_bytes, {.page_id = curr_row, .offset_bytes = offset}, {});
#else
        noc.async_read<NocOptions::DEFAULT, input_block_size_bytes>(
            input, index_scratch, input_block_size_bytes, {.page_id = curr_row, .offset_bytes = offset}, {});
#endif
        noc.async_read_barrier();

        for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
            const bool is_last = (chunk + 1 == num_chunks);
            const uint32_t this_chunk_tiles = is_last ? last_chunk_tiles : tiles_per_chunk;
            const uint32_t weight_chunk_size = is_last ? last_chunk_bytes : full_chunk_bytes;
            const uint32_t weight_chunk_offset = chunk * full_chunk_bytes;

            dfb_in0.reserve_back(this_chunk_tiles);
            uint32_t l1_write_addr = dfb_in0.get_write_ptr();

            for (uint32_t k = 0; k < tile_height; ++k) {
                input_token_t token;
                {
                    // The NoC rewrites this scratchpad with each new index block, so on Quasar the DM
                    // core's L2 may still hold the previous block; scoped_lock invalidates the line
                    // before the read. No-op on WH/BH.
                    auto index_lock = index_scratch.scoped_lock(k, 1);
                    token = index_scratch[k];
                }
#ifdef SERVER_ORDER
                if ((d * tile_height) + k >= valid || token > VMAX) {
                    token = VMAX;
                }
#endif
                read_token_async(
                    noc, token, weights, l1_write_addr, weight_chunk_size, weight_chunk_offset, weight_offset);
                l1_write_addr += weight_chunk_size;
            }
            noc.async_read_barrier();
            dfb_in0.push_back(this_chunk_tiles);
        }

#ifndef SERVER_ORDER
        offset += input_block_size_bytes;
        if (offset == input_page_size) {
            offset = 0;
            curr_row++;
        }
#endif
    }
}
