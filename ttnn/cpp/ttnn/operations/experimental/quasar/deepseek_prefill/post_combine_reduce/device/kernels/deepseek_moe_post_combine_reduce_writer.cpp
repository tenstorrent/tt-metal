// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"

// L1 address the CPU accesses a scratchpad through: on Quasar DM the uncached alias, as DataflowBuffer pointers are,
// since the CPU caches are not coherent with NoC writes to L1; on WH/BH the plain address.
template <typename T>
uint32_t cpu_l1_addr(const Scratchpad<T>& scratch) {
    uintptr_t addr = local_mem::detail::AddressAccess::address(scratch.local_mem());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
    addr += MEM_L1_UNCACHED_BASE;
#endif
    return static_cast<uint32_t>(addr);
}

// The dispatch_table / indices metadata args are always emitted by the program factory; when the GPT-OSS
// (weight==0) path is selected, their values are zero, and the dispatch_table / indices tensors and DFBs are
// bound only on the USE_DISPATCH_TABLE_SKIP path, which guards the code that consults them.
constexpr uint32_t num_experts = get_arg(args::num_experts);
constexpr uint32_t emb_dim_dfb_tiles = get_arg(args::emb_dim_dfb_tiles);
constexpr uint32_t emb_dim_out_tiles = get_arg(args::emb_dim_out_tiles);
constexpr uint32_t dispatch_table_num_pages = get_arg(args::dispatch_table_num_pages);
constexpr uint32_t dispatch_table_aligned_page_size = get_arg(args::dispatch_table_aligned_page_size);
constexpr uint32_t indices_aligned_page_size = get_arg(args::indices_aligned_page_size);

// Per-token weights ([..., num_experts]) (WEIGHTS_PER_TOKEN): a page holds one token's weights; a chunk's 32 pages are
// read once into the wtok scratchpad and each weight is copied into its weights tile by this RISC, instead of one DRAM
// read per weight.
#ifdef WEIGHTS_PER_TOKEN
constexpr bool weights_per_token = true;
#else
constexpr bool weights_per_token = false;
#endif
constexpr uint32_t weights_aligned_page_size = get_arg(args::weights_aligned_page_size);
// Weights read per NoC barrier: num_experts (all of a token's weights at once) when the factory sized
// weights for two tokens of them, else 1 (weights holds a single tile). Always divides num_experts.
constexpr uint32_t weights_per_batch = get_arg(args::weights_per_batch);

constexpr uint32_t TOKENS_PER_CHUNK = 32;

void kernel_main() {
    uint32_t token_start_idx = get_arg(args::token_start_idx);
    uint32_t num_chunks = get_arg(args::num_chunks);

    Noc noc;
    DataflowBuffer dfb_weights(dfb::weights);
#ifdef USE_DISPATCH_TABLE_SKIP
    DataflowBuffer dfb_dispatch_table(dfb::dispatch_table);
    DataflowBuffer dfb_indices(dfb::indices);
#endif
    DataflowBuffer dfb_output(dfb::output);

    const uint32_t weight_tile_size = dfb_weights.get_tile_size();
    const uint32_t output_tile_size = dfb_output.get_tile_size();

    const auto weight_addrg = TensorAccessor(tensor::weight);
    const auto weight_tok_addrg = TensorAccessor(tensor::weight);
    uint32_t wtok_l1 = 0;
#ifdef WEIGHTS_PER_TOKEN
    {
        Scratchpad<uint32_t> wtok(scratch::wtok);
        wtok_l1 = cpu_l1_addr(wtok);
    }
#endif
    const auto output_addrg = TensorAccessor(tensor::output);

    uint32_t dispatch_table_write_addr = 0;

#ifdef USE_DISPATCH_TABLE_SKIP
    {
        const auto dispatch_table_addrg = TensorAccessor(tensor::dispatch_table);

        // Pre-load dispatch table into its DFB — read once, used by compute for all chunks
        dfb_dispatch_table.reserve_back(dispatch_table_num_pages);
        dispatch_table_write_addr = dfb_dispatch_table.get_write_ptr();
        for (uint32_t i = 0; i < dispatch_table_num_pages; i++) {
            noc.async_read(
                dispatch_table_addrg,
                dfb_dispatch_table,
                dispatch_table_aligned_page_size,
                {.page_id = i},
                {.offset_bytes = i * dispatch_table_aligned_page_size});
        }
        noc.async_read_barrier();
        dfb_dispatch_table.push_back(dispatch_table_num_pages);
    }
#endif

    constexpr uint32_t dfb_output_tiles = emb_dim_dfb_tiles * TOKENS_PER_CHUNK;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Load indices for this chunk into its DFB — one chunk at a time to save L1
        uint32_t indices_write_addr = 0;
#ifdef USE_DISPATCH_TABLE_SKIP
        {
            const auto indices_addrg = TensorAccessor(tensor::indices);
            dfb_indices.reserve_back(TOKENS_PER_CHUNK);
            indices_write_addr = dfb_indices.get_write_ptr();
            for (uint32_t i = 0; i < TOKENS_PER_CHUNK; i++) {
                noc.async_read(
                    indices_addrg,
                    dfb_indices,
                    indices_aligned_page_size,
                    {.page_id = token_start_idx + i},
                    {.offset_bytes = i * indices_aligned_page_size});
            }
            noc.async_read_barrier();
            // Indices stay as uint16 in the DFB — the compute kernel reads them
            // directly as uint16 tile values.
            dfb_indices.push_back(TOKENS_PER_CHUNK);
        }
#endif

        if constexpr (weights_per_token) {
            for (uint32_t i = 0; i < TOKENS_PER_CHUNK; ++i) {
                noc.async_read(
                    weight_tok_addrg,
                    CoreLocalMem<uint32_t>(wtok_l1 + i * weights_aligned_page_size),
                    weights_aligned_page_size,
                    {.page_id = token_start_idx + i},
                    {});
            }
            noc.async_read_barrier();
        }

        // Phase 1: Stream one weight per expert per token for this chunk
        for (uint32_t token_idx = 0; token_idx < TOKENS_PER_CHUNK; ++token_idx) {
            uint32_t global_token_idx = token_start_idx + token_idx;

            bool has_local = true;
#ifdef USE_DISPATCH_TABLE_SKIP
            {
                tt_l1_ptr int32_t* dispatch_table = reinterpret_cast<tt_l1_ptr int32_t*>(dispatch_table_write_addr);
                // Indices stay as uint16 in the DFB — no in-place expansion.
                tt_l1_ptr uint16_t* token_indices =
                    reinterpret_cast<tt_l1_ptr uint16_t*>(indices_write_addr + token_idx * indices_aligned_page_size);
                has_local = false;
                for (uint32_t k = 0; k < num_experts; ++k) {
                    if (dispatch_table[token_indices[k]] != -1) {
                        has_local = true;
                        break;
                    }
                }
            }
#endif

            // weights_per_batch weights under one barrier. With all of a token's weights per batch, weights
            // holds two tokens so this runs ahead of compute instead of alternating with it one tile at
            // a time; with one per batch it is the original single-tile stream.
            for (uint32_t expert_base = 0; expert_base < num_experts; expert_base += weights_per_batch) {
                dfb_weights.reserve_back(weights_per_batch);
                for (uint32_t e = 0; e < weights_per_batch; ++e) {
                    const uint32_t expert_idx = expert_base + e;
                    const uint32_t slot_offset = e * weight_tile_size;
#ifdef USE_DISPATCH_TABLE_SKIP
                    {
                        bool is_last = (expert_idx == num_experts - 1);
                        if (!has_local && is_last) {
                            // No local experts for this token — zero the weight tile so compute's
                            // must_zero_init multiply produces zeros regardless of combine_output.
                            volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                                dfb_weights.get_write_ptr() + slot_offset);
                            for (uint32_t w = 0; w < weight_tile_size / sizeof(uint32_t); w++) {
                                ptr[w] = 0;
                            }
                            continue;
                        }
                    }
#endif
                    if constexpr (weights_per_token) {
                        // The compute kernel reads element 0 (scalar broadcast; the zero test reads elements 0-1).
                        volatile tt_l1_ptr uint16_t* dst =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb_weights.get_write_ptr() + slot_offset);
                        const volatile tt_l1_ptr uint16_t* src = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(
                            wtok_l1 + token_idx * weights_aligned_page_size);
                        dst[0] = src[expert_idx];
                        dst[1] = 0;
                        continue;
                    }
                    uint32_t weight_page_idx = global_token_idx * num_experts + expert_idx;
                    noc.async_read(
                        weight_addrg,
                        dfb_weights,
                        weight_tile_size,
                        {.page_id = weight_page_idx},
                        {.offset_bytes = slot_offset});
                }
                noc.async_read_barrier();
                dfb_weights.push_back(weights_per_batch);
            }
        }

        // Phase 2: Write output tiles after compute finishes this chunk.
        // The output DFB holds TOKENS_PER_CHUNK * emb_dim_dfb_tiles tile-sized entries
        // (including padding when emb_dim is not 1024-aligned); only the first
        // emb_dim_out_tiles of them hold real data for this 32-token block.
        dfb_output.wait_front(dfb_output_tiles);

        uint32_t tile_row = token_start_idx / TOKENS_PER_CHUNK;
        uint32_t start_tile_idx = tile_row * emb_dim_out_tiles;

        for (uint32_t tile_idx = 0; tile_idx < emb_dim_out_tiles; ++tile_idx) {
            noc.async_write(
                dfb_output,
                output_addrg,
                output_tile_size,
                {.offset_bytes = tile_idx * output_tile_size},
                {.page_id = start_tile_idx + tile_idx});
        }

        noc.async_write_barrier();

        dfb_output.pop_front(dfb_output_tiles);

        token_start_idx += TOKENS_PER_CHUNK;
    }
}
