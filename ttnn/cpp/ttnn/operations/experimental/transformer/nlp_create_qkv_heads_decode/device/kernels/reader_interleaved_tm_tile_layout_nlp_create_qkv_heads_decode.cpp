// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    Noc noc;

    uint32_t in_tile_offset_by_batch = get_arg(args::in_tile_offset_by_batch);

    constexpr uint32_t ELEMENT_SIZE = get_arg(args::ELEMENT_SIZE);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_arg(args::SUBTILE_LINE_BYTES);
    constexpr uint32_t head_size = get_arg(args::head_size);
    constexpr uint32_t num_q_heads = get_arg(args::num_q_heads);
    constexpr uint32_t num_kv_heads = get_arg(args::num_kv_heads);
    constexpr uint32_t head_size_num_tiles = get_arg(args::head_size_num_tiles);
    constexpr uint32_t PHASES_TO_READ =
        get_arg(args::PHASES_TO_READ);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase
    // USE_ALIGNED_PATH is defined (by the host, via the kernel's compile defines) when the input
    // lives in DRAM and the per-face-row read size (SUBTILE_LINE_BYTES) is below the device DRAM
    // read alignment. In that regime the direct noc_async_read path violates the NOC alignment rule
    // ((src & (alignment-1)) == (dst & (alignment-1))) for half the (batch, head) parities, and
    // silently returns wrong data on Blackhole. The aligned path stages each read through an L1
    // scratch buffer sized to a DRAM-aligned chunk per tile, then copies the desired sub-tile-line
    // into the output buffer. The copy uses tt_memmove, which routes through the NOC datamover for
    // L1→L1 transfers when the source/destination 16B parities match (the common case here:
    // SUBTILE_LINE_BYTES is a multiple of 16, write_addr is multi-of-16-tiled, and scratch_base
    // is DRAM-aligned). When parities don't match it falls back to baby-RISC memmove. The
    // datamover path is dramatically faster than std::memcpy on Blackhole. See issue #43270 for
    // the original symptom.
    // Named DRAM_ALIGN_BYTES rather than DRAM_ALIGNMENT to avoid collision with the
    // DRAM_ALIGNMENT macro in tt_metal/hw/inc/internal/tt-1xx/*/noc/noc_parameters.h.
    constexpr uint32_t DRAM_ALIGN_BYTES = get_arg(args::DRAM_ALIGN_BYTES);
    constexpr uint32_t tile_size = head_size / head_size_num_tiles;

    constexpr uint32_t HALF_TILE_ELEMENTS = tt::constants::FACE_HEIGHT * tt::constants::TILE_WIDTH;
    // Phase 1 is the top-left face (Face 0) for face-row indices 0..15 / Face 2 for 16..31; phase 2
    // is the horizontally-adjacent face (Face 1 / Face 3). The byte offset between phase 1 and
    // phase 2 within a tile is one face's worth of elements (FACE_HEIGHT * FACE_WIDTH), not
    // half-a-tile's worth.
    constexpr uint32_t PHASE_OFFSET_BYTES = tt::constants::FACE_HEIGHT * tt::constants::FACE_WIDTH * ELEMENT_SIZE;

    const auto qkv_reader = TensorAccessor(tensor::qkv_in);

    // The q/k/v output shards are written in place through their local L1 base addresses.
    // Formerly borrowed DFBs whose write pointer was only ever read at its base — converted to
    // LocalTensorAccessor (sync_free_dfbs.md). Required on Quasar: DFB get_write_ptr() returns the
    // uncached L1 alias on Gen2 DM, which NOC APIs reject; get_bank_base_address() is the plain
    // (cached) shard base on both generations (identical to the old value on Gen1). T = uint8_t:
    // the kernel never indexes elements, it only hands byte addresses to NOC transfers.
    LocalTensorAccessor<uint8_t> q_out(tensor::q_out);
    LocalTensorAccessor<uint8_t> k_out(tensor::k_out);
    LocalTensorAccessor<uint8_t> v_out(tensor::v_out);

    uint32_t qkv_tile_id = 0;

#ifdef USE_ALIGNED_PATH
    {
        // Per-instance staging region. Formerly a self-looped DFB (this kernel bound both
        // PRODUCER and CONSUMER — a shape Gen2 rejects for DM kernels), converted to a Scratchpad
        // per sync_free_dfbs.md: the kernel never made a FIFO call on it, only grabbed the base
        // address once, so the conversion drops no synchronization.
        Scratchpad<uint8_t> aligned_scratch(scratch::aligned_scratch);
        constexpr bool read_phase_1 = (PHASES_TO_READ == 0 || PHASES_TO_READ == 1);
        constexpr bool read_phase_2 = (PHASES_TO_READ == 0 || PHASES_TO_READ == 2);
        // The NOC alignment rule requires (src & (alignment-1)) == (dst & (alignment-1)).
        // Source addresses are aligned to DRAM_ALIGN_BYTES, so the scratch destination must
        // also be aligned. L1 buffer allocations are only L1-aligned (16 B on BH), so round up the
        // scratch base; the program factory oversizes the region by one DRAM_ALIGN_BYTES chunk to
        // accommodate this rounding.
        const uint32_t raw_scratch_base = aligned_scratch.get_base_address();
        const uint32_t scratch_base = (raw_scratch_base + DRAM_ALIGN_BYTES - 1u) & ~(DRAM_ALIGN_BYTES - 1u);

        auto stage_phase = [&](uint32_t write_addr_base, uint32_t starting_tile_id, uint32_t phase_offset) {
            // skew is the bytes between the unaligned per-batch row and the DRAM-aligned chunk
            // start. PHASE_OFFSET_BYTES = 512 is a multiple of any plausible DRAM alignment, so
            // both phases share the same skew when in_tile_offset_by_batch is the same.
            const uint32_t skew = (in_tile_offset_by_batch + phase_offset) & (DRAM_ALIGN_BYTES - 1);
            const uint32_t aligned_offset = (in_tile_offset_by_batch + phase_offset) - skew;

            // Stage 1: issue DRAM-aligned reads into the per-tile scratch slots.
            uint32_t scratch_offset = scratch_base;
            uint32_t local_tile_id = starting_tile_id;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(scratch_offset),
                    DRAM_ALIGN_BYTES,
                    {.page_id = local_tile_id, .offset_bytes = aligned_offset},
                    {});
                scratch_offset += DRAM_ALIGN_BYTES;
                local_tile_id++;
            }
            noc.async_read_barrier();

            // Stage 2: copy the desired SUBTILE_LINE_BYTES slice from each scratch slot into the
            // output buffer at the per-tile destination offset.
            //
            // tt_memmove<guaranteed_16B_aligned=false, copy_async=true, use_read_datamover=true,
            //            max_transfer_size=SUBTILE_LINE_BYTES>: at runtime the helper checks
            // (dst & 0xF) == (src & 0xF) and uses the NOC read datamover when it matches,
            // otherwise falls back to memmove. SUBTILE_LINE_BYTES is 32 (bf16) or 64 (fp32) —
            // both multiples of 16. write_addr stride is tile_size (multiple of 16). scratch is
            // DRAM-aligned at base, with skew < DRAM_ALIGN_BYTES added per phase; for 16-byte
            // multiples of phase offsets the parities line up and we hit the datamover fast
            // path. The trailing barrier below makes the (async) datamover reads complete before
            // the scratch buffer is reused by the next stage_phase invocation.
            uint32_t scratch_read_offset = scratch_base + skew;
            uint32_t write_addr = write_addr_base + phase_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                tt::data_movement::common::tt_memmove<
                    /*guaranteed_16B_aligned=*/false,
                    /*copy_async=*/true,
                    /*use_read_datamover=*/true,
                    /*max_transfer_size=*/SUBTILE_LINE_BYTES>(noc, write_addr, scratch_read_offset, SUBTILE_LINE_BYTES);
                scratch_read_offset += DRAM_ALIGN_BYTES;
                write_addr += tile_size;
            }
            // Sync the async tt_memmove reads before the scratch buffer is reused or callers expect
            // the output buffer region to be populated. The std::memcpy version was synchronous so no
            // barrier was needed; the NOC datamover path is async and writes to scratch must
            // complete before the next stage_phase issues stage-1 DRAM reads into the same buffer.
            noc.async_read_barrier();
        };

        auto handle_one_head = [&](uint32_t write_addr_base) {
            const uint32_t starting_tile_id = qkv_tile_id;
            if constexpr (read_phase_1) {
                stage_phase(write_addr_base, starting_tile_id, /*phase_offset=*/0);
            }
            if constexpr (read_phase_2) {
                stage_phase(write_addr_base, starting_tile_id, /*phase_offset=*/PHASE_OFFSET_BYTES);
            }
            qkv_tile_id += head_size_num_tiles;
        };

        // Q
        for (uint32_t q = 0; q < num_q_heads; ++q) {
            uint32_t tile_row_index = q / tt::constants::TILE_HEIGHT;
            uint32_t row_in_tile = q % tt::constants::TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < tt::constants::FACE_HEIGHT
                                          ? row_in_tile * SUBTILE_LINE_BYTES
                                          : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES +
                                                HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t q_write_addr = q_out.get_bank_base_address() + wptr_offset;
            handle_one_head(q_write_addr);
        }

        // K
        for (uint32_t k = 0; k < num_kv_heads; ++k) {
            uint32_t tile_row_index = k / tt::constants::TILE_HEIGHT;
            uint32_t row_in_tile = k % tt::constants::TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < tt::constants::FACE_HEIGHT
                                          ? row_in_tile * SUBTILE_LINE_BYTES
                                          : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES +
                                                HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t k_write_addr = k_out.get_bank_base_address() + wptr_offset;
            handle_one_head(k_write_addr);
        }

        // V
        for (uint32_t v = 0; v < num_kv_heads; ++v) {
            uint32_t tile_row_index = v / tt::constants::TILE_HEIGHT;
            uint32_t row_in_tile = v % tt::constants::TILE_HEIGHT;
            uint32_t offset_in_tile = row_in_tile < tt::constants::FACE_HEIGHT
                                          ? row_in_tile * SUBTILE_LINE_BYTES
                                          : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES +
                                                HALF_TILE_ELEMENTS * ELEMENT_SIZE;
            uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
            uint32_t v_write_addr = v_out.get_bank_base_address() + wptr_offset;
            handle_one_head(v_write_addr);
        }

        noc.async_read_barrier();
        return;
    }
#endif

    // Direct-read fast path: source/destination NOC alignment is naturally satisfied for this
    // (arch, dtype, buffer-type) combination. Unchanged from the original kernel.

    // Q
    uint32_t q_write_addr = 0;
    for (uint32_t q = 0; q < num_q_heads; ++q) {
        uint32_t tile_row_index = q / tt::constants::TILE_HEIGHT;
        uint32_t row_in_tile = q % tt::constants::TILE_HEIGHT;
        uint32_t offset_in_tile =
            row_in_tile < tt::constants::FACE_HEIGHT
                ? row_in_tile * SUBTILE_LINE_BYTES
                : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE;
        uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
        uint32_t q_write_addr = q_out.get_bank_base_address() + wptr_offset;

        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(q_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch},
                    {});
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(q_write_addr + 256 * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch + 256 * ELEMENT_SIZE},
                    {});
            }

            qkv_tile_id += 1;
            q_write_addr += tile_size;
        }
        noc.async_read_barrier();
    }

    // K
    uint32_t k_write_addr = 0;
    for (uint32_t k = 0; k < num_kv_heads; ++k) {
        uint32_t tile_row_index = k / tt::constants::TILE_HEIGHT;
        uint32_t row_in_tile = k % tt::constants::TILE_HEIGHT;
        uint32_t offset_in_tile =
            row_in_tile < tt::constants::FACE_HEIGHT
                ? row_in_tile * SUBTILE_LINE_BYTES
                : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE;
        uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
        uint32_t k_write_addr = k_out.get_bank_base_address() + wptr_offset;

        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(k_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch},
                    {});
            }
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(k_write_addr + 256 * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch + 256 * ELEMENT_SIZE},
                    {});
            }

            qkv_tile_id += 1;
            k_write_addr += tile_size;
        }
        noc.async_read_barrier();
    }

    // V
    uint32_t v_write_addr = 0;
    for (uint32_t v = 0; v < num_kv_heads; ++v) {
        uint32_t tile_row_index = v / tt::constants::TILE_HEIGHT;
        uint32_t row_in_tile = v % tt::constants::TILE_HEIGHT;
        uint32_t offset_in_tile =
            row_in_tile < tt::constants::FACE_HEIGHT
                ? row_in_tile * SUBTILE_LINE_BYTES
                : (row_in_tile - tt::constants::FACE_HEIGHT) * SUBTILE_LINE_BYTES + HALF_TILE_ELEMENTS * ELEMENT_SIZE;
        uint32_t wptr_offset = tile_row_index * head_size + offset_in_tile;
        uint32_t v_write_addr = v_out.get_bank_base_address() + wptr_offset;

        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(v_write_addr),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch},
                    {});
            }
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc.async_read(
                    qkv_reader,
                    CoreLocalMem<uint32_t>(v_write_addr + 256 * ELEMENT_SIZE),
                    SUBTILE_LINE_BYTES,
                    {.page_id = qkv_tile_id, .offset_bytes = in_tile_offset_by_batch + 256 * ELEMENT_SIZE},
                    {});
            }

            qkv_tile_id += 1;
            v_write_addr += tile_size;
        }
        noc.async_read_barrier();
    }

    noc.async_read_barrier();
}
