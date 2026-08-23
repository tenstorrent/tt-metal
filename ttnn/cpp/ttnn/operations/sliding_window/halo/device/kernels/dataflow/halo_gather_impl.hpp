// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

namespace halo {

constexpr uint16_t TILE_SIZE = 32;

template <uint32_t N, uint16_t PaddingValue>
FORCE_INLINE void fill_with_val(uint32_t begin_addr) {
    static_assert(N % 2 == 0, "The aligned padding scratch must contain complete uint32_t words");
    uint32_t* values = reinterpret_cast<uint32_t*>(begin_addr);
    constexpr uint32_t packed_padding_value = PaddingValue | (static_cast<uint32_t>(PaddingValue) << 16);
    for (uint32_t i = 0; i < N / 2; ++i) {
        values[i] = packed_padding_value;
    }
    // The NOC consumes this private L1 scratch through its address, which is not a C++ memory read.
    // Publish the optimized stores before programming the NOC without adding a hardware fence.
    asm volatile("" ::: "memory");
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_small_sticks(Noc noc, uint32_t padding_l1_addr, uint32_t dst_addr, uint16_t nsticks) {
    static_assert(MaxChunkSize >= StickNBytes, "Small-stick padding requires the chunk to hold at least one stick");
    constexpr uint32_t sticks_per_batch = MaxChunkSize / StickNBytes;
    constexpr uint32_t batch_size_bytes = sticks_per_batch * StickNBytes;
    static_assert(batch_size_bytes <= NOC_MAX_BURST_SIZE, "Padding batch must fit in one NOC packet");
    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];
    if constexpr (sticks_per_batch > 1) {
        const uint16_t num_full_batches = nsticks / sticks_per_batch;
        const uint16_t remaining_sticks = nsticks % sticks_per_batch;
        uint32_t current_dst = dst_addr;
        noc.set_async_read_state<NocOptions::DEFAULT, batch_size_bytes>(
            UnicastEndpoint{}, batch_size_bytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr});
        for (uint16_t batch = 0; batch < num_full_batches; ++batch) {
            noc.async_read_with_state<NocOptions::DEFAULT, batch_size_bytes>(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(current_dst),
                batch_size_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            current_dst += batch_size_bytes;
        }
        if (remaining_sticks > 0) {
            noc.set_async_read_state<NocOptions::DEFAULT, StickNBytes>(
                UnicastEndpoint{}, StickNBytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr});
            for (uint16_t k = 0; k < remaining_sticks; ++k) {
                noc.async_read_with_state<NocOptions::DEFAULT, StickNBytes>(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(current_dst),
                    StickNBytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                    {});
                current_dst += StickNBytes;
            }
        }
    } else {
        noc.set_async_read_state<NocOptions::DEFAULT, StickNBytes>(
            UnicastEndpoint{}, StickNBytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr});
        uint32_t current_dst = dst_addr;
        for (uint16_t k = 0; k < nsticks; ++k) {
            noc.async_read_with_state<NocOptions::DEFAULT, StickNBytes>(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(current_dst),
                StickNBytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            current_dst += StickNBytes;
        }
    }
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_large_sticks(Noc noc, uint32_t padding_l1_addr, uint32_t dst_addr, uint16_t nsticks) {
    constexpr uint32_t num_full_chunks = StickNBytes / MaxChunkSize;
    constexpr uint32_t remainder_bytes = StickNBytes % MaxChunkSize;
    constexpr uint32_t remainder_offset = num_full_chunks * MaxChunkSize;
    static_assert(MaxChunkSize <= NOC_MAX_BURST_SIZE, "Padding chunk must fit in one NOC packet");
    static_assert(remainder_bytes <= NOC_MAX_BURST_SIZE, "Padding remainder must fit in one NOC packet");
    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];
    if constexpr (num_full_chunks > 0) {
        noc.set_async_read_state<NocOptions::DEFAULT, MaxChunkSize>(
            UnicastEndpoint{}, MaxChunkSize, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr});
        uint32_t stick_base_addr = dst_addr;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            uint32_t chunk_addr = stick_base_addr;
            for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
                noc.async_read_with_state<NocOptions::DEFAULT, MaxChunkSize>(
                    UnicastEndpoint{},
                    CoreLocalMem<uint32_t>(chunk_addr),
                    MaxChunkSize,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                    {});
                chunk_addr += MaxChunkSize;
            }
            stick_base_addr += StickNBytes;
        }
    }
    if constexpr (remainder_bytes > 0) {
        noc.set_async_read_state<NocOptions::DEFAULT, remainder_bytes>(
            UnicastEndpoint{}, remainder_bytes, {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr});
        uint32_t remainder_base_addr = dst_addr + remainder_offset;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            noc.async_read_with_state<NocOptions::DEFAULT, remainder_bytes>(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(remainder_base_addr),
                remainder_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            remainder_base_addr += StickNBytes;
        }
    }
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding(
    Noc noc, uint32_t padding_config_l1_addr, uint32_t dst_base_addr, uint32_t padding_l1_addr) {
    volatile tt_l1_ptr uint16_t* config = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);
    uint16_t nsticks = 1;
    for (uint16_t j = 0; nsticks != 0; j += 2) {
        const uint16_t dst_local_idx = config[j];
        nsticks = config[j + 1];
        const uint32_t dst_addr = dst_base_addr + static_cast<uint32_t>(dst_local_idx) * StickNBytes;
        if constexpr (StickNBytes <= MaxChunkSize) {
            copy_padding_small_sticks<StickNBytes, MaxChunkSize>(noc, padding_l1_addr, dst_addr, nsticks);
        } else {
            copy_padding_large_sticks<StickNBytes, MaxChunkSize>(noc, padding_l1_addr, dst_addr, nsticks);
        }
    }
}

template <bool IsBlockSharded, bool IsWidthSharded, bool IsColumnMajor>
FORCE_INLINE void resolve_destination_coords(
    uint16_t destination_noc_x,
    uint16_t destination_noc_y,
    uint16_t my_noc_x,
    uint16_t my_noc_y,
    uint16_t& out_noc_x,
    uint16_t& out_noc_y) {
    static_assert(!(IsBlockSharded && IsWidthSharded), "A tensor cannot be both block- and width-sharded");
    out_noc_x = ((IsBlockSharded && !IsColumnMajor) || IsWidthSharded) ? my_noc_x : destination_noc_x;
    out_noc_y = ((IsBlockSharded && IsColumnMajor) || IsWidthSharded) ? my_noc_y : destination_noc_y;
}

template <uint32_t StickSizeBytes, bool EnableBlocking, uint32_t BlockHeightSticks, typename Src>
FORCE_INLINE void write_stick_async(
    Noc noc,
    const Src& input,
    uint32_t out_base_l1_addr,
    uint16_t dst_noc_x,
    uint16_t dst_noc_y,
    uint16_t src_offset_id,
    uint16_t dst_offset_id,
    uint16_t transfer_size) {
    const uint32_t src_offset =
        EnableBlocking ? (src_offset_id % BlockHeightSticks) * StickSizeBytes : src_offset_id * StickSizeBytes;
    const uint32_t dst_offset = dst_offset_id * StickSizeBytes;
    noc.async_write(
        input,
        UnicastEndpoint{},
        transfer_size * StickSizeBytes,
        {.offset_bytes = src_offset},
        {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = out_base_l1_addr + dst_offset});
}

template <
    uint32_t PadVal,
    uint32_t InputNPages,
    uint32_t SkipUntilize,
    uint32_t AlignedStickNBytes,
    uint32_t IsBlockSharded,
    uint32_t IsColumnMajor,
    uint32_t IsWidthSharded,
    uint32_t BlockSizeHeight,
    uint32_t BlockSizeWidthTiles,
    uint32_t BlockStartOffset,
    uint32_t BlockStride,
    uint32_t ConfigTensorInDram,
    uint32_t EnablePadding,
    uint32_t UsePadScratch>
FORCE_INLINE void gather(uint32_t config_read_index) {
    static_assert(BlockStride >= 1, "Block stride must be at least one");
    constexpr bool SrcProducer = BlockStartOffset == 0;
    constexpr bool enable_blocking = !SkipUntilize;
    constexpr uint32_t total_tiles_in_single_block = (BlockSizeHeight / TILE_SIZE) * BlockSizeWidthTiles;

    Noc noc;
    DataflowBuffer output(dfb::out);
    const uint32_t out_base_l1_addr = output.get_write_ptr();
    uint32_t gather_config_l1_addr;
    uint32_t padding_config_l1_addr = 0;
    if constexpr (ConfigTensorInDram) {
        TensorAccessor gather_config(tensor::gather_config);
        Scratchpad<uint32_t> gather_scratch(scratch::gather_config);
        gather_config_l1_addr = gather_scratch.get_base_address();
        noc.async_read(
            gather_config,
            CoreLocalMem<uint32_t>(gather_config_l1_addr),
            gather_scratch.size_in_bytes(),
            {.page_id = config_read_index},
            {});
        if constexpr (EnablePadding) {
            TensorAccessor padding_config(tensor::padding_config);
            Scratchpad<uint32_t> padding_scratch(scratch::padding_config);
            padding_config_l1_addr = padding_scratch.get_base_address();
            noc.async_read(
                padding_config,
                CoreLocalMem<uint32_t>(padding_config_l1_addr),
                padding_scratch.size_in_bytes(),
                {.page_id = config_read_index},
                {});
        }
        noc.async_read_barrier();
    } else {
        TensorAccessor gather_config(tensor::gather_config);
        gather_config_l1_addr = gather_config.get_bank_base_address();
        if constexpr (EnablePadding) {
            TensorAccessor padding_config(tensor::padding_config);
            padding_config_l1_addr = padding_config.get_bank_base_address();
        }
    }

    DataflowBuffer src(dfb::src);
    if constexpr (SrcProducer) {
        // The input shard is resident and shared by both split readers. Reader 0 alone owns the
        // reserve/push bookkeeping so the shared received/acked counters stay balanced.
        src.reserve_back(static_cast<uint16_t>(InputNPages));
        src.push_back(static_cast<uint16_t>(InputNPages));
    }

    if constexpr (EnablePadding) {
        if constexpr (PadVal == 0) {
            copy_padding<AlignedStickNBytes, MEM_ZEROS_SIZE>(
                noc, padding_config_l1_addr, out_base_l1_addr, MEM_ZEROS_BASE);
        } else {
            static_assert(UsePadScratch, "Nonzero padding requires the pad scratch binding");
            Scratchpad<uint32_t> pad(scratch::pad);
            constexpr uint32_t num_elements = AlignedStickNBytes / sizeof(uint16_t);
            fill_with_val<num_elements, static_cast<uint16_t>(PadVal)>(pad.get_base_address());
            copy_padding<AlignedStickNBytes, AlignedStickNBytes>(
                noc, padding_config_l1_addr, out_base_l1_addr, pad.get_base_address());
        }
    }

    if constexpr (SrcProducer && SkipUntilize) {
        // The other reader consumes the already-resident shard directly; making it wait/pop would race
        // this shared read pointer and retire pages while reader 0 may still be using them. Keep this
        // wait immediately before gather, after padding, to preserve the legacy stream-register and NOC
        // issue ordering.
        src.wait_front(static_cast<uint16_t>(InputNPages));
    }

    const tt_l1_ptr uint16_t* config = reinterpret_cast<const tt_l1_ptr uint16_t*>(gather_config_l1_addr);
    uint16_t config_index = 0;
    uint16_t segments_remaining = config[config_index++];
    if (segments_remaining != 0) {
        uint16_t block_boundary_offset = BlockSizeHeight + BlockSizeHeight * BlockStartOffset;
        DataflowBuffer input(SkipUntilize ? dfb::src : dfb::untilize_out);
        if constexpr (!SkipUntilize) {
            input.wait_front(total_tiles_in_single_block);
        }
        // Row-major input is already resident and ready; only reader 0 owns its FIFO bookkeeping.
        // This read pointer cannot advance on the row-major path, so materialize the source once.
        // Keeping it stable also preserves the legacy readers' relative NOC issue timing when both
        // RISCs gather into the same output shard.
        const CoreLocalMem<uint32_t> input_source(input.get_read_ptr());
        const uint16_t my_noc_x = my_x[noc.get_noc_id()];
        const uint16_t my_noc_y = my_y[noc.get_noc_id()];
        while (segments_remaining != 0) {
            const uint16_t destination_noc_x = config[config_index++];
            const uint16_t destination_noc_y = config[config_index++];
            uint16_t transfers_remaining = config[config_index++];
            uint16_t dst_noc_x;
            uint16_t dst_noc_y;
            resolve_destination_coords<IsBlockSharded != 0, IsWidthSharded != 0, IsColumnMajor != 0>(
                destination_noc_x, destination_noc_y, my_noc_x, my_noc_y, dst_noc_x, dst_noc_y);
            while (transfers_remaining != 0) {
                const uint16_t src_offset = config[config_index++];
                const uint16_t dst_offset = config[config_index++];
                const uint16_t transfer_size = config[config_index++];
                if constexpr (!SkipUntilize) {
                    // Transfers are globally ordered by ascending block ID, so retire blocks until the
                    // current transfer falls within this split reader's active block.
                    while (src_offset >= block_boundary_offset) {
                        noc.async_write_barrier();
                        input.pop_front(total_tiles_in_single_block);
                        input.wait_front(total_tiles_in_single_block);
                        block_boundary_offset += BlockSizeHeight * BlockStride;
                    }
                }
                if constexpr (SkipUntilize) {
                    write_stick_async<AlignedStickNBytes, enable_blocking, BlockSizeHeight>(
                        noc,
                        input_source,
                        out_base_l1_addr,
                        dst_noc_x,
                        dst_noc_y,
                        src_offset,
                        dst_offset,
                        transfer_size);
                } else {
                    write_stick_async<AlignedStickNBytes, enable_blocking, BlockSizeHeight>(
                        noc, input, out_base_l1_addr, dst_noc_x, dst_noc_y, src_offset, dst_offset, transfer_size);
                }
                --transfers_remaining;
            }
            --segments_remaining;
        }
        if constexpr (!SkipUntilize) {
            input.pop_front(total_tiles_in_single_block);
        }
    }

    noc.async_read_barrier();
    noc.async_write_barrier();
    if constexpr (SrcProducer && SkipUntilize) {
        // Balance reader 0's reserve/push/wait only after both its input reads and output writes finish.
        src.pop_front(static_cast<uint16_t>(InputNPages));
    }
}

}  // namespace halo
