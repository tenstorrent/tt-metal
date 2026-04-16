// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "api/debug/dprint.h"
#include "api/debug/dprint_pages.h"
#endif

constexpr uint16_t TILE_SIZE = 32;

template <uint32_t N, uint16_t PaddingValue>
FORCE_INLINE void fill_with_val(uint32_t begin_addr) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < N; ++i) {
        ptr[i] = PaddingValue;
    }
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_small_sticks(
    experimental::Noc noc, uint32_t padding_l1_addr, uint32_t dst_addr, uint16_t nsticks) {
    static_assert(MaxChunkSize >= StickNBytes, "This function assumes max chunk size > stick size");

    constexpr uint32_t sticks_per_batch = MaxChunkSize / StickNBytes;
    constexpr uint32_t batch_size_bytes = sticks_per_batch * StickNBytes;
    static_assert(batch_size_bytes <= NOC_MAX_BURST_SIZE, "batch_size_bytes must be single-packet");

    if constexpr (sticks_per_batch > 1) {
        const uint16_t num_full_batches = nsticks / sticks_per_batch;
        const uint16_t remaining_sticks = nsticks % sticks_per_batch;

        uint32_t current_dst = dst_addr;

        experimental::set_read_state<batch_size_bytes>(noc, padding_l1_addr);
        for (uint16_t batch = 0; batch < num_full_batches; ++batch) {
            experimental::read_with_state(noc, current_dst, padding_l1_addr);
            current_dst += batch_size_bytes;
        }

        if (remaining_sticks > 0) {
            experimental::set_read_state<StickNBytes>(noc, padding_l1_addr);
            for (uint16_t k = 0; k < remaining_sticks; ++k) {
                experimental::read_with_state(noc, current_dst, padding_l1_addr);
                current_dst += StickNBytes;
            }
        }
    } else {
        experimental::set_read_state<StickNBytes>(noc, padding_l1_addr);
        uint32_t current_dst = dst_addr;
        for (uint16_t k = 0; k < nsticks; ++k) {
            experimental::read_with_state(noc, current_dst, padding_l1_addr);
            current_dst += StickNBytes;
        }
    }
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_large_sticks(
    experimental::Noc noc, uint32_t padding_l1_addr, uint32_t dst_addr, uint16_t nsticks) {
    constexpr uint32_t num_full_chunks = StickNBytes / MaxChunkSize;
    constexpr uint32_t remainder_bytes = StickNBytes % MaxChunkSize;
    constexpr uint32_t remainder_offset = num_full_chunks * MaxChunkSize;
    static_assert(MaxChunkSize <= NOC_MAX_BURST_SIZE, "MaxChunkSize must be single-packet");
    static_assert(remainder_bytes <= NOC_MAX_BURST_SIZE, "remainder must be single-packet");

    // Copy all full chunks for all sticks
    if constexpr (num_full_chunks > 0) {
        experimental::set_read_state<MaxChunkSize>(noc, padding_l1_addr);

        uint32_t stick_base_addr = dst_addr;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            uint32_t chunk_addr = stick_base_addr;
            for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
                experimental::read_with_state(noc, chunk_addr, padding_l1_addr);
                chunk_addr += MaxChunkSize;
            }
            stick_base_addr += StickNBytes;
        }
    }

    // Copy all remainder chunks for all sticks
    if constexpr (remainder_bytes > 0) {
        experimental::set_read_state<remainder_bytes>(noc, padding_l1_addr);

        uint32_t remainder_base_addr = dst_addr + remainder_offset;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            experimental::read_with_state(noc, remainder_base_addr, padding_l1_addr);
            remainder_base_addr += StickNBytes;
        }
    }
}

template <uint32_t PaddingConfigCBId, uint32_t OutCBId, uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding(
    experimental::Noc noc, experimental::CB padding_config_cb, experimental::CB out_cb, uint32_t padding_l1_addr) {
    const uint32_t padding_config_l1_addr = padding_config_cb.get_read_ptr();
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);

    const uint32_t dst_base_addr = out_cb.get_write_ptr();

    uint16_t nsticks = 1;
    constexpr uint32_t stick_stride = StickNBytes;
    for (uint16_t j = 0; nsticks != 0; j += 2) {
        uint16_t dst_local_idx = config_data[j + 0];
        nsticks = config_data[j + 1];

        const uint32_t dst_addr = dst_base_addr + (static_cast<uint32_t>(dst_local_idx) * stick_stride);
        if constexpr (StickNBytes <= MaxChunkSize) {
            copy_padding_small_sticks<StickNBytes, MaxChunkSize>(noc, padding_l1_addr, dst_addr, nsticks);
        } else {
            copy_padding_large_sticks<StickNBytes, MaxChunkSize>(noc, padding_l1_addr, dst_addr, nsticks);
        }
    }
}

template <bool IsBlockSharded, bool IsWidthSharded, bool IsColumnMajor>
static inline void resolve_destination_coords(
    uint16_t destination_noc_x,
    uint16_t destination_noc_y,
    uint16_t my_noc_x,
    uint16_t my_noc_y,
    uint16_t& out_noc_x,
    uint16_t& out_noc_y) {
    static_assert(!(IsBlockSharded && IsWidthSharded), "Cannot be block and width sharding");
    if constexpr ((IsBlockSharded && !IsColumnMajor) || IsWidthSharded) {
        out_noc_x = my_noc_x;
    } else {
        out_noc_x = destination_noc_x;
    }
    if constexpr ((IsBlockSharded && IsColumnMajor) || IsWidthSharded) {
        out_noc_y = my_noc_y;
    } else {
        out_noc_y = destination_noc_y;
    }
}

template <uint32_t StickSizeBytes, bool EnableBlocking, uint32_t BlockHeightSticks>
static inline void write_stick_async(
    experimental::Noc noc,
    uint32_t in_base_l1_addr,
    uint32_t out_base_l1_addr,
    uint16_t dst_noc_x,
    uint16_t dst_noc_y,
    uint16_t src_offset_id,
    uint16_t dst_offset_id,
    uint16_t transfer_size) {
    uint32_t src_offset;
    if constexpr (EnableBlocking) {
        src_offset = (src_offset_id % BlockHeightSticks) * StickSizeBytes;
    } else {
        src_offset = src_offset_id * StickSizeBytes;
    }
    const uint32_t dst_offset = dst_offset_id * StickSizeBytes;
    const uint32_t size = transfer_size * StickSizeBytes;
    const uint32_t src_addr = in_base_l1_addr + src_offset;
    const uint32_t dst_addr = out_base_l1_addr + dst_offset;

    noc.async_write(
        experimental::CoreLocalMem<uint32_t>(src_addr),
        experimental::UnicastEndpoint{},
        size,
        {},
        {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = dst_addr});
}

template <
    uint32_t InputCBIndex,
    uint32_t OutputCBIndex,
    uint32_t StickSizeBytes,
    uint32_t BlockSizeHeight,
    uint32_t BlockSizeWidthTiles,
    uint32_t BlockStride,
    uint32_t BlockStartOffset,
    bool EnableBlocking,
    bool IsBlockSharded,
    bool IsWidthSharded,
    bool IsColumnMajor>
static inline void run_halo_gather(
    experimental::Noc noc,
    experimental::CB in_cb,
    experimental::CB out_cb,
    const tt_l1_ptr uint16_t* config,
    uint16_t my_noc_x,
    uint16_t my_noc_y) {
    static_assert(BlockStride >= 1, "Blocks stride must be at least 1");

    constexpr uint32_t block_size_height_tiles = BlockSizeHeight / TILE_SIZE;
    constexpr uint32_t total_tiles_in_single_block = block_size_height_tiles * BlockSizeWidthTiles;

    uint16_t current_config_index = 0;
    uint16_t number_of_segments_remaining = config[current_config_index++];

    if (number_of_segments_remaining == 0) {
        return;
    }

    uint32_t in_base_l1_addr = in_cb.get_read_ptr();
    const uint32_t out_base_l1_addr = out_cb.get_write_ptr();

    // Assume input is already ready when !EnableBlocking (like when using RM)
    if constexpr (EnableBlocking) {
        in_cb.wait_front(total_tiles_in_single_block);
    }

    uint16_t block_id = BlockStartOffset;
    uint16_t block_boundary_offset = BlockSizeHeight + (BlockSizeHeight * BlockStartOffset);
    while (number_of_segments_remaining) {
        //  Read header for to get destination for this route
        const uint16_t destination_noc_x = config[current_config_index++];
        const uint16_t destination_noc_y = config[current_config_index++];
        uint16_t transfers_remaining = config[current_config_index++];

        uint16_t dst_noc_x, dst_noc_y;
        resolve_destination_coords<IsBlockSharded, IsWidthSharded, IsColumnMajor>(
            destination_noc_x, destination_noc_y, my_noc_x, my_noc_y, dst_noc_x, dst_noc_y);

        // Perform all transfers in this route
        while (transfers_remaining > 0) {
            const uint16_t src_offset = config[current_config_index++];
            const uint16_t dst_offset = config[current_config_index++];
            const uint16_t transfer_size = config[current_config_index++];
            if constexpr (EnableBlocking) {
                // Pop blocks until we have the right one - this works because transfers are globally ordered by
                // ascending block IDs
                while (src_offset >= block_boundary_offset) {
                    noc.async_write_barrier();
                    in_cb.pop_front(total_tiles_in_single_block);
                    in_cb.wait_front(total_tiles_in_single_block);
                    block_boundary_offset +=
                        BlockSizeHeight *
                        BlockStride;  // When block stride > 1 we are expecting the input CB to skip
                                      // BlockStride number of blocks (like when splitting work across cores)
                    block_id += BlockStride;
                    in_base_l1_addr = in_cb.get_read_ptr();  // Ensure base address is at front of input CB
                }
            }
            write_stick_async<StickSizeBytes, EnableBlocking, BlockSizeHeight>(
                noc, in_base_l1_addr, out_base_l1_addr, dst_noc_x, dst_noc_y, src_offset, dst_offset, transfer_size);
            transfers_remaining--;
        }
        number_of_segments_remaining--;
    }

    if constexpr (EnableBlocking) {
        in_cb.pop_front(total_tiles_in_single_block);
    }
}

void kernel_main() {
    constexpr uint32_t padding_config_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t gather_config_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(6);
    constexpr uint32_t in_nsticks = get_compile_time_arg_val(7);
    constexpr uint32_t aligned_stick_nbytes = get_compile_time_arg_val(8);
    constexpr bool is_block_sharded = get_compile_time_arg_val(9) == 1;
    constexpr bool remote_read = get_compile_time_arg_val(10) == 1;
    constexpr bool is_col_major = get_compile_time_arg_val(11) == 1;
    constexpr bool is_width_sharded = get_compile_time_arg_val(12) == 1;
    constexpr bool skip_untilize = get_compile_time_arg_val(13) == 1;
    constexpr uint32_t block_size_height = get_compile_time_arg_val(14);
    constexpr uint32_t block_size_width_tiles = get_compile_time_arg_val(15);
    constexpr uint32_t block_start_offset = get_compile_time_arg_val(16);
    constexpr uint32_t block_stride = get_compile_time_arg_val(17);

    static_assert(!remote_read, "Remote read is not supported in this kernel");

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr bool enable_blocking = !skip_untilize;

    experimental::Noc noc;
    experimental::CB padding_config_cb(padding_config_cb_id);
    experimental::CB gather_config_cb(gather_config_cb_id);
    experimental::CB src_cb(src_cb_id);
    experimental::CB in_cb(in_cb_id);
    experimental::CB out_cb(out_cb_id);
    experimental::CB pad_cb(pad_cb_id);

#ifdef CONFIG_TENSOR_IN_DRAM
    constexpr uint32_t padding_config_dram_addr = get_compile_time_arg_val(18);
    constexpr uint32_t padding_config_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t gather_config_dram_addr = get_compile_time_arg_val(20);
    constexpr uint32_t gather_config_page_size = get_compile_time_arg_val(21);

    constexpr auto padding_config_tensor_args = TensorAccessorArgs<22>();
    constexpr auto gather_config_tensor_args =
        TensorAccessorArgs<padding_config_tensor_args.next_compile_time_args_offset()>();

    const auto padding_config_accessor = TensorAccessor(padding_config_tensor_args, padding_config_dram_addr);
    const auto gather_config_accessor = TensorAccessor(gather_config_tensor_args, gather_config_dram_addr);

    uint32_t config_read_index = get_arg_val<uint32_t>(0);

    noc.async_read(
        padding_config_accessor, padding_config_cb, padding_config_page_size, {.page_id = config_read_index}, {});
    noc.async_read(
        gather_config_accessor, gather_config_cb, gather_config_page_size, {.page_id = config_read_index}, {});
    noc.async_read_barrier();
#endif

    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];

    // Only one of the cores should push the input
    if constexpr (block_start_offset == 0) {
        src_cb.reserve_back(in_nsticks);
        src_cb.push_back(in_nsticks);
    }

    if constexpr (padding_config_cb_id != 0) {
        if constexpr (pad_val_u32 == 0) {
            // Use MEM_ZEROS_BASE if we are zero padded
            constexpr uint32_t padding_region_size = MEM_ZEROS_SIZE;
            copy_padding<padding_config_cb_id, out_cb_id, aligned_stick_nbytes, padding_region_size>(
                noc, padding_config_cb, out_cb, MEM_ZEROS_BASE);
        } else {
            constexpr uint16_t pad_val = static_cast<uint16_t>(pad_val_u32);
            constexpr uint32_t num_elements_to_fill = aligned_stick_nbytes / elem_nbytes;
            fill_with_val<num_elements_to_fill, pad_val>(pad_cb.get_write_ptr());

            // MaxChunkSize must be in bytes and >= StickNBytes to avoid misaligned NOC transactions
            constexpr uint32_t padding_region_size = aligned_stick_nbytes;
            copy_padding<padding_config_cb_id, out_cb_id, aligned_stick_nbytes, padding_region_size>(
                noc, padding_config_cb, out_cb, pad_cb.get_read_ptr());
        }
    }

    if constexpr (skip_untilize) {
        src_cb.wait_front(in_nsticks);
    }

    const uint32_t config_data_l1_addr = gather_config_cb.get_read_ptr();
    const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
    run_halo_gather<
        in_cb_id,
        out_cb_id,
        aligned_stick_nbytes,
        block_size_height,
        block_size_width_tiles,
        block_stride,
        block_start_offset,
        enable_blocking,
        is_block_sharded,
        is_width_sharded,
        is_col_major>(noc, in_cb, out_cb, config_data, my_noc_x, my_noc_y);

    noc.async_read_barrier();
    noc.async_write_barrier();
}
