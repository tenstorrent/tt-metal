// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"

#define ENABLE_DEBUG 0

#if ENABLE_DEBUG
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
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
FORCE_INLINE void copy_padding_small_sticks(uint64_t padding_l1_addr, uint64_t dst_addr, uint16_t nsticks) {
    static_assert(MaxChunkSize >= StickNBytes, "This function assumes max chunk size > stick size");

    constexpr uint32_t sticks_per_batch = MaxChunkSize / StickNBytes;
    constexpr uint32_t batch_size_bytes = sticks_per_batch * StickNBytes;

    if constexpr (sticks_per_batch > 1) {
        const uint16_t num_full_batches = nsticks / sticks_per_batch;
        const uint16_t remaining_sticks = nsticks % sticks_per_batch;

        uint64_t current_dst = dst_addr;
        for (uint16_t batch = 0; batch < num_full_batches; ++batch) {
            noc_async_read(padding_l1_addr, current_dst, batch_size_bytes);
            current_dst += batch_size_bytes;
        }

        for (uint16_t k = 0; k < remaining_sticks; ++k) {
            noc_async_read(padding_l1_addr, current_dst, StickNBytes);
            current_dst += StickNBytes;
        }
    } else {
        noc_async_read_one_packet_set_state(padding_l1_addr, StickNBytes);
        uint64_t current_dst = dst_addr;
        for (uint16_t k = 0; k < nsticks; ++k) {
            noc_async_read_one_packet_with_state(padding_l1_addr, current_dst);
            current_dst += StickNBytes;
        }
    }
}

template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_large_sticks(uint64_t padding_l1_addr, uint64_t dst_addr, uint16_t nsticks) {
    constexpr uint32_t num_full_chunks = StickNBytes / MaxChunkSize;
    constexpr uint32_t remainder_bytes = StickNBytes % MaxChunkSize;
    constexpr uint32_t remainder_offset = num_full_chunks * MaxChunkSize;

    // Copy all full chunks for all sticks
    if constexpr (num_full_chunks > 0) {
        noc_async_read_one_packet_set_state(padding_l1_addr, MaxChunkSize);

        uint64_t stick_base_addr = dst_addr;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            uint64_t chunk_addr = stick_base_addr;
            for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
                noc_async_read_one_packet_with_state(padding_l1_addr, chunk_addr);
                chunk_addr += MaxChunkSize;
            }
            stick_base_addr += StickNBytes;
        }
    }

    // Copy all remainder chunks for all sticks
    if constexpr (remainder_bytes > 0) {
        noc_async_read_one_packet_set_state(padding_l1_addr, remainder_bytes);

        uint64_t remainder_base_addr = dst_addr + remainder_offset;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            noc_async_read_one_packet_with_state(padding_l1_addr, remainder_base_addr);
            remainder_base_addr += StickNBytes;
        }
    }
}

template <uint32_t PaddingConfigCBId, uint32_t OutCBId, uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding(uint64_t padding_l1_addr) {
    const uint32_t padding_config_l1_addr = get_read_ptr(PaddingConfigCBId);
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);

    const uint64_t dst_base_addr = get_write_ptr(OutCBId);

    uint16_t nsticks = 1;
    constexpr uint32_t stick_stride = StickNBytes;
    for (uint16_t j = 0; nsticks != 0; j += 2) {
        uint16_t dst_local_idx = config_data[j + 0];
        nsticks = config_data[j + 1];

        const uint64_t dst_addr = dst_base_addr + (static_cast<uint64_t>(dst_local_idx) * stick_stride);
        if constexpr (StickNBytes <= MaxChunkSize) {
            copy_padding_small_sticks<StickNBytes, MaxChunkSize>(padding_l1_addr, dst_addr, nsticks);
        } else {
            copy_padding_large_sticks<StickNBytes, MaxChunkSize>(padding_l1_addr, dst_addr, nsticks);
        }
    }
}

template <bool IsBlockSharded, bool IsWidthSharded, bool IsColumnMajor>
static inline uint64_t get_remote_core_l1_noc_addr(
    uint16_t destination_noc_x,
    uint16_t destination_noc_y,
    uint16_t my_noc_x,
    uint16_t my_noc_y,
    uint32_t out_base_l1_addr) {
    static_assert(!(IsBlockSharded && IsWidthSharded), "Cannot be block and width sharding");
    uint16_t noc_x;
    if constexpr ((IsBlockSharded && !IsColumnMajor) || IsWidthSharded) {
        noc_x = my_noc_x;
    } else {
        noc_x = destination_noc_x;
    }
    uint16_t noc_y;
    if constexpr ((IsBlockSharded && IsColumnMajor) || IsWidthSharded) {
        noc_y = my_noc_y;
    } else {
        noc_y = destination_noc_y;
    }
    return get_noc_addr(noc_x, noc_y, out_base_l1_addr);
}

template <uint32_t StickSizeBytes, uint32_t PageSize, bool EnableBlocking, uint32_t BlockHeightSticks>
static inline void write_stick_async(
    uint32_t in_base_l1_addr,
    uint64_t out_base_l1_addr,
    uint16_t src_offset_id,
    uint16_t dst_offset_id,
    uint16_t transfer_size) {
    if constexpr (EnableBlocking) {
        const uint32_t src_offset = (src_offset_id % BlockHeightSticks) *
                                    PageSize;  // Convert from global stick offset to local block stick offset
        const uint32_t dst_offset = dst_offset_id * StickSizeBytes;
        const uint32_t size = transfer_size * StickSizeBytes;
        const uint32_t src_addr = in_base_l1_addr + src_offset;
        const uint64_t dst_addr = out_base_l1_addr + dst_offset;
        noc_async_write(src_addr, dst_addr, size);
    } else {
        const uint32_t src_offset = src_offset_id * PageSize;
        const uint32_t dst_offset = dst_offset_id * StickSizeBytes;
        const uint32_t size = transfer_size * StickSizeBytes;
        const uint32_t src_addr = in_base_l1_addr + src_offset;
        const uint64_t dst_addr = out_base_l1_addr + dst_offset;
        noc_async_write(src_addr, dst_addr, size);
    }
}

template <
    uint32_t InputCBIndex,
    uint32_t OutputCBIndex,
    uint32_t StickSizeBytes,
    uint32_t InputPageSizeAligned,
    uint32_t BlockSizeHeight,
    uint32_t BlockSizeWidthTiles,
    uint32_t BlockStride,
    uint32_t BlockStartOffset,
    bool EnableBlocking,
    bool IsBlockSharded,
    bool IsWidthSharded,
    bool IsColumnMajor>
static inline void run_halo_gather(const tt_l1_ptr uint16_t* config, uint32_t my_noc_x, uint32_t my_noc_y) {
    static_assert(BlockStride >= 1, "Blocks stride must be at least 1");

    constexpr uint32_t block_size_height_tiles = BlockSizeHeight / TILE_SIZE;
    constexpr uint32_t total_tiles_in_single_block = block_size_height_tiles * BlockSizeWidthTiles;

    uint16_t current_config_index = 0;
    uint16_t number_of_segments_remaining = config[current_config_index++];

    if (number_of_segments_remaining == 0) {
        return;
    }

    uint32_t in_base_l1_addr = get_read_ptr(InputCBIndex);
    const uint32_t out_base_l1_addr = get_write_ptr(OutputCBIndex);

    // Assume input is already ready when !EnableBlocking (like when using RM)
    if constexpr (EnableBlocking) {
        cb_wait_front(InputCBIndex, total_tiles_in_single_block);
    }

    uint64_t out_l1_addr = 0;
    uint16_t block_id = BlockStartOffset;
    uint16_t block_boundary_offset = BlockSizeHeight + (BlockSizeHeight * BlockStartOffset);
    while (number_of_segments_remaining) {
        //  Read header for to get destination for this route
        const uint16_t destination_noc_x = config[current_config_index++];
        const uint16_t destination_noc_y = config[current_config_index++];
        uint16_t transfers_remaining = config[current_config_index++];

        out_l1_addr = get_remote_core_l1_noc_addr<IsBlockSharded, IsWidthSharded, IsColumnMajor>(
            destination_noc_x, destination_noc_y, my_noc_x, my_noc_y, out_base_l1_addr);

        // Perform all transfers in this route
        while (transfers_remaining > 0) {
            const uint16_t src_offset = config[current_config_index++];
            const uint16_t dst_offset = config[current_config_index++];
            const uint16_t transfer_size = config[current_config_index++];
            if constexpr (EnableBlocking) {
                // Pop blocks until we have the right one - this works because transfers are globally ordered by
                // ascending block IDs
                while (src_offset >= block_boundary_offset) {
                    noc_async_write_barrier();
                    cb_pop_front(InputCBIndex, total_tiles_in_single_block);
                    cb_wait_front(InputCBIndex, total_tiles_in_single_block);
                    block_boundary_offset +=
                        BlockSizeHeight *
                        BlockStride;  // When block stride > 1 we are expecting the input CB to skip
                                      // BlockStride number of blocks (like when splitting work across cores)
                    block_id += BlockStride;
                    in_base_l1_addr = get_read_ptr(InputCBIndex);  // Ensure base address is at front of input CB
                }
            }
            write_stick_async<StickSizeBytes, InputPageSizeAligned, EnableBlocking, BlockSizeHeight>(
                in_base_l1_addr, out_l1_addr, src_offset, dst_offset, transfer_size);
            transfers_remaining--;
        }
        number_of_segments_remaining--;
    }

    if constexpr (EnableBlocking) {
        cb_pop_front(InputCBIndex, total_tiles_in_single_block);
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
    constexpr uint32_t stick_nbytes = get_compile_time_arg_val(8);
    constexpr bool is_block_sharded = get_compile_time_arg_val(9) == 1;
    constexpr bool remote_read = get_compile_time_arg_val(10) == 1;
    constexpr bool is_col_major = get_compile_time_arg_val(11) == 1;
    constexpr bool is_width_sharded = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t input_aligned_page_size = get_compile_time_arg_val(13);
    constexpr bool skip_untilize = get_compile_time_arg_val(14) == 1;
    constexpr uint32_t block_size_height = get_compile_time_arg_val(15);
    constexpr uint32_t block_size_width_tiles = get_compile_time_arg_val(16);
    constexpr uint32_t block_start_offset = get_compile_time_arg_val(17);
    constexpr uint32_t block_stride = get_compile_time_arg_val(18);

    static_assert(!remote_read, "Remote read is not supported in this kernel");

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr bool enable_blocking = !skip_untilize;

    const uint16_t my_noc_x = NOC_X(my_x[noc_index]);
    const uint16_t my_noc_y = NOC_Y(my_y[noc_index]);

    // Only one of the cores should push the input
    if constexpr (block_start_offset == 0) {
        cb_reserve_back(src_cb_id, in_nsticks);
        cb_push_back(src_cb_id, in_nsticks);
    }

    if constexpr (padding_config_cb_id != 0) {
        if constexpr (pad_val_u32 == 0) {
            // Use MEM_ZEROS_BASE if we are zero padded
            const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, MEM_ZEROS_BASE);
            constexpr uint32_t padding_region_size = MEM_ZEROS_SIZE;
            copy_padding<padding_config_cb_id, out_cb_id, stick_nbytes, padding_region_size>(padding_l1_addr);
        } else {
            constexpr uint16_t pad_val = static_cast<uint16_t>(pad_val_u32);
            constexpr uint32_t num_elements_to_fill = stick_nbytes / elem_nbytes;
            fill_with_val<num_elements_to_fill, pad_val>(get_write_ptr(pad_cb_id));

            const uint64_t padding_l1_addr = get_noc_addr(my_noc_x, my_noc_y, get_read_ptr(pad_cb_id));
            constexpr uint32_t padding_region_size = stick_nbytes / elem_nbytes;
            copy_padding<padding_config_cb_id, out_cb_id, stick_nbytes, padding_region_size>(padding_l1_addr);
        }
    }

    if constexpr (skip_untilize) {
        cb_wait_front(src_cb_id, in_nsticks);
    }

    const uint32_t config_data_l1_addr = get_read_ptr(gather_config_cb_id);
    const tt_l1_ptr uint16_t* config_data = reinterpret_cast<const tt_l1_ptr uint16_t*>(config_data_l1_addr);
    run_halo_gather<
        in_cb_id,
        out_cb_id,
        stick_nbytes,
        input_aligned_page_size,
        block_size_height,
        block_size_width_tiles,
        block_stride,
        block_start_offset,
        enable_blocking,
        is_block_sharded,
        is_width_sharded,
        is_col_major>(config_data, my_noc_x, my_noc_y);

    noc_async_read_barrier();
    noc_async_write_barrier();
}
