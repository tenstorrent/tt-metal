// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 halo (untilize_with_halo) gather reader.
//
// Two instances of this kernel (reader0 on RISCV_0, reader1 on RISCV_1) run on every
// node and scatter-write DISJOINT regions of the output shard.
//
// Resource model (see untilize_with_halo_program_factory.cpp):
//   tensor::out             - output shard (borrowed-memory address source: its raw L1 base,
//                             the same on every core; both readers scatter-write it).
//   tensor::in              - input shard (skip_untilize path only): read directly by
//                             base pointer (LocalTensorAccessor).
//   tensor::gather_config   - gather config (L1 path): read by base pointer (LocalTensorAccessor).
//   tensor::padding_config  - padding config (L1 path): read by base pointer (LocalTensorAccessor).
//   dfb::untilize_out       - untilized tiles produced by compute (tiled path): the
//                             gather source FIFO this reader consumes.
//   dfb::pad_fill / dfb::pad_read
//                           - cross-reader pad-immediate scratch (pad_val != 0): this reader fills
//                             pad_fill (its own pad DFB) and broadcasts from pad_read (the peer
//                             reader's identical pad DFB), avoiding a DM-kernel self-loop.
//   dfb::gather_config_scratch / dfb::padding_config_scratch
//                           - DRAM config landing (config_tensors_in_dram path).

#include <stdint.h>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/local_tensor_accessor.h"
#include "internal/tensor/transfer_noc_addr.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"

constexpr uint16_t TILE_SIZE = 32;

template <uint32_t N, uint16_t PaddingValue>
FORCE_INLINE void fill_with_val(uint32_t begin_addr) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < N; ++i) {
        ptr[i] = PaddingValue;
    }
}

// Copy `nsticks` copies of the resident pad value (at padding_l1_addr) to dst_addr.
template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding_small_sticks(Noc noc, uint32_t padding_l1_addr, uint32_t dst_addr, uint16_t nsticks) {
    static_assert(MaxChunkSize >= StickNBytes, "This function assumes max chunk size > stick size");

    constexpr uint32_t sticks_per_batch = MaxChunkSize / StickNBytes;
    constexpr uint32_t batch_size_bytes = sticks_per_batch * StickNBytes;
    static_assert(batch_size_bytes <= NOC_MAX_BURST_SIZE, "batch_size_bytes must be single-packet");

    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];

    if constexpr (sticks_per_batch > 1) {
        const uint16_t num_full_batches = nsticks / sticks_per_batch;
        const uint16_t remaining_sticks = nsticks % sticks_per_batch;

        uint32_t current_dst = dst_addr;
        for (uint16_t batch = 0; batch < num_full_batches; ++batch) {
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(current_dst),
                batch_size_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            current_dst += batch_size_bytes;
        }
        for (uint16_t k = 0; k < remaining_sticks; ++k) {
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(current_dst),
                StickNBytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            current_dst += StickNBytes;
        }
    } else {
        uint32_t current_dst = dst_addr;
        for (uint16_t k = 0; k < nsticks; ++k) {
            noc.async_read(
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
    static_assert(MaxChunkSize <= NOC_MAX_BURST_SIZE, "MaxChunkSize must be single-packet");
    static_assert(remainder_bytes <= NOC_MAX_BURST_SIZE, "remainder must be single-packet");

    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];

    if constexpr (num_full_chunks > 0) {
        uint32_t stick_base_addr = dst_addr;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            uint32_t chunk_addr = stick_base_addr;
            for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
                noc.async_read(
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
        uint32_t remainder_base_addr = dst_addr + remainder_offset;
        for (uint16_t stick = 0; stick < nsticks; ++stick) {
            noc.async_read(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(remainder_base_addr),
                remainder_bytes,
                {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = padding_l1_addr},
                {});
            remainder_base_addr += StickNBytes;
        }
    }
}

// Reads (dst_local_idx, nsticks) pairs from padding_config_l1_addr until nsticks == 0,
// filling each run of sticks at out_base + dst_local_idx*stick from padding_l1_addr.
template <uint32_t StickNBytes, uint32_t MaxChunkSize>
FORCE_INLINE void copy_padding(
    Noc noc, uint32_t padding_config_l1_addr, uint32_t dst_base_addr, uint32_t padding_l1_addr) {
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(padding_config_l1_addr);

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

// Scatter-write `transfer_size` sticks from the local in-source (in_base_l1_addr + src_offset)
// to the destination core's out shard (out_base + dst_offset).
template <uint32_t StickSizeBytes, bool EnableBlocking, uint32_t BlockHeightSticks>
static inline void write_stick_async(
    Noc noc,
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

    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];

    // src is local L1 (current core); dst is the destination core's L1 (NoC unicast).
    noc.async_write(
        UnicastEndpoint{},
        UnicastEndpoint{},
        size,
        {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_addr},
        {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = dst_addr});
}

void kernel_main() {
    constexpr uint32_t pad_val_u32 = get_arg(args::pad_val);
    constexpr uint32_t aligned_stick_nbytes = get_arg(args::aligned_stick_nbytes);
    constexpr bool is_block_sharded = get_arg(args::is_block_sharded) == 1;
    constexpr bool remote_read = get_arg(args::remote_read) == 1;
    constexpr bool is_col_major = get_arg(args::is_col_major) == 1;
    constexpr bool is_width_sharded = get_arg(args::is_width_sharded) == 1;
    constexpr bool skip_untilize = get_arg(args::skip_untilize) == 1;
    constexpr uint32_t block_size_height = get_arg(args::block_size_height);
    constexpr uint32_t block_size_width_tiles = get_arg(args::block_size_width_tiles);
    constexpr uint32_t block_start_offset = get_arg(args::block_start_offset);
    constexpr uint32_t block_stride = get_arg(args::block_stride);
    constexpr bool enable_padding = get_arg(args::enable_padding) == 1;

    static_assert(!remote_read, "Remote read is not supported in this kernel");
    static_assert(block_stride >= 1, "Block stride must be at least 1");

    constexpr uint32_t elem_nbytes = sizeof(uint16_t);
    constexpr bool enable_blocking = !skip_untilize;

    constexpr uint32_t block_size_height_tiles = block_size_height / TILE_SIZE;
    constexpr uint32_t total_tiles_in_single_block = block_size_height_tiles * block_size_width_tiles;

    // Several CTAs are consumed only via preprocessor gates / templates on some build
    // configurations; silence -Werror=unused on the inert configs.
    (void)pad_val_u32;
    (void)skip_untilize;
    (void)enable_padding;
    (void)elem_nbytes;
    (void)total_tiles_in_single_block;

    Noc noc;

    // Output shard base (borrowed-memory address source; both readers scatter-write here).
    TensorAccessor out_acc(tensor::out);
    // Raw: same L1 address on another core (scatter-write destination; no handle form yet). Also this core's
    // padding-fill base.
    const uint32_t out_base_l1_addr = tensor_accessor::detail::TransferAccess::bank_base(out_acc);

    // Gather/padding config base L1 addresses.
    uint32_t gather_config_l1_addr;
    uint32_t padding_config_l1_addr = 0;
#ifdef CONFIG_TENSOR_IN_DRAM
    // DRAM config: async-read the per-core page into a private L1 scratch, then read it.
    constexpr uint32_t config_read_index = get_arg(args::config_read_index);
    TensorAccessor gather_config_acc(tensor::gather_config);
    DataflowBuffer gather_config_scratch(dfb::gather_config_scratch);
    gather_config_l1_addr = gather_config_scratch.get_write_ptr();
    noc.async_read(
        gather_config_acc,
        gather_config_scratch,
        gather_config_scratch.get_entry_size(),
        {.page_id = config_read_index},
        {});
#ifdef ENABLE_PADDING
    TensorAccessor padding_config_acc(tensor::padding_config);
    DataflowBuffer padding_config_scratch(dfb::padding_config_scratch);
    padding_config_l1_addr = padding_config_scratch.get_write_ptr();
    noc.async_read(
        padding_config_acc,
        padding_config_scratch,
        padding_config_scratch.get_entry_size(),
        {.page_id = config_read_index},
        {});
#endif
    noc.async_read_barrier();
#else
    // L1 config: the config tensor is sharded one shard per core; read the local shard directly.
    // Raw address: shared with the DRAM-scratch path above and read through raw pointers below.
    LocalTensorAccessor<uint16_t> gather_config_shard(tensor::gather_config);
    gather_config_l1_addr =
        static_cast<uint32_t>(local_mem::detail::AddressAccess::address(gather_config_shard.local_mem()));
#ifdef ENABLE_PADDING
    LocalTensorAccessor<uint16_t> padding_config_shard(tensor::padding_config);
    padding_config_l1_addr =
        static_cast<uint32_t>(local_mem::detail::AddressAccess::address(padding_config_shard.local_mem()));
#endif
#endif

    const uint16_t my_noc_x = my_x[noc.get_noc_id()];
    const uint16_t my_noc_y = my_y[noc.get_noc_id()];

#ifndef SKIP_UNTILIZE
    // Blocks the compute untilizes on this core: the last block the gather config references + 1, which can be
    // fewer than the input shard holds (e.g. the padded tail rows of the last core).
    const uint32_t total_blocks = get_arg(args::total_blocks);
#endif

#ifdef SRC_PRODUCER
    // reader0 fake-pushes the borrowed SRC DFB: the input shard is already resident
    // (borrowed_from = IN), so we only need to advance the FIFO front pointer so the
    // compute's wait_front(src) succeeds.  No data is moved. Push exactly the pages the compute pops
    // (total_blocks blocks), not the whole shard: src_cb lives to kernel exit and its destructor waits for
    // the compute to pop everything pushed, which needs this reader to drain untilize_out first.
    const uint32_t src_npages = total_blocks * total_tiles_in_single_block;
    DataflowBuffer src_cb(dfb::src);
    if (src_npages > 0) {
        src_cb.reserve_back(static_cast<uint16_t>(src_npages));
        src_cb.push_back(static_cast<uint16_t>(src_npages));
    }
#endif

    // ----- Padding fill -----
#ifdef ENABLE_PADDING
    {
        // Materialize the immediate pad stick into this reader's pad DFB (pad_fill) and publish it,
        // then broadcast from the PEER reader's identical pad DFB (pad_read).  Cross-reader so neither
        // DM kernel self-loops a DFB; the pad value is the same constant on both readers and both run
        // on the same core, so the peer buffer is an equivalent local-L1 source.
        //
        // Quasar has no static MEM_ZEROS L1 region (WH/BH-only in dev_mem_map.h), so the zero-pad case
        // can't copy from it. Instead always go through the pad-scratch DFB (the factory now allocates
        // it for pad_val==0 too) and zero it with the Quasar noc zero-write for pad_val==0, else fill
        // with the immediate value.
        constexpr uint16_t pad_val = static_cast<uint16_t>(pad_val_u32);
        constexpr uint32_t num_elements_to_fill = aligned_stick_nbytes / elem_nbytes;
        DataflowBuffer pad_fill_cb(dfb::pad_fill);
        DataflowBuffer pad_read_cb(dfb::pad_read);

        pad_fill_cb.reserve_back(1);
        if constexpr (pad_val == 0) {
            noc.async_write_zeros(pad_fill_cb, aligned_stick_nbytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
        } else {
            fill_with_val<num_elements_to_fill, pad_val>(pad_fill_cb.get_write_ptr());
#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)
            // Quasar-only: flush the RISC CPU store to TL1 before push_back so the peer reader / unpacker
            // sees it (WH/BH CPU stores are coherent to the consumer, and flush_l2_cache_range is tt-2xx-only).
            flush_l2_cache_range(
                static_cast<uintptr_t>(pad_fill_cb.get_write_ptr()), static_cast<size_t>(aligned_stick_nbytes));
#endif
        }
        pad_fill_cb.push_back(1);

        pad_read_cb.wait_front(1);
        constexpr uint32_t padding_region_size = aligned_stick_nbytes;
        copy_padding<aligned_stick_nbytes, padding_region_size>(
            noc, padding_config_l1_addr, out_base_l1_addr, pad_read_cb.get_read_ptr());
        pad_read_cb.pop_front(1);
    }
#endif

    // ----- Gather -----
    const tt_l1_ptr uint16_t* config = reinterpret_cast<const tt_l1_ptr uint16_t*>(gather_config_l1_addr);

    uint16_t current_config_index = 0;
    uint16_t number_of_segments_remaining = config[current_config_index++];

#ifndef SKIP_UNTILIZE
    // The compute untilizes every block in [0, total_blocks) and sends this reader the ones at
    // block_start_offset, block_start_offset + block_stride, ...  This reader has to pop all of them, including
    // blocks after its last referenced one (or all of them if it has no transfers); otherwise the compute's
    // untilize_out drain never completes.
    DataflowBuffer in_cb(dfb::untilize_out);
    const uint32_t own_blocks =
        total_blocks > block_start_offset ? (total_blocks - block_start_offset + block_stride - 1) / block_stride : 0;
    uint32_t own_blocks_popped = 0;
#endif

    if (number_of_segments_remaining != 0) {
        // The gather source: untilized tiles (tiled path) or the resident input shard (skip path).
        uint32_t in_base_l1_addr = 0;

        uint16_t block_id = block_start_offset;
        uint16_t block_boundary_offset = block_size_height + (block_size_height * block_start_offset);
        (void)block_id;
        (void)block_boundary_offset;

#ifndef SKIP_UNTILIZE
        {
            in_cb.wait_front(total_tiles_in_single_block);
            in_base_l1_addr = in_cb.get_read_ptr();

            while (number_of_segments_remaining) {
                const uint16_t destination_noc_x = config[current_config_index++];
                const uint16_t destination_noc_y = config[current_config_index++];
                uint16_t transfers_remaining = config[current_config_index++];

                uint16_t dst_noc_x, dst_noc_y;
                resolve_destination_coords<is_block_sharded, is_width_sharded, is_col_major>(
                    destination_noc_x, destination_noc_y, my_noc_x, my_noc_y, dst_noc_x, dst_noc_y);

                while (transfers_remaining > 0) {
                    const uint16_t src_offset = config[current_config_index++];
                    const uint16_t dst_offset = config[current_config_index++];
                    const uint16_t transfer_size = config[current_config_index++];
                    // Pop blocks until we have the right one (transfers are globally ordered by ascending block id).
                    while (src_offset >= block_boundary_offset) {
                        noc.async_write_barrier();
                        in_cb.pop_front(total_tiles_in_single_block);
                        own_blocks_popped++;
                        in_cb.wait_front(total_tiles_in_single_block);
                        in_base_l1_addr = in_cb.get_read_ptr();
                        block_boundary_offset += block_size_height * block_stride;
                        block_id += block_stride;
                    }
                    write_stick_async<aligned_stick_nbytes, enable_blocking, block_size_height>(
                        noc,
                        in_base_l1_addr,
                        out_base_l1_addr,
                        dst_noc_x,
                        dst_noc_y,
                        src_offset,
                        dst_offset,
                        transfer_size);
                    transfers_remaining--;
                }
                number_of_segments_remaining--;
            }
            in_cb.pop_front(total_tiles_in_single_block);
            own_blocks_popped++;
        }
#else
        {
            // skip_untilize: the resident input shard is the gather source (read directly).
            // Raw address: write_stick_async takes the source as a raw L1 address (shared with the DFB path).
            LocalTensorAccessor<uint16_t> in_shard(tensor::in);
            in_base_l1_addr = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(in_shard.local_mem()));
            while (number_of_segments_remaining) {
                const uint16_t destination_noc_x = config[current_config_index++];
                const uint16_t destination_noc_y = config[current_config_index++];
                uint16_t transfers_remaining = config[current_config_index++];

                uint16_t dst_noc_x, dst_noc_y;
                resolve_destination_coords<is_block_sharded, is_width_sharded, is_col_major>(
                    destination_noc_x, destination_noc_y, my_noc_x, my_noc_y, dst_noc_x, dst_noc_y);

                while (transfers_remaining > 0) {
                    const uint16_t src_offset = config[current_config_index++];
                    const uint16_t dst_offset = config[current_config_index++];
                    const uint16_t transfer_size = config[current_config_index++];
                    write_stick_async<aligned_stick_nbytes, enable_blocking, block_size_height>(
                        noc,
                        in_base_l1_addr,
                        out_base_l1_addr,
                        dst_noc_x,
                        dst_noc_y,
                        src_offset,
                        dst_offset,
                        transfer_size);
                    transfers_remaining--;
                }
                number_of_segments_remaining--;
            }
        }
#endif
    }

#ifndef SKIP_UNTILIZE
    // Pop the blocks no transfer referenced (no data to send).
    for (; own_blocks_popped < own_blocks; own_blocks_popped++) {
        in_cb.wait_front(total_tiles_in_single_block);
        in_cb.pop_front(total_tiles_in_single_block);
    }
#endif

    noc.async_read_barrier();
    noc.async_write_barrier();
}
