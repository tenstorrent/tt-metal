// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <stdint.h>
#include <api/dataflow/dataflow_api.h>
#include <ttnn/cpp/ttnn/operations/pool/device/kernels/pool_kernels_common.hpp>
#include "../grid_sample_reader_common.hpp"
// experimental headers already included via grid_sample_reader_common.hpp

// Validate the (h, w) pixel index and read its stick (or the zero stick) into the output
template <uint32_t input_height, uint32_t input_width, uint32_t input_stick_nbytes, typename TensorAccessor>
ALWI void issue_nearest_read(
    Noc noc,
    int32_t nearest_h,
    int32_t nearest_w,
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset,
    DataflowBuffer output_dfb,
    uint32_t output_write_offset,
    uint32_t fill_stick_addr) {
    // Full coordinate validation for both precomputed and regular grids.
    // This catches out-of-bounds coordinates from padded shard data in addition
    // to the precomputed grid sentinel value (-1), which fails the >= 0 check.
    const bool h_valid = is_coordinate_valid(nearest_h, input_height);
    const bool w_valid = is_coordinate_valid(nearest_w, input_width);

    if (h_valid && w_valid) {
        // Read the nearest neighbor pixel
        const uint32_t input_stick_index = batch_offset + (nearest_h * input_width) + nearest_w;
        noc.async_read(
            input_tensor_accessor,
            output_dfb,
            input_stick_nbytes,
            {.page_id = input_stick_index},
            {.offset_bytes = output_write_offset});
    } else {
        UnicastEndpoint self_ep;
        noc.async_read(
            self_ep,
            output_dfb,
            input_stick_nbytes,
            experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
            {.offset_bytes = output_write_offset});
    }
}

// Copy a stick already in this core's output shard (src_off) into another slot (dst_off): a local NoC
// read, which costs the issuing core nothing on the NoC of the core that holds the image stick.
ALWI void local_stick_copy(Noc noc, DataflowBuffer output_dfb, uint32_t out_base, uint32_t src_off, uint32_t dst_off, uint32_t nbytes) {
    UnicastEndpoint self_ep;
    noc.async_read(
        self_ep, output_dfb, nbytes, experimental::local_addr(out_base + src_off, noc.get_noc_id()), {.offset_bytes = dst_off});
}

// Wait until a CB's "pages received" stream register (zeroed by the firmware at launch, written by the
// compute threads of grid_sample_nearest_index.cpp as plain counters) reaches want (16-bit, wrap safe).
ALWI void wait_received(uint32_t cb, uint32_t want) {
    volatile tt_reg_ptr uint32_t* p = get_cb_tiles_received_ptr(cb);
    const uint16_t w = static_cast<uint16_t>(want);
    while (static_cast<int16_t>(static_cast<uint16_t>(*p) - w) < 0) {
    }
}

// Process single grid point for nearest neighbor - adapted from common utilities
template <
    uint32_t grid_dtype,
    bool is_sharded,
    bool use_precomputed_grid,
    bool align_corners,
    uint32_t input_height,
    uint32_t input_width,
    uint32_t input_stick_nbytes,
    uint32_t output_cb_index,
    typename TensorAccessor,
    typename GridPtrType>
ALWI void process_grid_point_nearest(
    Noc noc,
    GridPtrType grid_ptr,
    uint32_t grid_idx,
    const TensorAccessor& input_tensor_accessor,
    uint32_t batch_offset,
    DataflowBuffer output_dfb,
    uint32_t output_write_offset,
    uint32_t fill_stick_addr) {
    // Compute scaling factors to match prepare_grid.cpp
    constexpr float input_height_f = float(input_height);
    constexpr float input_width_f = float(input_width);

    // Scale factors for coordinate transformation based on align_corners mode
    constexpr float height_scale =
        align_corners ? ((input_height > 1) ? (input_height_f - 1.0f) * 0.5f : 0.0f) : input_height_f * 0.5f;
    constexpr float width_scale =
        align_corners ? ((input_width > 1) ? (input_width_f - 1.0f) * 0.5f : 0.0f) : input_width_f * 0.5f;
    constexpr float height_offset = align_corners ? 0.0f : -0.5f;
    constexpr float width_offset = align_corners ? 0.0f : -0.5f;

    int32_t nearest_h, nearest_w;

    if constexpr (use_precomputed_grid) {
        // Each precomputed grid entry for nearest mode has 2 values (coordinates only)
        const uint32_t precomputed_data_offset = grid_idx * PRECOMPUTED_GRID_ELEMENTS_PER_POINT_NEAREST;
        const int16_t h_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 0]);
        const int16_t w_raw = *reinterpret_cast<volatile int16_t*>(&grid_ptr[precomputed_data_offset + 1]);

        nearest_h = static_cast<int32_t>(h_raw);
        nearest_w = static_cast<int32_t>(w_raw);
    } else {
        // Each regular grid entry has 2 values: x, y coordinates - compute nearest neighbor
        if constexpr (grid_dtype == DTYPE_FLOAT32) {
            // For FLOAT32 grid, each coordinate is a 32-bit float
            volatile tt_l1_ptr float* float_data = reinterpret_cast<volatile tt_l1_ptr float*>(grid_ptr);
            const uint32_t float_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            const float w_coord_rel = float_data[float_offset + 0];  // x coordinate
            const float h_coord_rel = float_data[float_offset + 1];  // y coordinate

            // Transform to image coordinates using the same formula as prepare_grid.cpp
            const float h_coord_image = ((h_coord_rel + 1.0f) * height_scale) + height_offset;
            const float w_coord_image = ((w_coord_rel + 1.0f) * width_scale) + width_offset;
            // Round half to even to match PyTorch grid_sample nearest (std::nearbyint). The rounding rule
            // is identical for both align_corners modes; only the coordinate transform above differs.
            nearest_h = round_half_to_even(h_coord_image);
            nearest_w = round_half_to_even(w_coord_image);
        } else {
            // For BFLOAT16 grid: exact integer arithmetic, same result as the fp32 formula above, without
            // the soft-float cost (the RISC-V has no FPU). -1 = outside the image.
            const uint32_t coordinate_pair_offset = grid_idx * STANDARD_GRID_ELEMENTS_PER_POINT;
            const uint16_t h_coord_raw = grid_ptr[coordinate_pair_offset + 1];  // y coordinate
            const uint16_t w_coord_raw = grid_ptr[coordinate_pair_offset + 0];  // x coordinate
            nearest_h = nearest_index_bf16<align_corners, input_height>(h_coord_raw);
            nearest_w = nearest_index_bf16<align_corners, input_width>(w_coord_raw);
        }
    }

    issue_nearest_read<input_height, input_width, input_stick_nbytes>(
        noc, nearest_h, nearest_w, input_tensor_accessor, batch_offset, output_dfb, output_write_offset, fill_stick_addr);
}

// Advance grid index utility - same as sharded reader
template <bool is_sharded>
ALWI void advance_grid_index(
    uint32_t& in_grid_row_idx,
    uint32_t& grid_stick_idx,
    uint32_t& l1_grid_addr,
    uint32_t& grid_points_processed,
    uint32_t& curr_batch,
    const uint32_t grid_batching_factor,
    const uint32_t grid_stick_nbytes,
    const uint32_t grid_hw,
    const uint32_t grid_nsticks_per_core) {
    ++in_grid_row_idx;
    if (in_grid_row_idx == grid_batching_factor) {
        in_grid_row_idx = 0;
        ++grid_stick_idx;
        if constexpr (is_sharded) {
            l1_grid_addr += grid_stick_nbytes;
        }
        ++grid_points_processed;
        if (grid_points_processed == grid_hw) {
            grid_points_processed = 0;
            ++curr_batch;
        }
    }
}

void kernel_main() {
    // Compile time arguments - same as sharded reader
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(1);  // output instead of scalar
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(2);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t input_height = get_compile_time_arg_val(4);
    constexpr uint32_t input_width = get_compile_time_arg_val(5);
    constexpr uint32_t grid_batching_factor = get_compile_time_arg_val(6);
    constexpr uint32_t grid_dtype = get_compile_time_arg_val(7);
    constexpr uint32_t grid_hw = get_compile_time_arg_val(8);
    constexpr uint32_t use_precomputed_grid = get_compile_time_arg_val(9);
    constexpr uint32_t align_corners = get_compile_time_arg_val(10);
    constexpr uint32_t split_reader = get_compile_time_arg_val(11);
    constexpr uint32_t reader_id = get_compile_time_arg_val(12);
    constexpr uint32_t grid_nsticks_per_core = get_compile_time_arg_val(13);
    constexpr uint32_t is_sharded = get_compile_time_arg_val(14);
    constexpr uint32_t fill_cb_index = get_compile_time_arg_val(15);
    constexpr uint32_t batch_size = get_compile_time_arg_val(16);
    // output_is_sharded decouples output offset logic from grid sharding.
    // When is_sharded=false (DRAM grid) but output is HEIGHT_SHARDED L1,
    // the output CB write pointer already points to this core's L1 shard start.
    // Using a global output_base would overflow L1. Use 0 instead.
    constexpr uint32_t output_is_sharded = get_compile_time_arg_val(17);
    // Sharded bf16 grid without precompute: the compute RISCs convert the grid shard into int16 (h, w)
    // indices in idx_cb (grid_sample_nearest_index.cpp), a third of the sticks each, and publish their
    // progress in the received counters of prog_cb0..2; this kernel then only reads indices.
    constexpr uint32_t idx_cb = get_compile_time_arg_val(18);
    constexpr uint32_t prog_cb0 = get_compile_time_arg_val(19);
    constexpr uint32_t prog_cb1 = get_compile_time_arg_val(20);
    constexpr uint32_t prog_cb2 = get_compile_time_arg_val(21);
    constexpr bool indices_from_cb = get_compile_time_arg_val(22) != 0;
    // Rank-5 (N, H_out, W_out, K, 2) grid in DRAM: this core's 4-byte grid pages are fetched into a staging CB at
    // start with one bulk read per DRAM bank (a bank's pages are contiguous, aligned page size apart), the two
    // readers taking half of the banks each and signalling on grid_ready_cb / grid_ready_cb1; the compute threads
    // then compute the indices straight from the staged runs.
    constexpr bool gather_grid = get_compile_time_arg_val(23) != 0;
    constexpr uint32_t stage_cb = get_compile_time_arg_val(24);
    constexpr uint32_t grid_ready_cb = get_compile_time_arg_val(25);
    constexpr uint32_t grid_page_aligned_bytes = get_compile_time_arg_val(26);
    constexpr uint32_t num_grid_banks = get_compile_time_arg_val(27);
    constexpr uint32_t stage_run_bytes = get_compile_time_arg_val(28);
    constexpr uint32_t grid_ready_cb1 = get_compile_time_arg_val(29);
    constexpr uint32_t grid_total_pages = get_compile_time_arg_val(30);
    // Read de-duplication (indices from the compute threads, sharded output). A grid that samples the
    // same image sticks over and over (a lookup table that maps many cells to one region) sends most of
    // its reads to a few source cores, and the op is then bound by those cores' NoC ports. The compute
    // threads detect the repeats (grid_sample_nearest_index.cpp) and publish them as "copy from sample
    // N"; this reader serves such a sample with a local copy from slot N instead of a NoC read. Reads are
    // issued in chunks with one barrier per chunk: a repeat of a sample read in an earlier chunk is copied
    // at once, one read in the current chunk (still in flight) is copied after the chunk's barrier.
    constexpr uint32_t dedup_cb = get_compile_time_arg_val(31);
    constexpr bool dedup = indices_from_cb && (output_is_sharded != 0) && (dedup_cb < 32);
    constexpr uint32_t DEDUP_ENTRIES = 256;  // per table of the compute threads (six tables lead the CB); must match the program factory and the compute kernel
    constexpr uint32_t DEDUP_CHUNK = 256;    // samples per read barrier; must match the program factory
    constexpr uint32_t DEDUP_TABLES_BYTES = 6 * DEDUP_ENTRIES * 8;

    uint32_t input_addr = 0;
    uint32_t global_grid_stick_start = 0;
    uint32_t grid_addr = 0;
    uint32_t start_page_id = 0;
    if constexpr (is_sharded) {
        // Runtime arguments - same as sharded reader
        input_addr = get_arg_val<uint32_t>(0);
        global_grid_stick_start = get_arg_val<uint32_t>(1);
        if constexpr (gather_grid) {
            grid_addr = get_arg_val<uint32_t>(2);
        }
    } else {
        input_addr = get_arg_val<uint32_t>(0);
        grid_addr = get_arg_val<uint32_t>(1);
        start_page_id = get_arg_val<uint32_t>(3);
        global_grid_stick_start = start_page_id;
    }

    // Input tensor accessor for remote NOC reads - same as sharded reader
    constexpr auto input_tensor_args = TensorAccessorArgs<32>();
    const auto input_tensor_accessor = TensorAccessor(input_tensor_args, input_addr);

    constexpr auto grid_tensor_args = TensorAccessorArgs<input_tensor_args.next_compile_time_args_offset()>();
    const auto grid_tensor_accessor = TensorAccessor(grid_tensor_args, grid_addr);

    // Calculate starting batch from global grid stick position
    const uint32_t starting_batch = global_grid_stick_start / grid_hw;

    // Get local grid data base address (already in L1)
    DataflowBuffer grid_dfb(grid_cb_index);
    DataflowBuffer output_dfb(output_cb_index);
    DataflowBuffer fill_dfb(fill_cb_index);
    Noc noc;

    const uint32_t l1_grid_base_addr = grid_dfb.get_write_ptr();
    if constexpr (gather_grid) {
        DataflowBuffer stage_dfb(stage_cb);
        const uint32_t stage_base = stage_dfb.get_write_ptr();
        const uint32_t page0 = global_grid_stick_start * grid_batching_factor;
        // the last output shard may be partial: never read past the grid's pages (those sticks are written as zeros)
        const uint32_t npages = (page0 >= grid_total_pages) ? 0 : ((grid_nsticks_per_core * grid_batching_factor < grid_total_pages - page0) ? grid_nsticks_per_core * grid_batching_factor : grid_total_pages - page0);
        // BRISC (NoC 0) takes two thirds of the banks: NoC 1 is markedly slower at DRAM reads
        constexpr uint32_t bank_split = (num_grid_banks * 2) / 3;
        const uint32_t bank_lo = (reader_id == 0) ? 0 : bank_split;
        const uint32_t bank_hi = (reader_id == 0) ? bank_split : num_grid_banks;
        // The pages page0.. of bank b are contiguous in that bank, aligned page size apart: one bulk read per bank
        // and chunk. The runs are fetched in GATHER_CHUNKS slices of the in-bank index so the compute threads can
        // start on the first sticks while the rest is still in flight; this reader's counter = chunks landed.
        constexpr uint32_t GATHER_CHUNKS = 8;
        // chunk boundaries are in units of the in-bank index and must match the compute kernel's, which derives them
        // from the full per-core page count; only the read lengths are clamped
        constexpr uint32_t max_count = (grid_nsticks_per_core * grid_batching_factor + num_grid_banks - 1) / num_grid_banks;
        constexpr uint32_t ic = (max_count + GATHER_CHUNKS - 1) / GATHER_CHUNKS;
        volatile tt_reg_ptr uint32_t* my_ready = get_cb_tiles_received_ptr(reader_id == 0 ? grid_ready_cb : grid_ready_cb1);
        for (uint32_t c = 0; c < GATHER_CHUNKS; ++c) {
            for (uint32_t b = bank_lo; b < bank_hi; ++b) {
                const uint32_t first = page0 + ((b + num_grid_banks - (page0 % num_grid_banks)) % num_grid_banks);
                if (first >= page0 + npages) {
                    continue;
                }
                const uint32_t count = (page0 + npages - 1 - first) / num_grid_banks + 1;
                const uint32_t i_lo = c * ic;
                const uint32_t i_hi = ((c + 1) * ic < count) ? (c + 1) * ic : count;
                if (i_lo >= i_hi) {
                    continue;
                }
                noc_async_read(
                    grid_tensor_accessor.get_noc_addr(first + i_lo * num_grid_banks),
                    stage_base + b * stage_run_bytes + i_lo * grid_page_aligned_bytes,
                    (i_hi - i_lo) * grid_page_aligned_bytes);
            }
            noc_async_read_barrier();
            *my_ready = static_cast<uint16_t>(c + 1);
        }
    }
    uint32_t idx_base = 0;
    if constexpr (indices_from_cb) {
        DataflowBuffer idx_dfb(idx_cb);
        idx_base = idx_dfb.get_write_ptr();
    }
    uint32_t dedup_defer = 0, dedup_out_base = 0;
    uint32_t dedup_n = 0, dedup_completed = 0, dedup_deferred = 0;  // samples below dedup_completed have landed
    if constexpr (dedup) {
        DataflowBuffer dedup_dfb(dedup_cb);
        dedup_defer = dedup_dfb.get_write_ptr() + DEDUP_TABLES_BYTES + reader_id * DEDUP_CHUNK * 8;
        dedup_out_base = output_dfb.get_write_ptr();
    }

    // Process each grid stick assigned to this core
    uint32_t grid_stick_idx = 0;
    uint32_t l1_grid_addr = l1_grid_base_addr;

    uint32_t fill_stick_addr = fill_dfb.get_write_ptr();
    zero_out_page(noc, fill_dfb);

    // For split reader: track grid point index starting from reader_id
    uint32_t in_grid_row_idx = 0;

    // Track current batch and grid position for batch increment logic
    uint32_t curr_batch = starting_batch;
    uint32_t grid_points_processed = global_grid_stick_start % grid_hw;
    // Advance at start if needed (for split reader)
    if constexpr (split_reader && reader_id == 1) {
        advance_grid_index<is_sharded>(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw,
            grid_nsticks_per_core);
    }

    while (grid_stick_idx < grid_nsticks_per_core) {
        volatile tt_l1_ptr uint16_t* grid_stick_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_grid_addr);

        uint32_t batch_offset = curr_batch * input_height * input_width;

        if constexpr (!is_sharded) {
            noc.async_read(
                grid_tensor_accessor, grid_dfb, grid_stick_nbytes, {.page_id = grid_stick_idx + start_page_id}, {});
            noc.async_read_barrier();
        }
        // For sharded output, the CB write pointer already starts at this core's
        // shard region (offset 0 is correct). For non-sharded output, add a
        // per-core spatial offset so each core writes to a distinct region.
        // output_is_sharded is set based on the OUTPUT tensor's sharding, not
        // the grid's, so DRAM grid + HEIGHT_SHARDED L1 output works correctly.
        const uint32_t output_base =
            output_is_sharded ? 0 : global_grid_stick_start * grid_batching_factor * input_stick_nbytes;
        uint32_t output_write_offset =
            output_base + grid_stick_idx * grid_batching_factor * input_stick_nbytes + in_grid_row_idx * input_stick_nbytes;

        if constexpr (indices_from_cb) {
            if (curr_batch < batch_size) {
                // wait for the compute thread that owns this stick (groups of 8: UNPACK 0-2, MATH 3-4, PACK 5-7);
                // a thread publishes a whole stick at once, so one wait per stick covers its K samples. With two
                // readers the samples alternate between them (reader r takes the sample indices of parity r), so
                // this reader's first sample of the stick is k = r ^ ((s * K) & 1).
                const uint32_t first_k = split_reader ? (reader_id ^ ((grid_stick_idx * grid_batching_factor) & 1)) : 0;
                if (in_grid_row_idx == first_k) {
                    const uint32_t slot = grid_stick_idx & 7;
                    const uint32_t group = grid_stick_idx >> 3;
                    const uint32_t owner = (slot < 3) ? 0 : ((slot < 5) ? 1 : 2);
                    const uint32_t local = (owner == 0) ? (group * 3 + slot) : ((owner == 1) ? (group * 2 + slot - 3) : (group * 3 + slot - 5));
                    wait_received((owner == 0) ? prog_cb0 : ((owner == 1) ? prog_cb1 : prog_cb2), local + 1);
                }
                const uint32_t hw = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    idx_base + (grid_stick_idx * grid_batching_factor + in_grid_row_idx) * 4);
                if constexpr (dedup) {
                    const uint32_t sample = grid_stick_idx * grid_batching_factor + in_grid_row_idx;
                    if (hw == 0xFFFFFFFFu) {
                        UnicastEndpoint self_ep;
                        noc.async_read(
                            self_ep,
                            output_dfb,
                            input_stick_nbytes,
                            experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
                            {.offset_bytes = output_write_offset});
                    } else if (hw & 0x80000000u) {
                        const uint32_t src = hw & 0x7FFFFFFFu;
                        if (src < dedup_completed) {
                            local_stick_copy(noc, output_dfb, dedup_out_base, src * input_stick_nbytes, output_write_offset, input_stick_nbytes);
                        } else {
                            uint32_t* d = reinterpret_cast<uint32_t*>(dedup_defer + dedup_deferred * 8);
                            d[0] = output_write_offset;
                            d[1] = src * input_stick_nbytes;
                            ++dedup_deferred;
                        }
                    } else {
                        noc.async_read(
                            input_tensor_accessor,
                            output_dfb,
                            input_stick_nbytes,
                            {.page_id = hw},
                            {.offset_bytes = output_write_offset});
                    }
                    if (++dedup_n == DEDUP_CHUNK) {
                        noc.async_read_barrier();
                        dedup_completed = sample + 1;
                        for (uint32_t i = 0; i < dedup_deferred; ++i) {
                            const uint32_t* d = reinterpret_cast<const uint32_t*>(dedup_defer + i * 8);
                            local_stick_copy(noc, output_dfb, dedup_out_base, d[1], d[0], input_stick_nbytes);
                        }
                        dedup_deferred = 0;
                        dedup_n = 0;
                    }
                } else {
                    issue_nearest_read<input_height, input_width, input_stick_nbytes>(
                        noc,
                        static_cast<int32_t>(static_cast<int16_t>(hw & 0xFFFF)),
                        static_cast<int32_t>(static_cast<int16_t>(hw >> 16)),
                        input_tensor_accessor,
                        batch_offset,
                        output_dfb,
                        output_write_offset,
                        fill_stick_addr);
                }
            } else {
                UnicastEndpoint self_ep;
                noc.async_read(
                    self_ep,
                    output_dfb,
                    input_stick_nbytes,
                    experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
                    {.offset_bytes = output_write_offset});
            }
        } else if (curr_batch < batch_size) {
            // Process nearest neighbor sampling and write directly to output
            process_grid_point_nearest<
                grid_dtype,
                is_sharded,
                use_precomputed_grid,
                align_corners,
                input_height,
                input_width,
                input_stick_nbytes,
                output_cb_index>(
                noc,
                grid_stick_ptr,
                in_grid_row_idx,
                input_tensor_accessor,
                batch_offset,
                output_dfb,
                output_write_offset,
                fill_stick_addr);
        } else {
            // Padding stick beyond valid batches - write zeros
            UnicastEndpoint self_ep;
            noc.async_read(
                self_ep,
                output_dfb,
                input_stick_nbytes,
                experimental::local_addr(fill_stick_addr, noc.get_noc_id()),
                {.offset_bytes = output_write_offset});
        }

        // Always advance once after processing
        advance_grid_index<is_sharded>(
            in_grid_row_idx,
            grid_stick_idx,
            l1_grid_addr,
            grid_points_processed,
            curr_batch,
            grid_batching_factor,
            grid_stick_nbytes,
            grid_hw,
            grid_nsticks_per_core);

        // For split reader, advance one more time to skip the coordinate that the other reader will process
        if constexpr (split_reader) {
            advance_grid_index<is_sharded>(
                in_grid_row_idx,
                grid_stick_idx,
                l1_grid_addr,
                grid_points_processed,
                curr_batch,
                grid_batching_factor,
                grid_stick_nbytes,
                grid_hw,
                grid_nsticks_per_core);
        }
    }
    if constexpr (dedup) {
        if (dedup_deferred != 0) {
            noc.async_read_barrier();
            for (uint32_t i = 0; i < dedup_deferred; ++i) {
                const uint32_t* d = reinterpret_cast<const uint32_t*>(dedup_defer + i * 8);
                local_stick_copy(noc, output_dfb, dedup_out_base, d[1], d[0], input_stick_nbytes);
            }
        }
    }
    noc.async_read_barrier();
}
