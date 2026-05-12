// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Kernel mode encoding passed as compile-time arg 3 (see indexed_fill_reader.cpp header).
// TODO: upgrade to two-vector RuntimeTensorShape pattern (mirrors unary_ng / binary_ng)
constexpr uint32_t MODE_GENERIC = 0;
constexpr uint32_t MODE_NATIVE = 1;
constexpr uint32_t MODE_SHARD_LOCAL_INTERLEAVED_B = 2;
constexpr uint32_t MODE_SHARD_LOCAL_SHARDED_B = 3;

// Geometry shared between create() and override_runtime_arguments().
struct IndexedFillGeometry {
    uint32_t outer_count;
    uint32_t inner_count;
    uint32_t S_dim;
    uint32_t outer_stride_a;  // doubles as output stride: output shape == input_a shape
    uint32_t outer_stride_b;
    uint32_t shard_ppb;
    uint32_t total_batches_per_core;
};

IndexedFillGeometry compute_geometry(
    const Tensor& input_a, int64_t dim, uint32_t b, uint32_t B, bool is_tile, bool is_shard_local) {
    const int64_t rank = static_cast<int64_t>(input_a.padded_shape().rank());
    const int64_t page_boundary = is_tile ? rank - 2 : rank - 1;

    uint32_t outer_count = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_count *= input_a.padded_shape()[i];
    }

    uint32_t inner_count = 1;
    for (int64_t i = dim + 1; i < page_boundary; ++i) {
        inner_count *= input_a.padded_shape()[i];
    }
    if (is_tile) {
        inner_count *= (input_a.padded_shape()[rank - 2] / tt::constants::TILE_HEIGHT) *
                       (input_a.padded_shape()[rank - 1] / tt::constants::TILE_WIDTH);
    }

    const uint32_t S_dim = input_a.padded_shape()[dim];

    uint32_t shard_ppb = 0;
    uint32_t total_batches_per_core = 0;
    if (is_shard_local) {
        const auto& shard_spec = *input_a.memory_config().shard_spec();
        const uint32_t shard_width = shard_spec.shape[1];
        // Rank-4 assumed ([B, H, W, D]); enforced by validate_on_program_cache_miss.
        const uint32_t H_N = input_a.padded_shape()[1] * input_a.padded_shape()[2];
        if (is_tile) {
            shard_ppb = (H_N / tt::constants::TILE_HEIGHT) * (shard_width / tt::constants::TILE_WIDTH);
        } else {
            shard_ppb = H_N;
        }
        using tt::tt_metal::TensorMemoryLayout;
        if (input_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            total_batches_per_core = B;
        } else {
            total_batches_per_core = B / shard_spec.grid.bounding_box().grid_size().y;
        }
    }

    const uint32_t outer_stride_a = S_dim * inner_count;
    return {
        .outer_count = outer_count,
        .inner_count = inner_count,
        .S_dim = S_dim,
        .outer_stride_a = outer_stride_a,
        .outer_stride_b = b * inner_count,
        .shard_ppb = shard_ppb,
        .total_batches_per_core = total_batches_per_core,
    };
}

// Column-offset args for the shard-local INTERLEAVED-B reader path.
struct ShardColOffsets {
    uint32_t b_full_ppb;
    uint32_t shard_tile_w;
    uint32_t full_tile_w;
    uint32_t col_page_offset;
    uint32_t col_byte_offset;
};

ShardColOffsets compute_shard_col_offsets(
    const Tensor& input_a, const tt::tt_metal::ShardSpec& shard_spec, uint32_t cx, bool is_tile) {
    // Rank-4 assumed ([B, H, W, D]); enforced by validate_on_program_cache_miss.
    const uint32_t b_full_ppb =
        is_tile ? (input_a.padded_shape()[1] * input_a.padded_shape()[2] / tt::constants::TILE_HEIGHT) *
                      (input_a.padded_shape()[-1] / tt::constants::TILE_WIDTH)
                : input_a.padded_shape()[1] * input_a.padded_shape()[2];
    const uint32_t shard_tile_w = is_tile ? (shard_spec.shape[1] / tt::constants::TILE_WIDTH) : 1u;
    const uint32_t full_tile_w = is_tile ? (input_a.padded_shape()[-1] / tt::constants::TILE_WIDTH) : 1u;
    return {
        .b_full_ppb = b_full_ppb,
        .shard_tile_w = shard_tile_w,
        .full_tile_w = full_tile_w,
        .col_page_offset = is_tile ? cx * shard_tile_w : 0u,
        .col_byte_offset = !is_tile ? cx * shard_spec.shape[1] * input_a.element_size() : 0u,
    };
}

}  // namespace

IndexedFillProgramFactory::cached_program_t IndexedFillProgramFactory::create(
    const IndexedFillParams& operation_attributes, const IndexedFillInputs& tensor_args, Tensor& output) {
    const auto& batch_ids = tensor_args.batch_id;
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;

    Program program{};

    const int64_t dim = operation_attributes.dim;

    const uint32_t B = input_a.padded_shape()[0];
    // `b` = number of replacement slices = size of target dimension in input_b.
    const uint32_t b = static_cast<uint32_t>(input_b.padded_shape()[dim]);

    TT_ASSERT(batch_ids.padded_shape()[-1] == b);

    // Worker grid: set by get_indexed_fill_worker_grid (always non-empty).
    const CoreRangeSet all_cores = operation_attributes.worker_grid;
    auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    const uint32_t num_cores_total = static_cast<uint32_t>(cores.size());

    const uint32_t cb_index = 0;
    const uint32_t batch_cb_index = 1;
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.dtype());

    const bool is_tile = input_a.layout() == Layout::TILE;

    // -----------------------------------------------------------------------------------
    // Path selection
    // -----------------------------------------------------------------------------------
    // Native HEIGHT_SHARDED and shard-local WIDTH/BLOCK_SHARDED paths are designed around
    // dim=0 (one shard per batch).  For dim != 0 the shard grid does not align with the
    // target dimension, so always use the generic 2D-stride path instead.
    const bool is_native = (dim == 0) && ttnn::operations::data_movement::indexed_fill::is_native_indexed_fill_sharding(
                                             input_a.tensor_spec(),
                                             input_b.tensor_spec(),
                                             batch_ids.tensor_spec(),
                                             operation_attributes.output_mem_config);

    const bool is_shard_local =
        (dim == 0) && !is_native &&
        ttnn::operations::data_movement::indexed_fill::is_shard_local_indexed_fill(
            input_a.tensor_spec(), input_b.tensor_spec(), operation_attributes.output_mem_config);

    const bool b_same_sharded =
        is_shard_local && input_b.is_sharded() && input_b.memory_config().shard_spec().has_value() &&
        input_b.memory_config().shard_spec()->grid == input_a.memory_config().shard_spec()->grid;

    uint32_t kernel_mode = MODE_GENERIC;
    if (is_native) {
        kernel_mode = MODE_NATIVE;
    } else if (is_shard_local) {
        kernel_mode = b_same_sharded ? MODE_SHARD_LOCAL_SHARDED_B : MODE_SHARD_LOCAL_INTERLEAVED_B;
    }

    // -----------------------------------------------------------------------------------
    // Page geometry
    //
    // Full-tensor geometry (native / generic paths):
    //   page_size  = one full row (ROW_MAJOR) or one tile (TILE)
    //   inner_count = pages per (outer, slice) pair (== pages per batch for dim=0)
    //
    // Shard-local geometry (shard_local path):
    //   shard_page_size = shard_width * elem_size (ROW_MAJOR) or tile_size (TILE)
    //   shard_ppb       = pages per batch within this shard
    //   total_batches   = B (WIDTH) or B/n_y (BLOCK) per core
    // -----------------------------------------------------------------------------------
    const auto geo = compute_geometry(input_a, dim, b, B, is_tile, is_shard_local);
    const uint32_t outer_count = geo.outer_count;
    const uint32_t inner_count = geo.inner_count;
    const uint32_t S_dim = geo.S_dim;
    const uint32_t outer_stride_a = geo.outer_stride_a;
    const uint32_t outer_stride_b = geo.outer_stride_b;
    const uint32_t shard_ppb = geo.shard_ppb;
    const uint32_t total_batches_per_core = geo.total_batches_per_core;

    // Generic path: distribute S_dim slices across num_cores_total cores using ceiling-division.
    // Cores [0, extra) receive slices_per_core + 1 slices; others receive slices_per_core.
    // When S_dim < num_cores_total, cores with index >= S_dim receive num_slices == 0 (idle).

    uint32_t page_size = 0;
    if (is_tile) {
        page_size = tt::tile_size(cb_data_format);
    } else {
        page_size = input_a.padded_shape()[-1] * input_a.element_size();
    }
    const uint32_t rounded_page_size = is_tile ? page_size : round_up_to_mul32(page_size);

    uint32_t shard_page_size = 0;
    if (is_shard_local) {
        const auto& shard_spec = *input_a.memory_config().shard_spec();
        const uint32_t shard_width = shard_spec.shape[1];
        shard_page_size = is_tile ? tt::tile_size(cb_data_format) : shard_width * input_a.element_size();
    }

    // Use shard geometry for the kernel when on the shard_local path.
    const uint32_t kernel_page_size = is_shard_local ? shard_page_size : page_size;
    const uint32_t kernel_rounded_page_size = is_shard_local ? round_up_to_mul32(shard_page_size) : rounded_page_size;
    const uint32_t batch_size_in_pages = is_shard_local ? shard_ppb : inner_count;

    // -----------------------------------------------------------------------------------
    // Data CB (cb_index == 0).
    //
    // Native / shard-local path: globally allocate to the output buffer so reader writes
    // land directly in the output's per-core L1 shard, and the writer becomes a tiny
    // wait/pop stub. The CB must hold one full shard.
    //
    // Fallback path: local CB with the original double-buffered (2-page) capacity.
    // -----------------------------------------------------------------------------------
    CBHandle cb_data_handle{};
    if (is_native) {
        auto [_, handle] = create_cb(
            cb_index,
            program,
            all_cores,
            kernel_rounded_page_size,
            batch_size_in_pages,
            cb_data_format,
            output.buffer());
        cb_data_handle = handle;
    } else if (is_shard_local) {
        const uint32_t total_pages_in_shard = total_batches_per_core * shard_ppb;
        auto [_, handle] = create_cb(
            cb_index,
            program,
            all_cores,
            kernel_rounded_page_size,
            total_pages_in_shard,
            cb_data_format,
            output.buffer());
        cb_data_handle = handle;
    } else {
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(2 * kernel_rounded_page_size, {{cb_index, cb_data_format}})
                .set_page_size(cb_index, kernel_rounded_page_size);
        cb_data_handle = CreateCircularBuffer(program, all_cores, cb_src0_config);
    }

    const uint32_t batch_page_size = round_up_to_mul32(b * sizeof(uint32_t));
    CircularBufferConfig batch_cb_config = CircularBufferConfig(2 * batch_page_size, {{batch_cb_index, cb_data_format}})
                                               .set_page_size(batch_cb_index, batch_page_size);
    CreateCircularBuffer(program, all_cores, batch_cb_config);

    KernelHandle reader_kernel_id{};
    KernelHandle writer_kernel_id{};

    // ---- Reader: single unified kernel; path selected via `mode` compile-time arg.
    {
        std::vector<uint32_t> reader_compile_time_args = {
            static_cast<uint32_t>(cb_index), static_cast<uint32_t>(batch_cb_index), kernel_page_size, kernel_mode};
        TensorAccessorArgs(*input_a.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*input_b.buffer()).append_to(reader_compile_time_args);
        TensorAccessorArgs(*batch_ids.buffer()).append_to(reader_compile_time_args);

        reader_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp",
            all_cores,
            ReaderDataMovementConfig(reader_compile_time_args));
    }

    // ---- Writer
    if (is_native || is_shard_local) {
        // CB is aliased to the output buffer: writer just synchronises on the CB.
        std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(cb_index)};
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/"
            "indexed_fill_writer.cpp",
            all_cores,
            WriterDataMovementConfig(writer_compile_time_args));
    } else {
        // Generic path: scatter-write via the strided writer kernel.
        std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(cb_index)};
        TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);
        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/"
            "indexed_fill_writer_strided.cpp",
            all_cores,
            WriterDataMovementConfig(writer_compile_time_args));
    }

    // ---- Per-core runtime arguments
    // Work-splitting for the generic path: distribute S_dim slices across available cores.
    const uint32_t slices_per_core = (num_cores_total > 0) ? S_dim / num_cores_total : 0;
    const uint32_t extra_slices = (num_cores_total > 0) ? S_dim % num_cores_total : 0;

    const uint32_t shard_n_x = is_shard_local ? all_cores.bounding_box().grid_size().x : 0;

    // Precompute per-column offsets for the shard-local path: only shard_n_x unique cx values
    // exist, but the loop runs over all n_x * n_y cores. Avoids redundant recomputation for
    // every core in the same column.
    std::vector<ShardColOffsets> col_offsets;
    if (is_shard_local && shard_n_x > 0) {
        const auto& shard_spec_pre = *input_a.memory_config().shard_spec();
        col_offsets.resize(shard_n_x);
        for (uint32_t cx = 0; cx < shard_n_x; ++cx) {
            col_offsets[cx] = compute_shard_col_offsets(input_a, shard_spec_pre, cx, is_tile);
        }
    }

    for (uint32_t i = 0; i < num_cores_total; ++i) {
        const CoreCoord& core = cores[i];

        if (is_native) {
            const bool active = i < B;
            const uint32_t local_b = active ? b : 0;
            const uint32_t local_batch_size = active ? batch_size_in_pages : 0;

            // input_a's per-core L1 base is the same address value on every core in the
            // shard grid (sharded L1 buffers allocate a per-core L1 slot at a uniform offset).
            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                local_b,
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                local_batch_size,
                i,
                0u,  // batch_offset_a (unused in native mode)
                0u   // total_local_batches (unused in native mode)
            };
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

            const std::array writer_runtime_args = {local_batch_size};
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        } else if (is_shard_local) {
            // Shard row/col for BLOCK_SHARDED: indices within the shard grid bounding box.
            const uint32_t shard_row = (shard_n_x > 0) ? (i / shard_n_x) : 0;
            const uint32_t cx = (shard_n_x > 0) ? (i % shard_n_x) : 0;
            const uint32_t batch_offset_a = shard_row * total_batches_per_core;
            const uint32_t total_pages_in_shard = total_batches_per_core * shard_ppb;

            // Args 8-12: column-offset state for the INTERLEAVED_B read path (precomputed above).
            const auto& col = col_offsets[cx];

            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                b,
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                shard_ppb,  // batch_size_in_pages == shard_ppb for this path
                0u,         // my_batch_id (unused in shard_local mode)
                batch_offset_a,
                total_batches_per_core,
                col.b_full_ppb,
                col.shard_tile_w,
                col.full_tile_w,
                col.col_page_offset,
                col.col_byte_offset};
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

            const std::array writer_runtime_args = {total_pages_in_shard};
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        } else {
            // Generic 2D-stride path: each core handles a contiguous range of slices.
            uint32_t slice_start, num_slices;
            if (i < extra_slices) {
                slice_start = i * (slices_per_core + 1);
                num_slices = slices_per_core + 1;
            } else if (slices_per_core > 0) {
                slice_start = extra_slices * (slices_per_core + 1) + (i - extra_slices) * slices_per_core;
                num_slices = slices_per_core;
            } else {
                // S_dim < num_cores_total: core i >= S_dim is idle.
                slice_start = S_dim;
                num_slices = 0;
            }

            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                b,  // arg[1] = b (kernel exits early when num_slices == 0)
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                inner_count,     // arg[4] = inner_count
                slice_start,     // arg[5] = slice_start (first slice for this core)
                outer_count,     // arg[6] = outer_count
                outer_stride_a,  // arg[7]
                outer_stride_b,  // arg[8]
                num_slices,      // arg[9] = num_slices (NEW: number of slices for this core)
            };
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

            const std::array writer_runtime_args = {
                output.buffer()->address(),
                kernel_page_size,
                outer_count,     // arg[2]
                inner_count,     // arg[3]
                outer_stride_a,  // arg[4] = outer stride in output
                slice_start,     // arg[5] = slice_start (first slice for this core)
                num_slices,      // arg[6] = num_slices (NEW)
            };
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
    }

    return cached_program_t{
        std::move(program),
        IndexedFillSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = std::move(cores),
            .page_size = kernel_page_size,
            .is_native = is_native,
            .is_shard_local = is_shard_local,
            .is_tile = is_tile,
            .cb_data_handle = cb_data_handle,
        }};
}

void IndexedFillProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const IndexedFillParams& operation_attributes,
    const IndexedFillInputs& tensor_args,
    Tensor& output) {
    const auto& batch_ids = tensor_args.batch_id;
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& kernel_page_size = cached_program.shared_variables.page_size;
    const bool is_native = cached_program.shared_variables.is_native;
    const bool is_shard_local = cached_program.shared_variables.is_shard_local;

    const int64_t dim = operation_attributes.dim;

    const uint32_t B = input_a.padded_shape()[0];
    const uint32_t b = static_cast<uint32_t>(input_b.padded_shape()[dim]);
    const bool is_tile = cached_program.shared_variables.is_tile;

    const auto geo = compute_geometry(input_a, dim, b, B, is_tile, is_shard_local);
    const uint32_t outer_count = geo.outer_count;
    const uint32_t inner_count = geo.inner_count;
    const uint32_t S_dim = geo.S_dim;
    const uint32_t outer_stride_a = geo.outer_stride_a;
    const uint32_t outer_stride_b = geo.outer_stride_b;
    const uint32_t shard_ppb = geo.shard_ppb;
    const uint32_t total_batches_per_core = geo.total_batches_per_core;
    const uint32_t batch_size_in_pages = inner_count;

    // Work-splitting for the generic path (same formula as in create()).
    const uint32_t num_cores = static_cast<uint32_t>(cores.size());
    const uint32_t slices_per_core_ov = (num_cores > 0) ? S_dim / num_cores : 0;
    const uint32_t extra_slices_ov = (num_cores > 0) ? S_dim % num_cores : 0;

    uint32_t shard_n_x = 0;
    if (is_shard_local) {
        shard_n_x = static_cast<uint32_t>(input_a.memory_config().shard_spec()->grid.bounding_box().grid_size().x);
    }

    // Native / shard-local: re-point the data CB at the (possibly moved) output buffer.
    if (is_native || is_shard_local) {
        UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_data_handle, *output.buffer());
    }

    // Precompute per-column offsets for the shard-local path (same reasoning as in create()).
    std::vector<ShardColOffsets> col_offsets;
    if (is_shard_local && shard_n_x > 0) {
        const auto& shard_spec_pre = *input_a.memory_config().shard_spec();
        col_offsets.resize(shard_n_x);
        for (uint32_t cx = 0; cx < shard_n_x; ++cx) {
            col_offsets[cx] = compute_shard_col_offsets(input_a, shard_spec_pre, cx, is_tile);
        }
    }

    uint32_t core_id = 0;
    for (const auto& core : cores) {
        if (is_native) {
            const bool active = core_id < B;
            const uint32_t local_b = active ? b : 0;
            const uint32_t local_batch_size = active ? batch_size_in_pages : 0;

            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                local_b,
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                local_batch_size,
                core_id,
                0u,
                0u};
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            const std::array writer_runtime_args = {local_batch_size};
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        } else if (is_shard_local) {
            const uint32_t shard_row = (shard_n_x > 0) ? (core_id / shard_n_x) : 0;
            const uint32_t cx = (shard_n_x > 0) ? (core_id % shard_n_x) : 0;
            const uint32_t batch_offset_a = shard_row * total_batches_per_core;
            const uint32_t total_pages_in_shard = total_batches_per_core * shard_ppb;

            const auto& col = col_offsets[cx];

            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                b,
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                shard_ppb,
                0u,
                batch_offset_a,
                total_batches_per_core,
                col.b_full_ppb,
                col.shard_tile_w,
                col.full_tile_w,
                col.col_page_offset,
                col.col_byte_offset};
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            const std::array writer_runtime_args = {total_pages_in_shard};
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        } else {
            // Generic 2D-stride path: same work-splitting as create().
            uint32_t slice_start, num_slices;
            if (core_id < extra_slices_ov) {
                slice_start = core_id * (slices_per_core_ov + 1);
                num_slices = slices_per_core_ov + 1;
            } else if (slices_per_core_ov > 0) {
                slice_start =
                    extra_slices_ov * (slices_per_core_ov + 1) + (core_id - extra_slices_ov) * slices_per_core_ov;
                num_slices = slices_per_core_ov;
            } else {
                slice_start = S_dim;
                num_slices = 0;
            }

            const std::array reader_runtime_args = {
                batch_ids.buffer()->address(),
                b,
                input_a.buffer()->address(),
                input_b.buffer()->address(),
                inner_count,
                slice_start,
                outer_count,
                outer_stride_a,
                outer_stride_b,
                num_slices,
            };
            SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
            const std::array writer_runtime_args = {
                output.buffer()->address(),
                kernel_page_size,
                outer_count,
                inner_count,
                outer_stride_a,
                slice_start,
                num_slices,
            };
            SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }
        core_id++;
    }
}

}  // namespace ttnn::prim
