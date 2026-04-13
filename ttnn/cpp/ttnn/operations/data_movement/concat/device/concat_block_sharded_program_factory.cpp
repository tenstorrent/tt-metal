// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_block_sharded_program_factory.hpp"

#include <algorithm>
#include <numeric>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

struct TransferDesc {
    CoreCoord src_physical_core;
    uint32_t src_cb_id;
    uint32_t src_l1_offset;
    uint32_t src_stride;
    uint32_t dst_offset;
    uint32_t dst_stride;
    uint32_t copy_size;
    uint32_t num_rows;
};

ConcatBlockShardedProgramFactory::cached_program_t ConcatBlockShardedProgramFactory::create(
    const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const uint32_t dim = operation_attributes.dim;
    Tensor& output = tensor_return_value;

    const uint32_t rank = input_tensors[0].logical_shape().rank();
    TT_FATAL(
        dim == rank - 2 || dim == rank - 1,
        "Block-sharded concat only supports the last two dims (H={}, W={}), got dim={}",
        rank - 2,
        rank - 1,
        dim);
    const bool is_width_concat = dim == rank - 1;

    Program program = CreateProgram();

    const uint32_t num_input_tensors = input_tensors.size();
    const uint32_t cb_dst_id = 16;
    TT_FATAL(num_input_tensors <= cb_dst_id, "Not enough circular buffers for {} inputs.", num_input_tensors);

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const bool rm_layout = output.layout() == Layout::ROW_MAJOR;
    const uint32_t element_size = input_tensors[0].element_size();

    const auto all_cores = input_tensors[0].shard_spec().value().grid;
    TT_FATAL(
        all_cores.ranges().size() == 1,
        "Block-sharded concat requires a single contiguous rectangular CoreRange, got {} ranges",
        all_cores.ranges().size());
    const auto& core_range = *all_cores.ranges().begin();
    const uint32_t start_x = core_range.start_coord.x;
    const uint32_t start_y = core_range.start_coord.y;
    const uint32_t grid_cols = core_range.end_coord.x - start_x + 1;
    const uint32_t grid_rows = core_range.end_coord.y - start_y + 1;

    auto* device = input_tensors[0].device();

    uint32_t unit_h, unit_w, unit_size;
    if (rm_layout) {
        unit_h = 1;
        unit_w = 1;
        unit_size = element_size;
    } else {
        unit_h = TILE_HEIGHT;
        unit_w = TILE_WIDTH;
        unit_size = tt::tile_size(cb_data_format);
    }

    std::vector<uint32_t> input_shard_h(num_input_tensors);
    std::vector<uint32_t> input_shard_w(num_input_tensors);

    for (uint32_t i = 0; i < num_input_tensors; i++) {
        const auto& ss = input_tensors[i].shard_spec().value();
        input_shard_h[i] = ss.shape[0];
        input_shard_w[i] = ss.shape[1];
        if (!rm_layout) {
            TT_FATAL(
                input_shard_h[i] % TILE_HEIGHT == 0 && input_shard_w[i] % TILE_WIDTH == 0,
                "Input {} shard shape ({}, {}) is not tile-aligned ({}x{})",
                i,
                input_shard_h[i],
                input_shard_w[i],
                TILE_HEIGHT,
                TILE_WIDTH);
        }
    }

    const auto& out_ss = output.shard_spec().value();
    const uint32_t output_shard_h = out_ss.shape[0];
    const uint32_t output_shard_w = out_ss.shape[1];
    if (!rm_layout) {
        TT_FATAL(
            output_shard_h % TILE_HEIGHT == 0 && output_shard_w % TILE_WIDTH == 0,
            "Output shard shape ({}, {}) is not tile-aligned ({}x{})",
            output_shard_h,
            output_shard_w,
            TILE_HEIGHT,
            TILE_WIDTH);
    }

    auto to_units_h = [&](uint32_t h) { return h / unit_h; };
    auto to_units_w = [&](uint32_t w) { return w / unit_w; };

    const uint32_t out_units_w = to_units_w(output_shard_w);
    const uint32_t dst_stride_bytes = out_units_w * unit_size;

    // Create input CBs
    std::vector<CBHandle> cb_inputs(num_input_tensors);
    for (uint32_t i = 0; i < num_input_tensors; i++) {
        const uint32_t in_num_units = to_units_h(input_shard_h[i]) * to_units_w(input_shard_w[i]);
        const CircularBufferConfig cb_config = CircularBufferConfig(unit_size * in_num_units, {{i, cb_data_format}})
                                                   .set_page_size(i, unit_size)
                                                   .set_globally_allocated_address(*input_tensors[i].buffer());
        cb_inputs[i] = CreateCircularBuffer(program, all_cores, cb_config);
    }

    // Create output CB
    const uint32_t out_num_units = to_units_h(output_shard_h) * out_units_w;
    const CircularBufferConfig out_cb_config =
        CircularBufferConfig(unit_size * out_num_units, {{cb_dst_id, cb_data_format}})
            .set_page_size(cb_dst_id, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = CreateCircularBuffer(program, all_cores, out_cb_config);

    // Pre-compute physical core map
    std::vector<std::vector<CoreCoord>> physical_cores(grid_rows, std::vector<CoreCoord>(grid_cols));
    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            physical_cores[gy][gx] = device->worker_core_from_logical_core(CoreCoord(start_x + gx, start_y + gy));
        }
    }

    // Pass 1: compute transfer descriptors for all cores
    // Index as [gy * grid_cols + gx]
    std::vector<std::vector<TransferDesc>> all_transfers(grid_rows * grid_cols);

    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            auto& transfers = all_transfers[gy * grid_cols + gx];

            if (is_width_concat) {
                const uint32_t out_col_start = gx * output_shard_w;
                const uint32_t out_col_end = out_col_start + output_shard_w;
                const uint32_t num_rows_units = to_units_h(output_shard_h);

                uint32_t cum_w = 0;
                for (uint32_t inp_id = 0; inp_id < num_input_tensors; inp_id++) {
                    const uint32_t inp_total_w = input_shard_w[inp_id] * grid_cols;
                    const uint32_t inp_shard_w_val = input_shard_w[inp_id];

                    const uint32_t overlap_start = std::max(out_col_start, cum_w);
                    const uint32_t overlap_end = std::min(out_col_end, cum_w + inp_total_w);

                    if (overlap_start < overlap_end) {
                        const uint32_t local_col_start = overlap_start - cum_w;
                        const uint32_t local_col_end = overlap_end - cum_w;

                        const uint32_t first_src_c = local_col_start / inp_shard_w_val;
                        const uint32_t last_src_c = (local_col_end - 1) / inp_shard_w_val;

                        for (uint32_t src_c = first_src_c; src_c <= last_src_c; src_c++) {
                            const uint32_t src_shard_col_start = src_c * inp_shard_w_val;
                            const uint32_t needed_col_start = std::max(local_col_start, src_shard_col_start);
                            const uint32_t needed_col_end =
                                std::min(local_col_end, src_shard_col_start + inp_shard_w_val);
                            const uint32_t needed_cols = needed_col_end - needed_col_start;

                            const uint32_t src_col_off = needed_col_start - src_shard_col_start;
                            const uint32_t global_col = cum_w + needed_col_start;
                            const uint32_t dst_col_off = global_col - out_col_start;

                            const uint32_t src_stride_bytes = to_units_w(inp_shard_w_val) * unit_size;

                            transfers.push_back(TransferDesc{
                                .src_physical_core = physical_cores[gy][src_c],
                                .src_cb_id = inp_id,
                                .src_l1_offset = to_units_w(src_col_off) * unit_size,
                                .src_stride = src_stride_bytes,
                                .dst_offset = to_units_w(dst_col_off) * unit_size,
                                .dst_stride = dst_stride_bytes,
                                .copy_size = to_units_w(needed_cols) * unit_size,
                                .num_rows = num_rows_units,
                            });
                        }
                    }
                    cum_w += inp_total_w;
                }
            } else {
                // Height concat (dim=2)
                const uint32_t out_row_start = gy * output_shard_h;
                const uint32_t out_row_end = out_row_start + output_shard_h;
                const uint32_t copy_width = to_units_w(output_shard_w) * unit_size;

                uint32_t cum_h = 0;
                uint32_t dst_row_accum_units = 0;
                for (uint32_t inp_id = 0; inp_id < num_input_tensors; inp_id++) {
                    const uint32_t inp_total_h = input_shard_h[inp_id] * grid_rows;
                    const uint32_t inp_shard_h_val = input_shard_h[inp_id];

                    const uint32_t overlap_start = std::max(out_row_start, cum_h);
                    const uint32_t overlap_end = std::min(out_row_end, cum_h + inp_total_h);

                    if (overlap_start < overlap_end) {
                        const uint32_t local_row_start = overlap_start - cum_h;
                        const uint32_t local_row_end = overlap_end - cum_h;

                        const uint32_t first_src_r = local_row_start / inp_shard_h_val;
                        const uint32_t last_src_r = (local_row_end - 1) / inp_shard_h_val;

                        for (uint32_t src_r = first_src_r; src_r <= last_src_r; src_r++) {
                            const uint32_t src_shard_row_start = src_r * inp_shard_h_val;
                            const uint32_t needed_row_start = std::max(local_row_start, src_shard_row_start);
                            const uint32_t needed_row_end =
                                std::min(local_row_end, src_shard_row_start + inp_shard_h_val);
                            const uint32_t needed_rows = needed_row_end - needed_row_start;

                            const uint32_t src_row_off = needed_row_start - src_shard_row_start;
                            const uint32_t src_stride_bytes = to_units_w(input_shard_w[inp_id]) * unit_size;
                            const uint32_t src_offset = to_units_h(src_row_off) * src_stride_bytes;

                            transfers.push_back(TransferDesc{
                                .src_physical_core = physical_cores[src_r][gx],
                                .src_cb_id = inp_id,
                                .src_l1_offset = src_offset,
                                .src_stride = src_stride_bytes,
                                .dst_offset = dst_row_accum_units * dst_stride_bytes,
                                .dst_stride = dst_stride_bytes,
                                .copy_size = copy_width,
                                .num_rows = to_units_h(needed_rows),
                            });
                            dst_row_accum_units += to_units_h(needed_rows);
                        }
                    }
                    cum_h += inp_total_h;
                }
            }
        }
    }

    // Find max transfers across all cores (should be uniform for equal-shaped inputs)
    uint32_t max_num_transfers = 0;
    for (const auto& t : all_transfers) {
        max_num_transfers = std::max(max_num_transfers, static_cast<uint32_t>(t.size()));
    }
    TT_FATAL(max_num_transfers > 0, "No transfers computed for block-sharded concat");

    // Pass 2: create kernels and split transfers between reader and writer RISCs.
    // Both RISCs run the same kernel but with different subsets of transfers,
    // doubling effective NOC bandwidth (reader uses NOC0, writer uses NOC1).
    const std::vector<uint32_t> compile_time_args = {cb_dst_id};

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_writer_block_sharded_concat.cpp",
        all_cores,
        ReaderDataMovementConfig(compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/"
        "reader_writer_block_sharded_concat.cpp",
        all_cores,
        WriterDataMovementConfig(compile_time_args));

    auto build_runtime_args = [](const std::vector<TransferDesc>& descs, uint32_t start, uint32_t count) {
        std::vector<uint32_t> args;
        args.reserve(1 + count * 9);
        args.push_back(count);
        for (uint32_t i = start; i < start + count; i++) {
            const auto& td = descs[i];
            args.push_back(td.src_physical_core.x);
            args.push_back(td.src_physical_core.y);
            args.push_back(td.src_cb_id);
            args.push_back(td.src_l1_offset);
            args.push_back(td.src_stride);
            args.push_back(td.dst_offset);
            args.push_back(td.dst_stride);
            args.push_back(td.copy_size);
            args.push_back(td.num_rows);
        }
        return args;
    };

    for (uint32_t gy = 0; gy < grid_rows; gy++) {
        for (uint32_t gx = 0; gx < grid_cols; gx++) {
            const auto& transfers = all_transfers[gy * grid_cols + gx];
            const uint32_t total = static_cast<uint32_t>(transfers.size());
            const uint32_t reader_count = (total + 1) / 2;  // ceil(total / 2)
            const uint32_t writer_count = total - reader_count;

            auto reader_args = build_runtime_args(transfers, 0, reader_count);
            auto writer_args = build_runtime_args(transfers, reader_count, writer_count);

            CoreCoord logical_core(start_x + gx, start_y + gy);
            SetRuntimeArgs(program, reader_kernel_id, logical_core, reader_args);
            SetRuntimeArgs(program, writer_kernel_id, logical_core, writer_args);
        }
    }

    return {
        std::move(program),
        {.num_input_tensors = num_input_tensors,
         .cb_inputs = cb_inputs,
         .cb_output = cb_output,
         .all_cores = all_cores}};
}

void ConcatBlockShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const ConcatParams& /*operation_attributes*/,
    const ConcatInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_vars = cached_program.shared_variables;

    for (uint32_t i = 0; i < shared_vars.num_input_tensors; i++) {
        tt::tt_metal::UpdateDynamicCircularBufferAddress(
            program, shared_vars.cb_inputs[i], *tensor_args.input_tensors[i].buffer());
    }
    tt::tt_metal::UpdateDynamicCircularBufferAddress(program, shared_vars.cb_output, *tensor_return_value.buffer());
}

}  // namespace ttnn::prim
