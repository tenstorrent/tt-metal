// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "fold_device_op.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

struct FoldTransfer {
    uint16_t src_noc_x;
    uint16_t src_noc_y;
    uint16_t src_local_idx;
    uint16_t length = 1;
};

// Special marker for padding pixels - kernel will fill with zeros
constexpr uint16_t PADDING_NOC_MARKER = 0xFFFF;

std::vector<std::vector<FoldTransfer>> generate_fold_transfers(
    const Tensor& input,
    const Tensor& output,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right) {
    auto input_shape = input.logical_shape();

    // Input shard spec - for source NOC coordinates
    auto input_shard_spec = input.shard_spec();
    uint32_t input_shard_height = input_shard_spec->shape[0];
    uint32_t input_num_cores = input_shard_spec->grid.num_cores();

    // Output shard spec - for destination core assignment
    auto output_shard_spec = output.shard_spec();
    uint32_t output_shard_height = output_shard_spec->shape[0];
    uint32_t output_num_cores = output_shard_spec->grid.num_cores();

    uint32_t N = input_shape[0];
    uint32_t H = input_shape[1];
    uint32_t W = input_shape[2];

    // Padded dimensions
    uint32_t padded_H = H + pad_top + pad_bottom;
    uint32_t padded_W = W + pad_left + pad_right;

    uint32_t out_H = padded_H / stride_h;
    uint32_t out_W = padded_W / stride_w;

    // Get NOC coordinates for INPUT cores (source of reads)
    auto input_logical_cores = tt::tt_metal::corerange_to_cores(
        input_shard_spec->grid, input_num_cores, input_shard_spec->orientation == ShardOrientation::ROW_MAJOR);

    std::vector<CoreCoord> input_noc_coords;
    for (const auto& core : input_logical_cores) {
        input_noc_coords.push_back(input.device()->worker_core_from_logical_core(core));
    }

    // Per-core transfers sized for OUTPUT cores
    std::vector<std::vector<FoldTransfer>> per_core_transfers(output_num_cores);

    uint32_t output_pixel = 0;
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t h = 0; h < out_H; h++) {
            for (uint32_t w = 0; w < out_W; w++) {
                // Destination core based on OUTPUT shard height
                uint32_t dst_core_idx = output_pixel / output_shard_height;

                // For each row in the stride group, try to coalesce stride_w consecutive elements
                for (uint32_t s_h = 0; s_h < stride_h; s_h++) {
                    uint32_t padded_h = h * stride_h + s_h;
                    uint32_t padded_w_start = w * stride_w;

                    // Check if entire row is in height padding region
                    bool row_h_padding = (padded_h < pad_top) || (padded_h >= pad_top + H);

                    if (row_h_padding) {
                        // Entire row is padding - coalesce all stride_w as padding
                        per_core_transfers[dst_core_idx].push_back(
                            {PADDING_NOC_MARKER, PADDING_NOC_MARKER, 0, static_cast<uint16_t>(stride_w)});
                    } else {
                        // Row is not in height padding, check width elements
                        uint32_t in_h = padded_h - pad_top;
                        uint32_t s_w = 0;

                        while (s_w < stride_w) {
                            uint32_t padded_w = padded_w_start + s_w;
                            bool is_w_padding = (padded_w < pad_left) || (padded_w >= pad_left + W);

                            if (is_w_padding) {
                                // Count consecutive padding elements
                                uint32_t pad_count = 0;
                                while (s_w + pad_count < stride_w) {
                                    uint32_t pw = padded_w_start + s_w + pad_count;
                                    if ((pw >= pad_left) && (pw < pad_left + W)) {
                                        break;  // No longer padding
                                    }
                                    pad_count++;
                                }
                                per_core_transfers[dst_core_idx].push_back(
                                    {PADDING_NOC_MARKER, PADDING_NOC_MARKER, 0, static_cast<uint16_t>(pad_count)});
                                s_w += pad_count;
                            } else {
                                // Real data - try to coalesce consecutive elements on same core
                                uint32_t in_w = padded_w - pad_left;
                                uint32_t in_idx = (n * H * W) + (in_h * W) + in_w;
                                uint32_t first_core_id = in_idx / input_shard_height;
                                uint32_t first_local_idx = in_idx % input_shard_height;

                                // Count how many consecutive elements are on the same core and not padding
                                uint32_t coalesce_count = 1;
                                while (s_w + coalesce_count < stride_w) {
                                    uint32_t next_padded_w = padded_w_start + s_w + coalesce_count;
                                    // Check if next element is padding
                                    if ((next_padded_w < pad_left) || (next_padded_w >= pad_left + W)) {
                                        break;  // Next is padding
                                    }
                                    // Check if next element is on same core
                                    uint32_t next_in_w = next_padded_w - pad_left;
                                    uint32_t next_in_idx = (n * H * W) + (in_h * W) + next_in_w;
                                    uint32_t next_core_id = next_in_idx / input_shard_height;
                                    if (next_core_id != first_core_id) {
                                        break;  // Different core, can't coalesce
                                    }
                                    coalesce_count++;
                                }

                                auto [src_noc_x, src_noc_y] = input_noc_coords[first_core_id];
                                per_core_transfers[dst_core_idx].push_back(
                                    {static_cast<uint16_t>(src_noc_x),
                                     static_cast<uint16_t>(src_noc_y),
                                     static_cast<uint16_t>(first_local_idx),
                                     static_cast<uint16_t>(coalesce_count)});
                                s_w += coalesce_count;
                            }
                        }
                    }
                }
                output_pixel++;
            }
        }
    }
    return per_core_transfers;
}

uint32_t get_max_transfers_per_core(const std::vector<std::vector<FoldTransfer>>& per_core_transfers) {
    uint32_t max_entries = 0;
    for (const auto& core_transfers : per_core_transfers) {
        max_entries = std::max(max_entries, static_cast<uint32_t>(core_transfers.size()));
    }
    return max_entries;
}

Tensor create_fold_transfers_tensor(
    const std::vector<std::vector<FoldTransfer>>& per_core_transfers,
    const CoreRangeSet& core_grid,
    ShardOrientation orientation,
    const Tensor& input_tensor) {
    uint32_t num_cores = core_grid.num_cores();
    uint32_t max_entries = get_max_transfers_per_core(per_core_transfers);

    uint32_t config_shard_width = max_entries * 4;

    // Flatten the transfers into a single vector
    std::vector<uint16_t> flattened_data;
    for (auto core_transfers : per_core_transfers) {
        for (auto transfer : core_transfers) {
            flattened_data.push_back(transfer.src_noc_x);
            flattened_data.push_back(transfer.src_noc_y);
            flattened_data.push_back(transfer.src_local_idx);
            flattened_data.push_back(transfer.length);
        }
        uint32_t padding = (max_entries - core_transfers.size()) * 4;
        for (uint32_t i = 0; i < padding; i++) {
            flattened_data.push_back(0);
        }
    }
    ttnn::Shape config_shape({num_cores, config_shard_width});
    auto config_buffer = HostBuffer(std::move(flattened_data));
    Tensor host_tensor(std::move(config_buffer), config_shape, DataType::UINT16, Layout::ROW_MAJOR);

    ShardSpec config_shard_spec = ShardSpec(core_grid, {1, config_shard_width}, orientation);

    MemoryConfig config_memory_config =
        MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1_SMALL, config_shard_spec);
    return host_tensor.to_device(input_tensor.device(), config_memory_config);
}

Fold::MultiCore::cached_program_t fold_multi_core(
    const Tensor& input,
    const Tensor& output,
    uint32_t stride_h,
    uint32_t stride_w,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right) {
    Program program = CreateProgram();

    // Input tensor info
    auto input_shard_spec = input.shard_spec();
    auto input_shard_shape = input_shard_spec->shape;
    auto input_cores = input_shard_spec->grid;

    // Output tensor info - kernel runs on output cores
    auto output_shard_spec = output.shard_spec();
    auto output_shard_shape = output_shard_spec->shape;
    auto output_cores = output_shard_spec->grid;

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());

    // Input pixel info
    uint32_t pixel_size = input_shard_shape[1] * input.element_size();
    uint32_t input_shard_height = input_shard_shape[0];
    uint32_t aligned_pixel_size = tt::align(pixel_size, hal::get_l1_alignment());

    // Output pixel info
    uint32_t output_shard_height = output_shard_shape[0];

    // Input CB - globally allocated on input buffer, created on output cores for NOC reads
    uint32_t cb_src0_index = tt::CBIndex::c_0;
    auto src_cb_config =
        CircularBufferConfig(input_shard_height * aligned_pixel_size, {{cb_src0_index, cb_data_format}})
            .set_page_size(cb_src0_index, aligned_pixel_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = CreateCircularBuffer(program, output_cores, src_cb_config);

    // Output CB - globally allocated on output buffer, created on output cores
    uint32_t cb_dst0_index = tt::CBIndex::c_16;
    uint32_t output_pixel_size = output_shard_shape[1] * output.element_size();
    uint32_t aligned_output_pixel_size = tt::align(output_pixel_size, hal::get_l1_alignment());
    auto dst_cb_config =
        CircularBufferConfig(output_shard_height * aligned_output_pixel_size, {{cb_dst0_index, cb_data_format}})
            .set_page_size(cb_dst0_index, aligned_output_pixel_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = CreateCircularBuffer(program, output_cores, dst_cb_config);

    // Generate config tensor with input and output info (now with padding)
    auto per_core_transfers =
        generate_fold_transfers(input, output, stride_h, stride_w, pad_top, pad_bottom, pad_left, pad_right);
    uint32_t num_transfers = get_max_transfers_per_core(per_core_transfers);

    auto config_tensor =
        create_fold_transfers_tensor(per_core_transfers, output_cores, output_shard_spec->orientation, input);

    // Config CB - on output cores
    uint32_t config_cb_index = tt::CBIndex::c_1;
    uint32_t config_page_size = num_transfers * 4 * sizeof(uint16_t);

    auto config_cb_config = CircularBufferConfig(config_page_size, {{config_cb_index, tt::DataFormat::UInt16}})
                                .set_page_size(config_cb_index, config_page_size)
                                .set_globally_allocated_address(*config_tensor.buffer());
    CreateCircularBuffer(program, output_cores, config_cb_config);

    std::vector<uint32_t> compile_time_args = {
        cb_src0_index,       // 0: input CB
        cb_dst0_index,       // 1: output CB
        config_cb_index,     // 2: config CB
        pixel_size,          // 3: bytes per pixel
        aligned_pixel_size,  // 4: aligned pixel size
        num_transfers,       // 5: number of transfers per core
    };

    // Single kernel for fold with config tensor - runs on OUTPUT cores
    tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fold/device/kernels/dataflow/fold_with_config.cpp",
        output_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = compile_time_args});

    return {std::move(program), {kernel_id, kernel_id, stride_h, stride_w, cb_src0, cb_dst0}};
}

Fold::MultiCore::cached_program_t Fold::MultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    return fold_multi_core(
        tensor_args.input_tensor,
        output_tensor,
        operation_attributes.stride_h,
        operation_attributes.stride_w,
        operation_attributes.pad_top,
        operation_attributes.pad_bottom,
        operation_attributes.pad_left,
        operation_attributes.pad_right);
}

void Fold::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& cb_src0 = cached_program.shared_variables.cb_src0;
    auto& cb_dst0 = cached_program.shared_variables.cb_dst0;

    auto& program = cached_program.program;
    const auto& input_tensor = tensor_args.input_tensor;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cb_dst0, *dst_buffer);
}

}  // namespace ttnn::operations::data_movement
