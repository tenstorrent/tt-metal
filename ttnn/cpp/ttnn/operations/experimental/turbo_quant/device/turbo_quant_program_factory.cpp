// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "turbo_quant_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstring>

namespace ttnn::operations::experimental::turbo_quant {

// Helper: reinterpret a float as uint32_t (bit-cast for kernel compile args).
static uint32_t float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

TurboQuantDeviceOperation::MultiCore::cached_program_t TurboQuantDeviceOperation::MultiCore::create(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tile_size(cb_data_format);
    uint32_t num_tiles = input_tensor.physical_volume() / tt::constants::TILE_HW;

    // ── Distribute tiles across all available cores ──
    IDevice* device = input_tensor.device();
    auto grid = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(grid, num_tiles);

    // ── Circular buffers: input (c_0) and output (c_2), 2 tiles each ──
    uint32_t num_cb_tiles = 2;
    CircularBufferConfig cb_in_cfg =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_in_cfg);

    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_out_cfg);

    // ── Dataflow kernels: generic unary reader / writer ──
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    std::vector<uint32_t> writer_ct_args = {static_cast<uint32_t>(CBIndex::c_2)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelHandle reader_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    KernelHandle writer_kernel = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Compute kernel: select bucketize or gather based on op_type ──
    std::string kernel_path;
    if (attrs.op_type == TurboQuantOpType::BUCKETIZE) {
        kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/turbo_quant/device/kernels/compute/"
            "turbo_quant_bucketize.cpp";
    } else {
        kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/turbo_quant/device/kernels/compute/"
            "turbo_quant_gather_centroids.cpp";
    }

    // Build compile-time args: [num_tiles, num_params, param_bits…]
    auto make_compute_args = [&](uint32_t tiles_per_core) -> std::vector<uint32_t> {
        std::vector<uint32_t> args;
        args.push_back(tiles_per_core);
        args.push_back(static_cast<uint32_t>(attrs.params.size()));
        for (float p : attrs.params) {
            args.push_back(float_to_bits(p));
        }
        return args;
    };

    CreateKernel(
        program,
        kernel_path,
        core_group_1,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = false,
            .compile_args = make_compute_args(tiles_per_core_1)});

    if (!core_group_2.ranges().empty()) {
        CreateKernel(
            program,
            kernel_path,
            core_group_2,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
                .compile_args = make_compute_args(tiles_per_core_2)});
    }

    // ── Runtime args for reader / writer ──
    for (uint32_t i = 0, offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t tiles_this_core = core_group_1.contains(core) ? tiles_per_core_1 : tiles_per_core_2;

        SetRuntimeArgs(program, reader_kernel, core, {src_buffer->address(), tiles_this_core, offset});
        SetRuntimeArgs(program, writer_kernel, core, {dst_buffer->address(), tiles_this_core, offset});
        offset += tiles_this_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel,
         .writer_kernel_id = writer_kernel,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void TurboQuantDeviceOperation::MultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& program = cached_program.program;
    auto& reader_kid = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kid = cached_program.shared_variables.writer_kernel_id;
    auto num_cores = cached_program.shared_variables.num_cores;
    auto num_cores_y = cached_program.shared_variables.num_cores_y;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt::tt_metal::GetRuntimeArgs(program, reader_kid, core)[0] = src_buffer->address();
        tt::tt_metal::GetRuntimeArgs(program, writer_kid, core)[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::turbo_quant
