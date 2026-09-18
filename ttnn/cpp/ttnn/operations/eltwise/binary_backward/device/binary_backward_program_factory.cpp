// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_backward_program_factory.hpp"

#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "binary_backward_op_utils.hpp"

namespace ttnn::operations::binary_backward {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor BinaryBackwardProgramFactory::create_descriptor(
    const BinaryBackwardParams& args, const BinaryBackwardInputs& tensor_args, std::vector<Tensor>& outputs) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;
    auto& input_grad = outputs.at(0);
    auto& other_grad = outputs.at(1);

    const auto& spec = kernel_spec(args.op_type);

    // Pair each CB's format with the buffer bound to it, and pick fp32_dest_acc
    // from the spec flag OR from the actual destination dtype. This is the
    // symmetric fix for the two tanh_bw bugs called out in PR #56061.
    const auto grad_out_df = datatype_to_dataformat_converter(grad_output.dtype());
    const auto input_df = datatype_to_dataformat_converter(input.dtype());
    const auto other_df = datatype_to_dataformat_converter(other.dtype());
    const auto input_grad_df = datatype_to_dataformat_converter(input_grad.dtype());
    const auto other_grad_df = datatype_to_dataformat_converter(other_grad.dtype());

    const uint32_t grad_out_tile_size = tt::tile_size(grad_out_df);
    const uint32_t input_tile_size = tt::tile_size(input_df);
    const uint32_t other_tile_size = tt::tile_size(other_df);
    const uint32_t input_grad_tile_size = tt::tile_size(input_grad_df);
    const uint32_t other_grad_tile_size = tt::tile_size(other_grad_df);

    const auto dest_dtype_is_wide = [](tt::DataFormat df) {
        return df == tt::DataFormat::Float32 || df == tt::DataFormat::Int32 || df == tt::DataFormat::UInt32;
    };
    const bool fp32_dest_acc_en =
        spec.force_fp32_dest_acc || dest_dtype_is_wide(input_grad_df) || dest_dtype_is_wide(other_grad_df);

    IDevice* device = input.device();
    const uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = grid.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_g1, tiles_g2] = split_work_to_cores(grid, num_tiles);

    // CB layout: c_0 = grad_output, c_1 = input, c_2 = other, c_3 = input_grad, c_4 = other_grad
    constexpr uint32_t cb_grad_out = tt::CBIndex::c_0;
    constexpr uint32_t cb_input = tt::CBIndex::c_1;
    constexpr uint32_t cb_other = tt::CBIndex::c_2;
    constexpr uint32_t cb_input_grad = tt::CBIndex::c_3;
    constexpr uint32_t cb_other_grad = tt::CBIndex::c_4;
    constexpr uint32_t double_buffered = 2;

    auto* grad_out_buf = grad_output.buffer();
    auto* input_buf = input.buffer();
    auto* other_buf = other.buffer();
    auto* input_grad_buf = input_grad.buffer();
    auto* other_grad_buf = other_grad.buffer();

    ProgramDescriptor desc;

    const auto push_cb = [&](uint8_t cb_id, tt::DataFormat df, uint32_t tile_size) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = double_buffered * tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = cb_id,
                .data_format = df,
                .page_size = tile_size,
            }}},
        });
    };
    push_cb(cb_grad_out, grad_out_df, grad_out_tile_size);
    push_cb(cb_input, input_df, input_tile_size);
    push_cb(cb_other, other_df, other_tile_size);
    push_cb(cb_input_grad, input_grad_df, input_grad_tile_size);
    push_cb(cb_other_grad, other_grad_df, other_grad_tile_size);

    // Reader compile-time args: 3 CB IDs + 3 TensorAccessorArgs (order matches
    // ternary_reader_nobcast_ttt.cpp's TensorAccessorArgs<3,0>() indices).
    std::vector<uint32_t> reader_compile_time_args = {cb_grad_out, cb_input, cb_other};
    TensorAccessorArgs(*grad_out_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(*input_buf).append_to(reader_compile_time_args);
    TensorAccessorArgs(*other_buf).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {cb_input_grad, cb_other_grad};
    TensorAccessorArgs(*input_grad_buf).append_to(writer_compile_time_args);
    TensorAccessorArgs(*other_grad_buf).append_to(writer_compile_time_args);

    // UnpackToDestFp32 is only meaningful when DEST is actually accumulating in
    // float32 -- otherwise the request is a no-op that risks tripping asserts
    // (the other pre-existing tanh_bw bug called out in PR #56061).
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[cb_grad_out] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_input] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cb_other] = UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/dataflow/ternary_reader_nobcast_ttt.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary_backward/device/kernels/dataflow/"
        "writer_binary_backward_dual_nobcast.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(spec.compute_kernel_path);
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = spec.math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };

    for (uint32_t i = 0, tiles_done = 0; i < num_cores; ++i) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t tiles_here = 0;
        if (core_group_1.contains(core)) {
            tiles_here = tiles_g1;
        } else if (core_group_2.contains(core)) {
            tiles_here = tiles_g2;
        } else {
            TT_THROW("binary_backward factory: core {},{} outside computed groups", core.x, core.y);
        }

        reader_desc.emplace_runtime_args(core, {grad_out_buf, input_buf, other_buf, tiles_here, tiles_done});
        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{tiles_here});
        writer_desc.emplace_runtime_args(core, {input_grad_buf, other_grad_buf, tiles_here, tiles_done});

        tiles_done += tiles_here;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::binary_backward
