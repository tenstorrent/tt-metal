// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_backward_program_factory.hpp"

#include <map>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "unary_backward_device_operation_types.hpp"
#include "unary_backward_op_utils.hpp"

namespace ttnn::operations::unary_backward {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor UnaryBackwardProgramFactory::create_descriptor(
    const UnaryBackwardParams& args, const UnaryBackwardInputs& tensor_args, Tensor& output) {
    const auto& grad_output = tensor_args.grad_output;
    const auto& input = tensor_args.input;

    const UnaryBackwardKernelSpec& spec = get_kernel_spec(args.op_type);

    // c_0 carries grad_output and c_1 carries input, in both the CB formats and the buffers
    // bound to them. Kernels index the operands by these buffer indices, so the pairing is
    // part of the contract in UnaryBackwardKernelSpec, not a detail of one op.
    const DataFormat grad_output_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const uint32_t grad_output_single_tile_size = tile_size(grad_output_cb_data_format);
    const DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tile_size(input_cb_data_format);
    const DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tile_size(output_cb_data_format);

    const uint32_t num_tiles = input.physical_volume() / TILE_HW;

    IDevice* device = input.device();
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    constexpr uint32_t grad_output_cb_index = CBIndex::c_0;
    constexpr uint32_t input_cb_index = CBIndex::c_1;
    constexpr uint32_t grad_input_cb_index = CBIndex::c_2;

    ProgramDescriptor desc;

    // ---- Circular buffers ----

    const auto push_cb = [&desc, &all_cores](
                             uint32_t index, DataFormat data_format, uint32_t tile_size, uint32_t tiles) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = tiles * tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(index),
                .data_format = data_format,
                .page_size = tile_size,
            }}},
        });
    };

    push_cb(grad_output_cb_index, grad_output_cb_data_format, grad_output_single_tile_size, num_input_tiles);
    push_cb(input_cb_index, input_cb_data_format, input_single_tile_size, num_input_tiles);
    push_cb(grad_input_cb_index, output_cb_data_format, output_single_tile_size, num_output_tiles);

    auto* grad_output_buffer = grad_output.buffer();
    auto* input_buffer = input.buffer();
    auto* grad_input_buffer = output.buffer();

    // ---- Reader / writer kernels ----
    //
    // Both are the generic interleaved dataflow kernels already used by the forward ops: a
    // unary gradient reads two same-shaped operands and writes one, which is exactly the
    // binary reader's and the unary writer's contract.

    std::vector<uint32_t> reader_compile_time_args = {0};
    TensorAccessorArgs(*grad_output_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*input_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {grad_input_cb_index};
    TensorAccessorArgs(*grad_input_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Compute kernel ----
    //
    // A gradient chains several SFPU steps in DEST, so a float32 operand has to stay float32
    // all the way through: accumulate in fp32 if any operand or the output is fp32, and only
    // then ask the unpacker for fp32 DEST values, per operand that actually is fp32.
    const bool fp32_dest_acc_en = spec.force_fp32_dest_acc || (grad_output_cb_data_format == DataFormat::Float32) ||
                                  (input_cb_data_format == DataFormat::Float32) ||
                                  (output_cb_data_format == DataFormat::Float32);

    // Ask the unpacker for float32 DEST values only where they can actually be held: when DEST
    // is accumulating in float32. Requesting it otherwise would widen values into a DEST that
    // cannot represent them.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[grad_output_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    // Supplied to every kernel whether or not it copies between DEST slots or packs bfloat16:
    // both are properties of the program the factory built, so a kernel that needs them must
    // not have to re-derive them, and one that does not simply leaves the define unused.
    std::map<std::string, std::string> compute_defines;
    compute_defines["COPY_DEST_DATA_FORMAT"] = fp32_dest_acc_en ? "DataFormat::Float32" : "DataFormat::Float16_b";
    compute_defines["BF16_ROUNDING_MODE"] = (output.dtype() == DataType::BFLOAT16)
                                                ? "ckernel::DstRoundingMode::NearestEven"
                                                : "ckernel::DstRoundingMode::Default";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(spec.compute_kernel_path);
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = spec.math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = {unpack_to_dest_mode.begin(), unpack_to_dest_mode.end()},
    };
    compute_desc.defines = {compute_defines.begin(), compute_defines.end()};

    // ---- Per-core runtime args ----

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(
            core, {grad_output_buffer, input_buffer, num_tiles_per_core, num_tiles_written, 0u, 0u, num_cores_y});
        compute_desc.emplace_runtime_args(core, {num_tiles_per_core});
        writer_desc.emplace_runtime_args(core, {grad_input_buffer, num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::unary_backward
