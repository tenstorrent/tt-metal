// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation/device/accumulation_device_operation_types.hpp"
#include "accumulation_device_operation.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <map>
#include <string>

namespace ttnn::prim {

using AccumulationProgramFactory = AccumulationDeviceOperation::AccumulationProgramFactory;

// calculate the offset between consecutive tiles between accumulation axis and last dimension
uint32_t AccumulationProgramFactory::calc_input_tile_offset(
    const Shape& input_shape, const int32_t& dim, uint32_t tile_height, uint32_t tile_width) {
    uint32_t input_tile_offset{1};
    for (int32_t i = dim + 1; i < input_shape.rank() - 2; ++i) {
        input_tile_offset *= input_shape[i];
    }
    if (input_shape.rank() > 1) {
        input_tile_offset *= (input_shape[-2] / tile_height);
    }
    if (input_shape.rank() > 0) {
        input_tile_offset *= (input_shape[-1] / tile_width);
    }

    return input_tile_offset;
}

tt::tt_metal::ProgramDescriptor AccumulationProgramFactory::create_descriptor(
    const AccumulationParams& operation_attributes,
    const AccumulationInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    ProgramDescriptor desc;

    IDevice* device{input_tensor.device()};

    auto* src_buffer{input_tensor.buffer()};
    auto* dst_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(output_tensor.dtype())};

    const uint32_t input_rank{input_tensor.padded_shape().rank()};

    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;
    TT_FATAL(num_cores_y != 0, "Compute grid y-dimension must be non-zero");

    const int32_t dim{
        (operation_attributes.dim >= 0) ? operation_attributes.dim : (input_rank + operation_attributes.dim)};

    const auto& tile = input_tensor.tensor_spec().tile();
    // how many tiles along accumulation axis
    const uint32_t tiles_per_row{input_tensor.padded_shape()[dim]};
    TT_FATAL(tiles_per_row != 0, "tiles_per_row must be non-zero (got 0 for dim={})", dim);
    // all work units (product of all row lengths besides the accumulation row)
    const uint32_t num_rows_total{input_tensor.physical_volume() / tile.get_tile_hw() / tiles_per_row};
    // tiles between consecutive tiles along accumulation row
    const uint32_t input_tile_offset{calc_input_tile_offset(input_shape, dim, tile.get_height(), tile.get_width())};
    TT_FATAL(input_tile_offset != 0, "input_tile_offset must be non-zero (got 0 for dim={})", dim);

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_rows_total);

    constexpr uint32_t in_tiles = 4;
    constexpr uint32_t acc_tiles = 1;
    constexpr uint32_t out_tiles = 4;

    auto acc_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());
    if (!is_integer_format(acc_dataformat)) {
        acc_dataformat = DataFormat::Float32;
    }
    auto acc_dataformat_name = fmt::format("DataFormat::{}", acc_dataformat);

    const auto input_dataformat = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto output_dataformat = datatype_to_dataformat_converter(output_tensor.dtype());

    auto push_cb = [&](const tt::DataFormat& data_format,
                       AccumulationProgramFactory::AccumulationCB accumulation_cb,
                       uint32_t num_tiles) {
        const uint32_t cb_id{static_cast<uint32_t>(accumulation_cb)};
        const uint32_t single_tile_size{tt::tile_size(data_format)};
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id),
                .data_format = data_format,
                .page_size = single_tile_size,
            }}},
        });
    };

    push_cb(input_dataformat, AccumulationProgramFactory::AccumulationCB::SRC, in_tiles);
    push_cb(acc_dataformat, AccumulationProgramFactory::AccumulationCB::ACC, acc_tiles);
    push_cb(output_dataformat, AccumulationProgramFactory::AccumulationCB::DST, out_tiles);

    std::vector<UnpackToDestMode> unpack_to_dst(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dst[static_cast<unsigned>(AccumulationProgramFactory::AccumulationCB::ACC)] =
        UnpackToDestMode::UnpackToDestFp32;

    if (input_dataformat != DataFormat::Float16_b) {
        unpack_to_dst[static_cast<unsigned>(AccumulationProgramFactory::AccumulationCB::SRC)] =
            UnpackToDestMode::UnpackToDestFp32;
    }

    std::map<std::string, std::string> defines_kernel_args = {};

    if (is_integer_format(dst_cb_data_format)) {
        defines_kernel_args["BINARY_OP_INIT"] = operation_attributes.op == AccumulationOp::CUMSUM
                                                    ? "add_int_tile_init"
                                                    : fmt::format("mul_int_tile_init<{}>", acc_dataformat_name);
        defines_kernel_args["BINARY_OP"] = operation_attributes.op == AccumulationOp::CUMSUM
                                               ? fmt::format("add_int_tile<{}>", acc_dataformat_name)
                                               : fmt::format("mul_int_tile<{}>", acc_dataformat_name);
        defines_kernel_args["FILL_TILE"] = fmt::format("fill_tile_int<{}>", acc_dataformat_name);
    } else {
        defines_kernel_args["BINARY_OP_INIT"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile_init" : "mul_binary_tile_init";
        defines_kernel_args["BINARY_OP"] =
            operation_attributes.op == AccumulationOp::CUMSUM ? "add_binary_tile" : "mul_binary_tile";
        defines_kernel_args["FILL_TILE"] = "fill_tile_bitcast";
    }

    float default_acc_value = 0.f;
    if (operation_attributes.op == AccumulationOp::CUMPROD) {
        default_acc_value = 1.f;
        if (is_integer_format(dst_cb_data_format)) {
            // Kernel reinterprets the 4-byte CT arg as int32 in the integer path; pack the bit
            // pattern 0x00000001 so it lands as integer 1, not float 1.0f's bit pattern.
            default_acc_value = std::bit_cast<float>(1U);
        }
    }

    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // fp32_dest_acc_en will be True for FLOAT32 inputs (set below), so use HiFi3 as default on Wormhole B0.
    const auto is_wormhole = device->arch() == tt::ARCH::WORMHOLE_B0;
    const auto default_math_fidelity =
        (is_wormhole && output_tensor.dtype() == DataType::FLOAT32) ? MathFidelity::HiFi3 : MathFidelity::HiFi4;

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = AccumulationProgramFactory::KERNEL_PATHS[0];
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = AccumulationProgramFactory::KERNEL_PATHS[2];
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    KernelDescriptor::Defines compute_defines(defines_kernel_args.begin(), defines_kernel_args.end());

    KernelDescriptor compute_desc_1;
    compute_desc_1.kernel_source = AccumulationProgramFactory::KERNEL_PATHS[1];
    compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_1.core_ranges = core_group_1;
    compute_desc_1.compile_time_args = {std::bit_cast<uint32_t>(default_acc_value)};
    compute_desc_1.defines = compute_defines;
    compute_desc_1.config = ComputeConfigDescriptor{
        .math_fidelity = default_math_fidelity,
        .fp32_dest_acc_en = true,
        .dst_full_sync_en = false,
        .unpack_to_dest_mode = unpack_to_dst,
        .math_approx_mode = false,
    };

    std::optional<KernelDescriptor> compute_desc_2;
    if (!core_group_2.ranges().empty()) {
        KernelDescriptor cd2;
        cd2.kernel_source = AccumulationProgramFactory::KERNEL_PATHS[1];
        cd2.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd2.core_ranges = core_group_2;
        cd2.compile_time_args = {std::bit_cast<uint32_t>(default_acc_value)};
        cd2.defines = compute_defines;
        cd2.config = ComputeConfigDescriptor{
            .math_fidelity = default_math_fidelity,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = false,
            .unpack_to_dest_mode = unpack_to_dst,
            .math_approx_mode = false,
        };
        compute_desc_2 = std::move(cd2);
    }

    for (uint32_t i{0}, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core{i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        reader_desc.emplace_runtime_args(
            core,
            {src_buffer,
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset,
             static_cast<uint32_t>(operation_attributes.flip)});

        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,
             num_tiles_per_core,
             tiles_per_row,
             input_tile_offset,
             tile_offset,
             tile_offset / input_tile_offset,
             tile_offset % input_tile_offset,
             static_cast<uint32_t>(operation_attributes.flip)});

        if (core_group_1.contains(core)) {
            compute_desc_1.emplace_runtime_args(core, {num_tiles_per_core, tiles_per_row});
        } else if (core_group_2.contains(core)) {
            TT_ASSERT(compute_desc_2.has_value());
            compute_desc_2->emplace_runtime_args(core, {num_tiles_per_core, tiles_per_row});
        } else {
            TT_THROW("Core not in any predefined core range.");
        }

        tile_offset += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_1));
    if (compute_desc_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_2));
    }

    return desc;
}

}  // namespace ttnn::prim
