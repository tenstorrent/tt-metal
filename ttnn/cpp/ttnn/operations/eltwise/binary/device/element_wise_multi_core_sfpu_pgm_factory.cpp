// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "binary_device_operation.hpp"
#include "cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::binary {

tt::tt_metal::ProgramDescriptor BinaryDeviceOperation::ElementWiseMultiCoreSfpu::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using ttnn::operations::unary::UnaryWithParam;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto a_dtype = a.get_dtype();
    auto b_dtype = b.has_value() ? b->get_dtype() : a_dtype;
    auto& output = tensor_return_value;
    const auto& op_type = operation_attributes.binary_op_type;

    std::vector<UnaryWithParam> fused_activations =
        operation_attributes.activations.value_or(std::vector<UnaryWithParam>{});

    ProgramDescriptor program;

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a_dtype);
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b_dtype);
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt::DataFormat interim_cb0_format = src0_cb_data_format;
    tt::DataFormat interim_cb1_format = src1_cb_data_format;

    tt_metal::Buffer* src0_buffer = a.buffer();
    tt_metal::Buffer* src1_buffer = b->buffer();

    tt_metal::IDevice* device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool src1_sharded = b->memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    bool block_or_width_sharded = false;

    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
        block_or_width_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (src1_sharded) {
        shard_spec = b->shard_spec().value();
        block_or_width_sharded = b->memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (out_sharded) {
        shard_spec = output.shard_spec().value();
        block_or_width_sharded = output.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    }

    uint32_t max_block_size = 1, num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        max_block_size = find_max_block_size(num_tiles_per_shard);
    }

    tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const auto& all_device_cores = operation_attributes.worker_grid;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = src0_sharded ? num_tiles_per_shard : 2 * max_block_size;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_device_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }},
        .buffer = src0_sharded ? a.buffer() : nullptr,
    });

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    num_input_tiles = src1_sharded ? num_tiles_per_shard : 2 * max_block_size;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src1_single_tile_size,
        .core_ranges = all_device_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = src1_cb_data_format,
            .page_size = src1_single_tile_size,
        }},
        .buffer = src1_sharded ? b->buffer() : nullptr,
    });

    tt::tt_metal::KernelDescriptor::Defines eltwise_defines;
    utils::append_defines(
        eltwise_defines, op_type, a_dtype, b_dtype, fused_activations, operation_attributes.input_tensor_a_activation);

    uint32_t src0interim_cb_index = tt::CBIndex::c_3;
    if (std::find_if(eltwise_defines.begin(), eltwise_defines.end(), [](const auto& define) {
            return define.first == "SFPU_OP_INIT_PRE_IN0_0";
        }) != eltwise_defines.end()) {
        uint32_t interim0_single_tile_size = tt_metal::detail::TileSize(interim_cb0_format);
        program.cbs.push_back(CBDescriptor{
            .total_size = max_block_size * interim0_single_tile_size,
            .core_ranges = all_device_cores.ranges(),
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = src0interim_cb_index,
                .data_format = interim_cb0_format,
                .page_size = interim0_single_tile_size,
            }},
        });
    }
    uint32_t src1interim_cb_index = tt::CBIndex::c_4;
    if (std::find_if(eltwise_defines.begin(), eltwise_defines.end(), [](const auto& define) {
            return define.first == "SFPU_OP_INIT_PRE_IN1_0";
        }) != eltwise_defines.end()) {
        uint32_t interim1_single_tile_size = tt_metal::detail::TileSize(interim_cb1_format);
        program.cbs.push_back(CBDescriptor{
            .total_size = max_block_size * interim1_single_tile_size,
            .core_ranges = all_device_cores.ranges(),
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = src1interim_cb_index,
                .data_format = interim_cb1_format,
                .page_size = interim1_single_tile_size,
            }},
        });
    }

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = (out_sharded || block_or_width_sharded) ? num_tiles_per_shard : 2 * max_block_size;
    program.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_device_cores.ranges(),
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }},
        .buffer = out_sharded ? output.buffer() : nullptr,
    });

    std::map<string, string> writer_defines;
    if (out_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    constexpr size_t num_kernels = 3;
    program.kernels.resize(num_kernels);

    auto& binary_reader_kernel = program.kernels[0];
    binary_reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp";
    binary_reader_kernel.core_ranges = all_device_cores.ranges();
    binary_reader_kernel.compile_time_args = {
        (std::uint32_t)src0_is_dram,
        (std::uint32_t)src1_is_dram,
        (std::uint32_t)block_or_width_sharded,
    };
    binary_reader_kernel.config = tt_metal::ReaderConfigDescriptor{};
    if (src0_sharded) {
        binary_reader_kernel.defines.emplace_back("IN0_SHARDED", "1");
    }
    if (src1_sharded) {
        binary_reader_kernel.defines.emplace_back("IN1_SHARDED", "1");
    }
    binary_reader_kernel.reserve_runtime_args();

    auto& unary_writer_kernel = program.kernels[1];
    unary_writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    unary_writer_kernel.core_ranges = all_device_cores.ranges();
    unary_writer_kernel.compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};
    unary_writer_kernel.config = tt_metal::WriterConfigDescriptor{};
    if (out_sharded) {
        unary_writer_kernel.defines.emplace_back("OUT_SHARDED", "1");
    }
    unary_writer_kernel.reserve_runtime_args();

    bool fp32_dest_acc_en = (dst_cb_data_format == tt::DataFormat::Float32) ||
                            (dst_cb_data_format == tt::DataFormat::Int32) ||
                            (dst_cb_data_format == tt::DataFormat::UInt32);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (op_type != BinaryOpType::POWER) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src0interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[src1interim_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    } else {
        unpack_to_dest_mode[src0_cb_index] =
            (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
        unpack_to_dest_mode[src1_cb_index] =
            (b_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
        unpack_to_dest_mode[src0interim_cb_index] =
            (a_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
        unpack_to_dest_mode[src1interim_cb_index] =
            (b_dtype == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;
    }

    auto& eltwise_binary_kernel = program.kernels[2];
    eltwise_binary_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel.cpp";
    eltwise_binary_kernel.core_ranges = all_device_cores.ranges();
    eltwise_binary_kernel.defines = std::move(eltwise_defines);
    eltwise_binary_kernel.config = tt_metal::ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en, .unpack_to_dest_mode = unpack_to_dest_mode};
    eltwise_binary_kernel.reserve_runtime_args();

    set_eltwise_binary_runtime_args(
        a,
        *b,
        output,
        binary_reader_kernel,
        unary_writer_kernel,
        eltwise_binary_kernel,
        all_device_cores,
        src0_single_tile_size,
        src1_single_tile_size,
        dst_single_tile_size);

    return program;
}

}  // namespace ttnn::operations::binary
