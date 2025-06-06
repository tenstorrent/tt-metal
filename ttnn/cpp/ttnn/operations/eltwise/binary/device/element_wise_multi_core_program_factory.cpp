// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "binary_device_operation.hpp"
#include "cpp/ttnn/operations/eltwise/binary/device/eltwise_multi_core_program_factory_common.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

#include <tt-metalium/work_split.hpp>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::binary {

BinaryDeviceOperation::ElementWiseMultiCore::cached_program_t BinaryDeviceOperation::ElementWiseMultiCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using ttnn::operations::unary::UnaryWithParam;
    using namespace tt::constants;

    const auto& a = tensor_args.input_tensor_a;
    const auto& b = tensor_args.input_tensor_b;
    auto& output = tensor_return_value;
    const auto& op_type = operation_attributes.binary_op_type;

    std::vector<UnaryWithParam> fused_activations =
        operation_attributes.activations.value_or(std::vector<UnaryWithParam>{});

    Program program{};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b->dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
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
        block_or_width_sharded = a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (src1_sharded) {
        shard_spec = b->shard_spec().value();
        block_or_width_sharded = b->memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (out_sharded) {
        shard_spec = output.shard_spec().value();
        block_or_width_sharded = output.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED;
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
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    if (src0_sharded) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*a.buffer());
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    num_input_tiles = src1_sharded ? num_tiles_per_shard : 2 * max_block_size;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    if (src1_sharded) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*b->buffer());
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    std::map<string, string> eltwise_defines = utils::get_defines(
        op_type, a.dtype(), output.dtype(), fused_activations, operation_attributes.input_tensor_a_activation);

    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN0_0") != eltwise_defines.end()) {
        if (op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP ||
            op_type == BinaryOpType::LOGADDEXP2) {
            interim_cb0_format = tt::DataFormat::Float16_b;
        }
        uint32_t interim0_single_tile_size = tt_metal::detail::TileSize(interim_cb0_format);
        tt_metal::CircularBufferConfig cb_interm_config =
            tt_metal::CircularBufferConfig(
                max_block_size * interim0_single_tile_size, {{tt::CBIndex::c_3, interim_cb0_format}})
                .set_page_size(tt::CBIndex::c_3, interim0_single_tile_size);
        auto cb_interm = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm_config);
    }
    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN1_0") != eltwise_defines.end()) {
        if (op_type == BinaryOpType::LOGADDEXP || op_type == BinaryOpType::LDEXP ||
            op_type == BinaryOpType::LOGADDEXP2) {
            interim_cb1_format = tt::DataFormat::Float16_b;
        }
        uint32_t interim1_single_tile_size = tt_metal::detail::TileSize(interim_cb1_format);
        tt_metal::CircularBufferConfig cb_interm2_config =
            tt_metal::CircularBufferConfig(
                max_block_size * interim1_single_tile_size, {{tt::CBIndex::c_4, interim_cb1_format}})
                .set_page_size(tt::CBIndex::c_4, interim1_single_tile_size);
        auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm2_config);
    }

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = (out_sharded || block_or_width_sharded) ? num_tiles_per_shard : 2 * max_block_size;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    if (out_sharded) {
        cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    std::map<string, string> reader_defines;
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (src1_sharded) {
        reader_defines["IN1_SHARDED"] = "1";
    }
    std::map<string, string> writer_defines;
    if (out_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram, (std::uint32_t)block_or_width_sharded};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        (block_or_width_sharded and not out_sharded)
            ? "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
              "writer_unary_sharded_blocks_interleaved_start_id.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    bool fp32_dest_acc_en = dst_cb_data_format == tt::DataFormat::UInt32 ||
                            dst_cb_data_format == tt::DataFormat::Int32 ||
                            dst_cb_data_format == tt::DataFormat::Float32;
    auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .defines = eltwise_defines});

    set_eltwise_binary_runtime_args<true>(
        program,
        a,
        *b,
        output,
        binary_reader_kernel_id,
        unary_writer_kernel_id,
        eltwise_binary_kernel_id,
        cb_src0,
        cb_src1,
        cb_output,
        all_device_cores,
        src0_single_tile_size,
        src1_single_tile_size,
        dst_single_tile_size);

    return {
        std::move(program),
        {binary_reader_kernel_id,
         unary_writer_kernel_id,
         eltwise_binary_kernel_id,
         cb_src0,
         cb_src1,
         cb_output,
         all_device_cores,
         src0_single_tile_size,
         src1_single_tile_size,
         dst_single_tile_size}};
}

void BinaryDeviceOperation::ElementWiseMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    const auto& shared_variables = cached_program.shared_variables;

    set_eltwise_binary_runtime_args<false>(
        cached_program.program,
        input_tensor_a,
        *input_tensor_b,
        output_tensor,
        shared_variables.binary_reader_kernel_id,
        shared_variables.unary_writer_kernel_id,
        shared_variables.eltwise_binary_kernel_id,
        shared_variables.cb_src0,
        shared_variables.cb_src1,
        shared_variables.cb_output,
        shared_variables.all_device_cores,
        shared_variables.src0_single_tile_size,
        shared_variables.src1_single_tile_size,
        shared_variables.dst_single_tile_size);
}
}  // namespace ttnn::operations::binary
