// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/move/device/move_device_operation.hpp"

#include <cmath>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor MoveShardedProgramFactory::create_descriptor(
    const MoveOperationAttributes& /*operation_attributes*/,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto shard_spec = input.shard_spec().value();
    const auto shard_shape = shard_spec.shape;
    const auto shard_grid = shard_spec.grid;
    const auto& input_shape = input.logical_shape();
    const DataType input_dtype = input.dtype();
    const Layout input_layout = input.layout();
    TT_FATAL(
        input_layout == output.layout() && input_dtype == output.dtype() &&
            shard_shape == output.shard_spec().value().shape && input_shape == output.logical_shape(),
        "Error");
    const uint32_t src_cb_sharded = tt::CBIndex::c_0;
    const uint32_t dst_cb_sharded = tt::CBIndex::c_1;

    const uint32_t total_size_bytes = input.buffer()->aligned_size_per_bank();
    const uint32_t page_size_bytes = input.buffer()->aligned_page_size();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const uint32_t input_buffer_address = src_buffer->address();
    const uint32_t output_buffer_address = dst_buffer->address();

    const uint32_t move_chunk_size_bytes = output_buffer_address - input_buffer_address;
    TT_FATAL(
        src_buffer->alignment() == dst_buffer->alignment(),
        "Expected input buffer alignment ({} B) and output buffer alignment ({} B) to be equal",
        src_buffer->alignment(),
        dst_buffer->alignment());
    TT_FATAL(
        move_chunk_size_bytes % src_buffer->alignment() == 0,
        "Expected chunk size bytes to move to be {} byte aligned.",
        src_buffer->alignment());
    const uint32_t num_chunks = total_size_bytes / move_chunk_size_bytes;
    const uint32_t remainder_chunk_size_bytes = total_size_bytes % move_chunk_size_bytes;

    ProgramDescriptor desc;

    // Sharded src CB: dynamic globally-allocated; framework rebinds on cache hit.
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_size_bytes,
        .core_ranges = shard_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src_cb_sharded),
            .data_format = cb_data_format,
            .page_size = page_size_bytes,
        }}},
        .buffer = src_buffer,
    });

    // Sharded dst CB: dynamic globally-allocated; framework rebinds on cache hit.
    desc.cbs.push_back(CBDescriptor{
        .total_size = total_size_bytes,
        .core_ranges = shard_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dst_cb_sharded),
            .data_format = cb_data_format,
            .page_size = page_size_bytes,
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> reader_compile_time_args = {src_cb_sharded, dst_cb_sharded};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = shard_grid;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = DataMovementConfigDescriptor{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1};

    // Reader args derive from the address arithmetic (output_addr - input_addr) and are recomputed
    // every dispatch by override_runtime_arguments re-running create_descriptor (#48928). CB
    // addresses are re-patched via desc.cbs[*].buffer in apply_descriptor_runtime_args().
    const auto cores = corerange_to_cores(shard_grid, std::nullopt, true);
    for (const auto& core : cores) {
        reader_desc.emplace_runtime_args(
            core,
            {total_size_bytes,
             num_chunks,
             move_chunk_size_bytes,  // re-applied on cache hit via override_runtime_arguments (#48928)
             remainder_chunk_size_bytes});
    }

    desc.kernels.push_back(std::move(reader_desc));

    return desc;
}

// Re-derive ALL per-dispatch state from the same create_descriptor the miss path uses (mirror
// select_program_factory) and re-apply to the cached program. Supersedes get_dynamic + resolve_bindings.
void MoveDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramDescriptor desc;
    switch (operation_attributes.move_op_parallelization_strategy) {
        case MoveOpParallelizationStrategy::MULTI_CORE_SHARDED:
            desc = MoveShardedProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
            break;
        case MoveOpParallelizationStrategy::MULTI_CORE_OVERLAP:
            desc = MoveOverlapProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
            break;
        case MoveOpParallelizationStrategy::MULTI_CORE:
            desc = MoveProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
            break;
        default: TT_FATAL(false, "Invalid move operation parallelization strategy");
    }
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}

}  // namespace ttnn::prim
