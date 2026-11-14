// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/concat_heads_decode/device/concat_heads_decode_device_operation.hpp"

#include <filesystem>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/llama31_8b/concat_heads_decode/device/concat_heads_decode_plan.hpp"

namespace ttnn::prim::llama31_8b_concat_heads_decode {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr const char* KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/llama31_8b/concat_heads_decode/device/kernels/gather.cpp";

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

ConcatHeadsDecodeDeviceOperation::program_factory_t ConcatHeadsDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ConcatHeadsDecodeProgramFactory{};
}

void ConcatHeadsDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    const auto grid = in.device()->compute_with_storage_grid_size();
    TT_FATAL(
        in.storage_type() == StorageType::DEVICE && in.layout() == Layout::TILE && in.dtype() == DataType::BFLOAT16,
        "concat_heads_decode: input must be a bfloat16 TILE device tensor");
    TT_FATAL(
        !in.is_sharded() && in.buffer()->buffer_type() == BufferType::DRAM,
        "concat_heads_decode: input must be DRAM interleaved");
    TT_FATAL(
        operation_attributes.num_heads == plan::kHeads && in.padded_shape()[0] == 1 && in.padded_shape()[1] == 1 &&
            in.padded_shape()[2] == plan::kHeads && in.padded_shape()[3] == plan::kHeadDim,
        "concat_heads_decode: input must be [1, 1, {}, {}]",
        plan::kHeads,
        plan::kHeadDim);
    TT_FATAL(
        in.device()->arch() == tt::ARCH::BLACKHOLE && grid.x == plan::kGridX && grid.y == plan::kGridY,
        "concat_heads_decode: built for the Blackhole {}x{} grid",
        plan::kGridX,
        plan::kGridY);
}

ConcatHeadsDecodeDeviceOperation::spec_return_value_t ConcatHeadsDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    const uint32_t head_dim = in.padded_shape()[3];
    const auto cores = tt::tt_metal::num_cores_to_corerangeset(
        operation_attributes.num_heads, in.device()->compute_with_storage_grid_size(), true);
    const tt::tt_metal::ShardSpec shard_spec{cores, {tt::constants::TILE_HEIGHT, head_dim}};
    const tt::tt_metal::MemoryConfig mem_config{TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1, shard_spec};
    return tt::tt_metal::TensorSpec(
        Shape({in.padded_shape()[0], 1, tt::constants::TILE_HEIGHT, operation_attributes.num_heads * head_dim}),
        tt::tt_metal::TensorLayout(in.dtype(), tt::tt_metal::PageConfig(Layout::TILE), mem_config));
}

ConcatHeadsDecodeDeviceOperation::tensor_return_value_t ConcatHeadsDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttnn::device_operation::ProgramArtifacts ConcatHeadsDecodeProgramFactory::create_program_artifacts(
    const ConcatHeadsDecodeParams&, const ConcatHeadsDecodeInputs& tensor_args, Tensor& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    auto kernel = [&](const KernelSpecName& id, DataMovementHardwareConfig hw) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(KERNEL),
            .tensor_bindings =
                {{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                 {.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .compile_time_args = {{"row_bytes", plan::kRowBytes}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"n"}, .runtime_array_args = {{.name = "plan", .size = 2 * plan::kSlots}}},
            .hw_config = std::move(hw),
        };
    };
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    std::vector<CoreRange> ranges;
    for (const auto& c : plan::kCores) {
        const CoreCoord core{c.x, c.y};
        ranges.emplace_back(core, core);
        AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"n", plan::kSlots}});
        AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"n", plan::kSlots}});
        reader_args.runtime_array_arg_values["plan"][core] = std::vector<uint32_t>(c.reader.begin(), c.reader.end());
        writer_args.runtime_array_arg_values["plan"][core] = std::vector<uint32_t>(c.writer.begin(), c.writer.end());
    }
    const auto& out_grid = output.shard_spec().value().grid;
    bool same_cores = out_grid.num_cores() == ranges.size();
    for (const auto& r : ranges) {
        same_cores = same_cores && out_grid.contains(r.start_coord);
    }
    TT_FATAL(same_cores, "concat_heads_decode: plan cores differ from the output shard grid");
    ProgramSpec spec{
        .name = "llama31_8b_concat_heads_decode",
        .kernels =
            {kernel(READER, create_reader_datamovement_config()), kernel(WRITER, create_writer_datamovement_config())},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = tensor_args.input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = CoreRangeSet(ranges),
        }},
    };
    ProgramRunArgs run;
    run.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run.tensor_args = {{INPUT, tensor_args.input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim::llama31_8b_concat_heads_decode
