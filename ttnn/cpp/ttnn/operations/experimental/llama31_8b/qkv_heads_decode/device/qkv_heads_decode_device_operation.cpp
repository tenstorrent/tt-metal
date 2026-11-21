// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/qkv_heads_decode/device/qkv_heads_decode_device_operation.hpp"

#include <filesystem>
#include <span>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/llama31_8b/qkv_heads_decode/device/qkv_heads_decode_plan.hpp"
#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.hpp"

namespace ttnn::prim::llama31_8b_qkv_heads_decode {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

using TTOp = ttnn::experimental::prim::NLPCreateQKVHeadsDecodeDeviceOperation;

constexpr const char* KERNEL =
    "ttnn/cpp/ttnn/operations/experimental/llama31_8b/qkv_heads_decode/device/kernels/push.cpp";

std::span<const plan::Core> select_plan(
    const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeParams& attrs,
    const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeInputs& args,
    const std::vector<Tensor>& output) {
    const auto& in = args.input_tensor;
    const auto grid = in.device()->compute_with_storage_grid_size();
    const bool llama_shape = attrs.num_q_heads == 32 && attrs.num_kv_heads == 8 && attrs.head_dim == 128 &&
                             in.dtype() == DataType::BFLOAT16 && in.padded_shape()[-2] == 32 &&
                             in.padded_shape()[-1] == 6144 && !args.batch_offset.has_value();
    if (!llama_shape || in.device()->arch() != tt::ARCH::BLACKHOLE || grid.x != plan::kGridX ||
        grid.y != plan::kGridY || in.buffer()->buffer_type() != BufferType::L1 ||
        output[0].shard_spec().value().grid.num_cores() != 1) {
        return {};
    }
    if (!in.is_sharded()) {
        return plan::kInterleaved;
    }
    const auto ss = in.shard_spec().value();
    if (in.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED && ss.shape[0] == 32 &&
        ss.shape[1] == 192 && ss.orientation == ShardOrientation::ROW_MAJOR &&
        ss.grid == CoreRangeSet(CoreRange({0, 0}, {7, 3}))) {
        return plan::kWidthSharded;
    }
    if (in.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED && ss.shape[0] == 32 &&
        ss.shape[1] == 192 && ss.orientation == ShardOrientation::ROW_MAJOR &&
        ss.grid == CoreRangeSet(std::vector<CoreRange>{CoreRange({0, 0}, {10, 1}), CoreRange({0, 2}, {9, 2})})) {
        return plan::kWidthSharded11;
    }
    return {};
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

using namespace CMAKE_UNIQUE_NAMESPACE;

QkvHeadsDecodeDeviceOperation::program_factory_t QkvHeadsDecodeDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QkvHeadsDecodeProgramFactory{};
}

void QkvHeadsDecodeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TTOp::validate_on_program_cache_miss(operation_attributes, tensor_args);
}

QkvHeadsDecodeDeviceOperation::spec_return_value_t QkvHeadsDecodeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TTOp::compute_output_specs(operation_attributes, tensor_args);
}

QkvHeadsDecodeDeviceOperation::tensor_return_value_t QkvHeadsDecodeDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return TTOp::create_output_tensors(operation_attributes, tensor_args);
}

ttnn::device_operation::ProgramArtifacts QkvHeadsDecodeProgramFactory::create_program_artifacts(
    const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeParams& operation_attributes,
    const ttnn::experimental::prim::NlpCreateQkvHeadsDecodeInputs& tensor_args,
    std::vector<Tensor>& output) {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto cores = select_plan(operation_attributes, tensor_args, output);
    if (cores.empty()) {
        return std::visit(
            [&](auto factory) {
                return decltype(factory)::create_program_artifacts(operation_attributes, tensor_args, output);
            },
            TTOp::select_program_factory(operation_attributes, tensor_args));
    }

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const TensorParamName QKV_IN{"qkv_in"};
    const TensorParamName Q_OUT{"q_out"};
    const TensorParamName K_OUT{"k_out"};
    const TensorParamName V_OUT{"v_out"};
    auto kernel = [&](const KernelSpecName& id, DataMovementHardwareConfig hw) {
        return KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(KERNEL),
            .tensor_bindings =
                {{.tensor_parameter_name = QKV_IN, .accessor_name = "qkv_in"},
                 {.tensor_parameter_name = Q_OUT, .accessor_name = "q_out"},
                 {.tensor_parameter_name = K_OUT, .accessor_name = "k_out"},
                 {.tensor_parameter_name = V_OUT, .accessor_name = "v_out"}},
            .compile_time_args = {{"row_bytes", plan::kRowBytes}},
            .runtime_arg_schema =
                {.runtime_arg_names = {"n"}, .runtime_array_args = {{.name = "plan", .size = 2 * plan::kSlots}}},
            .hw_config = std::move(hw),
        };
    };
    KernelRunArgs reader_args{.kernel = READER};
    KernelRunArgs writer_args{.kernel = WRITER};
    std::vector<CoreRange> ranges;
    for (const auto& c : cores) {
        const CoreCoord core{c.x, c.y};
        ranges.emplace_back(core, core);
        AddRuntimeArgsForNode(reader_args.runtime_arg_values, core, {{"n", c.n_reader}});
        AddRuntimeArgsForNode(writer_args.runtime_arg_values, core, {{"n", c.n_writer}});
        reader_args.runtime_array_arg_values["plan"][core] = std::vector<uint32_t>(c.reader.begin(), c.reader.end());
        writer_args.runtime_array_arg_values["plan"][core] = std::vector<uint32_t>(c.writer.begin(), c.writer.end());
    }
    ProgramSpec spec{
        .name = "llama31_8b_qkv_heads_decode",
        .kernels =
            {kernel(READER, create_reader_datamovement_config()), kernel(WRITER, create_writer_datamovement_config())},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = QKV_IN, .spec = tensor_args.input_tensor.tensor_spec()},
                TensorParameter{.unique_id = Q_OUT, .spec = output[0].tensor_spec()},
                TensorParameter{.unique_id = K_OUT, .spec = output[1].tensor_spec()},
                TensorParameter{.unique_id = V_OUT, .spec = output[2].tensor_spec()},
            },
        .work_units = {WorkUnitSpec{
            .name = "main",
            .kernels = {READER, WRITER},
            .target_nodes = CoreRangeSet(ranges),
        }},
    };
    ProgramRunArgs run;
    run.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run.tensor_args = {
        {QKV_IN, tensor_args.input_tensor.mesh_tensor()},
        {Q_OUT, output[0].mesh_tensor()},
        {K_OUT, output[1].mesh_tensor()},
        {V_OUT, output[2].mesh_tensor()},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim::llama31_8b_qkv_heads_decode
