// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

// Generic NLP CreateHeads op
void NlpCreateHeadsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto input_shape = input_tensor.padded_shape();

    // NOTE: Checks for head_dim and shape[3] is done in nlp_create_qkv_heads because it's needed to infer head_dim
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Operands to TM need to be on device! {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0, "Unsupported input height {} is not tile aligned", input_shape[2]);
    TT_FATAL(input_shape[1] == 1, "Unsupported input sequence length {} is not equal to 1", input_shape[1]);
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] ==
                input_tensor.physical_volume() / input_tensor.padded_shape()[-1],
            "Shard spec shape[0] ({}) must equal physical volume / padded shape[-1] ({})",
            input_tensor.shard_spec().value().shape[0],
            input_tensor.physical_volume() / input_tensor.padded_shape()[-1]);
        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded() &&
                operation_attributes.output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "Output memory config must be sharded and not WIDTH_SHARDED but got memory_layout: {}",
            operation_attributes.output_mem_config.memory_layout());
        TT_FATAL(
            input_tensor.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
            "Input tensor shard orientation must be ROW_MAJOR but got {}",
            input_tensor.shard_spec().value().orientation);
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        uint32_t num_cores = core_grid.x * core_grid.y;
        // 1 Head Per Core Max for now
        TT_FATAL(
            operation_attributes.num_q_heads <= num_cores,
            "Number of Q heads ({}) must be <= number of cores ({})",
            operation_attributes.num_q_heads,
            num_cores);
        TT_FATAL(
            operation_attributes.num_kv_heads <= num_cores,
            "Number of KV heads ({}) must be <= number of cores ({})",
            operation_attributes.num_kv_heads,
            num_cores);
        TT_FATAL(
            operation_attributes.num_q_heads >= operation_attributes.num_kv_heads,
            "Number of Q heads ({}) must be >= number of KV heads ({})",
            operation_attributes.num_q_heads,
            operation_attributes.num_kv_heads);
        TT_FATAL(
            operation_attributes.num_q_heads % input_tensor.shard_spec().value().num_cores() == 0,
            "Number of Q heads ({}) must be divisible by number of cores ({})",
            operation_attributes.num_q_heads,
            input_tensor.shard_spec().value().num_cores());
        if (tensor_args.input_tensor_kv.has_value()) {
            TT_FATAL(tensor_args.input_tensor_kv.value().is_sharded(), "Input tensor KV must be sharded");
            TT_FATAL(
                input_tensor.shard_spec().value().grid == tensor_args.input_tensor_kv.value().shard_spec().value().grid,
                "Input tensor and KV tensor must have the same shard grid");
            TT_FATAL(
                input_tensor.shard_spec().value().orientation ==
                    tensor_args.input_tensor_kv.value().shard_spec().value().orientation,
                "Input tensor and KV tensor must have the same shard orientation");
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] ==
                    (operation_attributes.num_q_heads / operation_attributes.num_kv_heads) *
                        operation_attributes.head_dim,
                "Shard spec shape[1] ({}) must equal (num_q_heads / num_kv_heads) * head_dim ({})",
                input_tensor.shard_spec().value().shape[1],
                (operation_attributes.num_q_heads / operation_attributes.num_kv_heads) * operation_attributes.head_dim);
        } else {
            TT_FATAL(
                operation_attributes.num_kv_heads % input_tensor.shard_spec().value().num_cores() == 0,
                "Number of KV heads ({}) must be divisible by number of cores ({})",
                operation_attributes.num_kv_heads,
                input_tensor.shard_spec().value().num_cores());
            TT_FATAL(
                input_tensor.shard_spec().value().shape[1] ==
                    (operation_attributes.num_q_heads / operation_attributes.num_kv_heads + 2) *
                        operation_attributes.head_dim,
                "Shard spec shape[1] ({}) must equal (num_q_heads / num_kv_heads + 2) * head_dim ({})",
                input_tensor.shard_spec().value().shape[1],
                (operation_attributes.num_q_heads / operation_attributes.num_kv_heads + 2) *
                    operation_attributes.head_dim);
        }
        TT_FATAL(!operation_attributes.transpose_k_heads, "Transpose K heads must be false");
    } else {
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Output memory config layout must be INTERLEAVED but got {}",
            operation_attributes.output_mem_config.memory_layout());
    }

    if (tensor_args.input_tensor_kv.has_value()) {
        const auto& input_tensor_kv = tensor_args.input_tensor_kv.value();
        const auto input_shape_kv = input_tensor_kv.padded_shape();

        TT_FATAL(input_tensor_kv.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
        TT_FATAL(input_tensor_kv.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
        TT_FATAL(input_tensor_kv.dtype() == input_tensor.dtype(), "KV tensor dtype must be same as Q tensor dtype!");
        TT_FATAL(
            input_tensor_kv.layout() == Layout::TILE,
            "Input tensor KV layout must be TILE but got {}",
            input_tensor_kv.layout());

        TT_FATAL(input_shape_kv[0] == input_shape[0], "KV tensor batch dim must be same as Q tensor batch!");
        TT_FATAL(input_shape_kv[1] == 1, "Unsupported input shape {} is not equal to 1", input_shape_kv[1]);
        TT_FATAL(input_shape_kv[2] == input_shape[2], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        if (input_tensor_kv.is_sharded()) {
            TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded when KV tensor is sharded");
            TT_FATAL(
                input_tensor_kv.shard_spec().value().shape[0] ==
                    input_tensor_kv.physical_volume() / input_tensor_kv.padded_shape()[-1],
                "KV tensor shard spec shape[0] ({}) must equal physical volume / padded shape[-1] ({})",
                input_tensor_kv.shard_spec().value().shape[0],
                input_tensor_kv.physical_volume() / input_tensor_kv.padded_shape()[-1]);
            TT_FATAL(
                input_tensor_kv.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                "KV tensor shard orientation must be ROW_MAJOR but got {}",
                input_tensor_kv.shard_spec().value().orientation);
            TT_FATAL(
                input_tensor_kv.shard_spec().value().shape[1] == 2 * operation_attributes.head_dim,
                "KV tensor shard spec shape[1] ({}) must equal 2 * head_dim ({})",
                input_tensor_kv.shard_spec().value().shape[1],
                2 * operation_attributes.head_dim);
            TT_FATAL(
                operation_attributes.num_kv_heads % input_tensor_kv.shard_spec().value().num_cores() == 0,
                "Number of KV heads ({}) must be divisible by KV tensor number of cores ({})",
                operation_attributes.num_kv_heads,
                input_tensor_kv.shard_spec().value().num_cores());
        }
    }
}

void NlpCreateHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

NlpCreateHeadsDeviceOperation::spec_return_value_t NlpCreateHeadsDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    if (tensor_args.optional_output_tensors.size() == 3) {
        const auto& output_tensors = tensor_args.optional_output_tensors;
        return {
            output_tensors.at(0)->tensor_spec(),
            output_tensors.at(1)->tensor_spec(),
            output_tensors.at(2)->tensor_spec()};
    }

    const auto& input_tensor = tensor_args.input_tensor_q;
    const auto input_shape = input_tensor.logical_shape();

    auto sequence_length = input_shape[2];
    auto head_dim = operation_attributes.head_dim;

    if (head_dim % TILE_WIDTH != 0) {
        head_dim = (head_dim / TILE_WIDTH + 1) * TILE_WIDTH;
    }

    const Shape q_output_shape({input_shape[0], operation_attributes.num_q_heads, sequence_length, head_dim});
    const Shape v_output_shape({input_shape[0], operation_attributes.num_kv_heads, sequence_length, head_dim});
    const Shape k_output_shape =
        operation_attributes.transpose_k_heads
            ? Shape({input_shape[0], operation_attributes.num_kv_heads, head_dim, sequence_length})
            : v_output_shape;

    if (operation_attributes.output_mem_config.is_sharded()) {
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        auto q_shard_grid = tt::tt_metal::num_cores_to_corerangeset(operation_attributes.num_q_heads, core_grid, true);
        tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto q_mem_config = tt::tt_metal::MemoryConfig(
            operation_attributes.output_mem_config.memory_layout(),
            operation_attributes.output_mem_config.buffer_type(),
            q_shard_spec);
        auto kv_shard_grid =
            tt::tt_metal::num_cores_to_corerangeset(operation_attributes.num_kv_heads, core_grid, true);
        tt::tt_metal::ShardSpec kv_shard_spec{kv_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto kv_mem_config = tt::tt_metal::MemoryConfig(
            operation_attributes.output_mem_config.memory_layout(),
            operation_attributes.output_mem_config.buffer_type(),
            kv_shard_spec);
        return {
            TensorSpec(
                q_output_shape,
                tt::tt_metal::TensorLayout(
                    input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), q_mem_config)),
            TensorSpec(
                k_output_shape,
                tt::tt_metal::TensorLayout(
                    input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), kv_mem_config)),
            TensorSpec(
                v_output_shape,
                tt::tt_metal::TensorLayout(
                    input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), kv_mem_config))};
    }

    return {
        TensorSpec(
            q_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config)),
        TensorSpec(
            k_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config)),
        TensorSpec(
            v_output_shape,
            tt::tt_metal::TensorLayout(
                input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), operation_attributes.output_mem_config))};
}

NlpCreateHeadsDeviceOperation::tensor_return_value_t NlpCreateHeadsDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input_tensor_q;
    if (tensor_args.optional_output_tensors.size() == 3) {
        const auto& output_tensors = tensor_args.optional_output_tensors;
        return {output_tensors.at(0).value(), output_tensors.at(1).value(), output_tensors.at(2).value()};
    }
    auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), input_tensor.device()),
        create_device_tensor(std::get<1>(output_specs), input_tensor.device()),
        create_device_tensor(std::get<2>(output_specs), input_tensor.device()),
    };
}

std::vector<tt::tt_metal::DynamicRuntimeArg> NlpCreateHeadsDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    using namespace tt::constants;
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;

    const auto& input_tensor = tensor_args.input_tensor_q;
    // The Interleaved factory binds q/k/v input/output addresses as patchable Buffer* rt-args, so the
    // framework re-patches them automatically. Only the Sharded factory bakes raw address scalars.
    if (!input_tensor.is_sharded()) {
        return dynamic_args;
    }

    // Sharded always uses (reader=kernel 0, writer=kernel 1); transpose_k_heads is forbidden for sharded,
    // so no compute kernels precede them in ProgramDescriptor::kernels.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;

    // Single source of truth: re-derive the per-core args the same way Sharded::create_descriptor() does,
    // then re-apply only the address slots (recomputed from the CURRENT buffers — these are what change
    // across dispatches). The remaining shape/scalar slots are stable across cache hits, so they are left
    // baked. Behaviour matches the slow-path rebuild, just emitted as dynamic tuples.
    auto per_core =
        compute_nlp_create_qkv_heads_sharded_per_core_args(operation_attributes, tensor_args, tensor_return_value);

    dynamic_args.reserve(per_core.cores.size() * per_core.addr_indices.size() * 2);
    for (uint32_t i = 0; i < per_core.cores.size(); ++i) {
        if (!per_core.is_work_core[i]) {
            continue;
        }
        const auto& core = per_core.cores[i];
        const auto& r = per_core.reader_args[i];
        const auto& w = per_core.writer_args[i];
        for (uint32_t idx : per_core.addr_indices) {
            dynamic_args.push_back({kReaderKernelIdx, core, idx, r[idx]});
        }
        for (uint32_t idx : per_core.addr_indices) {
            dynamic_args.push_back({kWriterKernelIdx, core, idx, w[idx]});
        }
    }

    return dynamic_args;
}

NlpCreateHeadsDeviceOperation::program_factory_t NlpCreateHeadsDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    if (input_tensor.is_sharded()) {
        return Sharded{};
    }
    return Interleaved{};
}

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::NlpCreateHeadsDeviceOperation;

    auto operation_attributes = OperationType::operation_attributes_t{
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads.value_or(num_q_heads),
        .head_dim = head_dim,
        .transpose_k_heads = transpose_k_heads,
        .output_mem_config = memory_config.value_or(input_tensor_q.memory_config())};
    auto tensor_args = OperationType::tensor_args_t{
        .input_tensor_q = input_tensor_q,
        .input_tensor_kv = input_tensor_kv,
        .optional_output_tensors = optional_output_tensors.value_or(std::vector<std::optional<Tensor>>{})};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
