// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_boltz_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::transformer {
// Generic NLP CreateHeads op
void NlpCreateHeadsBoltzDeviceOperation::validate_on_program_cache_miss(
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
        "Unsupported data type");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor layout must be TILE but got {}", input_tensor.layout());

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0, "Unsupported input height {} is not tile aligned", input_shape[2]);
    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.shard_spec().value().shape[0] ==
                input_tensor.logical_shape().volume() / input_tensor.padded_shape()[-1],
            "Shard spec shape[0] ({}) must equal logical volume / padded shape[-1] ({})",
            input_tensor.shard_spec().value().shape[0],
            input_tensor.logical_shape().volume() / input_tensor.padded_shape()[-1]);
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
                "Error");
            TT_FATAL(
                input_tensor.shard_spec().value().orientation ==
                    tensor_args.input_tensor_kv.value().shard_spec().value().orientation,
                "Error");
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
        TT_FATAL(input_shape_kv[2] == input_shape[2], "KV tensor seq_len dim must be same as Q tensor seq_len!");
        if (input_tensor_kv.is_sharded()) {
            TT_FATAL(input_tensor.is_sharded(), "Input tensor must be sharded when KV tensor is sharded");
            TT_FATAL(
                input_tensor_kv.shard_spec().value().shape[0] ==
                    input_tensor_kv.logical_shape().volume() / input_tensor_kv.padded_shape()[-1],
                "KV tensor shard spec shape[0] ({}) must equal logical volume / padded shape[-1] ({})",
                input_tensor_kv.shard_spec().value().shape[0],
                input_tensor_kv.logical_shape().volume() / input_tensor_kv.padded_shape()[-1]);
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

void NlpCreateHeadsBoltzDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

NlpCreateHeadsBoltzDeviceOperation::spec_return_value_t NlpCreateHeadsBoltzDeviceOperation::compute_output_specs(
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

    const Shape q_output_shape({operation_attributes.num_q_heads, sequence_length, sequence_length, head_dim});
    const Shape v_output_shape({operation_attributes.num_kv_heads, sequence_length, sequence_length, head_dim});
    const Shape k_output_shape =
        operation_attributes.transpose_k_heads
            ? Shape({operation_attributes.num_kv_heads, head_dim, sequence_length, sequence_length})
            : v_output_shape;

    if (operation_attributes.output_mem_config.is_sharded()) {
        auto core_grid = input_tensor.device()->compute_with_storage_grid_size();
        auto q_shard_grid = tt::tt_metal::num_cores_to_corerangeset(operation_attributes.num_q_heads, core_grid, true);
        tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto q_mem_config = operation_attributes.output_mem_config.with_shard_spec(q_shard_spec);
        auto kv_shard_grid =
            tt::tt_metal::num_cores_to_corerangeset(operation_attributes.num_kv_heads, core_grid, true);
        tt::tt_metal::ShardSpec kv_shard_spec{kv_shard_grid, {TILE_HEIGHT, operation_attributes.head_dim}};
        auto kv_mem_config = operation_attributes.output_mem_config.with_shard_spec(kv_shard_spec);
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

NlpCreateHeadsBoltzDeviceOperation::tensor_return_value_t NlpCreateHeadsBoltzDeviceOperation::create_output_tensors(
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

std::vector<tt::tt_metal::DynamicRuntimeArg> NlpCreateHeadsBoltzDeviceOperation::get_dynamic_runtime_args(
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

    const auto& input_tensor_kv = tensor_args.input_tensor_kv;
    auto& output = tensor_return_value;
    const auto head_dim = operation_attributes.head_dim;
    const auto num_q_heads = operation_attributes.num_q_heads;
    const auto num_kv_heads = operation_attributes.num_kv_heads;

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const bool read_from_input_tensor_kv = input_tensor_kv.has_value();
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t head_tiles = head_dim / TILE_WIDTH;
    const uint32_t head_size = head_tiles * single_tile_size;

    const auto q_shard_spec = std::get<0>(output).shard_spec().value();
    const auto q_cores = q_shard_spec.grid;

    const uint32_t per_core_out_q_heads = num_q_heads / q_cores.num_cores();
    const uint32_t per_risc0_out_q_heads = tt::div_up(per_core_out_q_heads, 2);
    const uint32_t per_risc1_out_q_heads = per_core_out_q_heads / 2;
    const uint32_t per_core_in_q_heads = num_q_heads / input_tensor.shard_spec().value().num_cores();

    const auto k_shard_spec = std::get<1>(output).shard_spec().value();
    const auto k_cores = k_shard_spec.grid;

    const uint32_t per_core_out_kv_heads = num_kv_heads / k_cores.num_cores();
    const uint32_t per_core_in_kv_heads =
        num_kv_heads / (read_from_input_tensor_kv ? input_tensor_kv.value().shard_spec().value().num_cores()
                                                  : input_tensor.shard_spec().value().num_cores());

    // Recompute base addresses from the CURRENT buffers (these are what change across dispatches).
    const uint32_t q_base_addr = input_tensor.buffer()->address();
    uint32_t k_base_addr = 0;
    if (read_from_input_tensor_kv) {
        k_base_addr = input_tensor_kv.value().buffer()->address();
    } else {
        k_base_addr = q_base_addr + per_core_in_q_heads * head_tiles * single_tile_size;
    }
    const uint32_t v_base_addr = k_base_addr + (per_core_in_kv_heads * head_tiles * single_tile_size);

    const uint32_t num_cores = std::max(q_cores.num_cores(), k_cores.num_cores());
    const auto core_grid = q_cores.bounding_box();
    const uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = tt::tt_metal::grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // Sharded always uses (reader=kernel 0, writer=kernel 1); transpose_k_heads is forbidden for sharded,
    // so no compute kernels precede them in ProgramDescriptor::kernels.
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    // Address-derived rt-arg indices, identical for reader and writer arg vectors:
    //   [6]=q_base_addr, [7]=q_start_addr, [15]=k_base_addr/v_base_addr, [16]=k_start_addr/v_start_addr.
    constexpr uint32_t kIdxQBase = 6;
    constexpr uint32_t kIdxQStart = 7;
    constexpr uint32_t kIdxKBase = 15;
    constexpr uint32_t kIdxKStart = 16;

    dynamic_args.reserve(num_cores * 8);

    // Mirror the exact per-core address state machine in Sharded::create_descriptor.
    uint32_t remote_q_head_start_idx = 0;
    uint32_t remote_kv_head_start_idx = 0;
    uint32_t q_start_addr = q_base_addr;
    uint32_t k_start_addr = k_base_addr;
    uint32_t v_start_addr = v_base_addr;
    [[maybe_unused]] uint32_t remote_q_read = 0;
    [[maybe_unused]] uint32_t remote_kv_read = 0;

    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        const bool read_kv_heads = i < k_cores.num_cores();

        // Reader rt-args snapshot taken at the top of the loop (before the writer mutations).
        const uint32_t reader_q_start_addr = q_start_addr;
        const uint32_t reader_k_start_addr = k_start_addr;
        dynamic_args.push_back({kReaderKernelIdx, core, kIdxQBase, q_base_addr});
        dynamic_args.push_back({kReaderKernelIdx, core, kIdxQStart, reader_q_start_addr});
        dynamic_args.push_back({kReaderKernelIdx, core, kIdxKBase, k_base_addr});
        dynamic_args.push_back({kReaderKernelIdx, core, kIdxKStart, reader_k_start_addr});

        // Advance q state for risc0. The factory writes reader_runtime_args[7] exactly once here (the
        // post-risc0 value), then moves that same vector into the writer; the subsequent risc1 block only
        // updates local scalars for the NEXT core, so the writer's q_start_addr is the post-risc0 value.
        remote_q_read += per_risc0_out_q_heads;
        remote_q_head_start_idx = (remote_q_head_start_idx + per_risc0_out_q_heads) % per_core_in_q_heads;
        q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
        const uint32_t writer_q_start_addr = q_start_addr;

        if (per_risc1_out_q_heads > 0) {
            remote_q_read += per_risc1_out_q_heads;
            remote_q_head_start_idx = (per_risc1_out_q_heads + remote_q_head_start_idx) % per_core_in_q_heads;
            q_start_addr = q_base_addr + remote_q_head_start_idx * head_size;
        }

        // Writer rt-args: the kv slots carry v_* when this core reads kv heads, otherwise they retain the
        // reader's k_* snapshot (the factory leaves reader_runtime_args[15]/[16] unmutated in that case).
        const uint32_t writer_k_base = read_kv_heads ? v_base_addr : k_base_addr;
        const uint32_t writer_k_start = read_kv_heads ? v_start_addr : reader_k_start_addr;
        dynamic_args.push_back({kWriterKernelIdx, core, kIdxQBase, q_base_addr});
        dynamic_args.push_back({kWriterKernelIdx, core, kIdxQStart, writer_q_start_addr});
        dynamic_args.push_back({kWriterKernelIdx, core, kIdxKBase, writer_k_base});
        dynamic_args.push_back({kWriterKernelIdx, core, kIdxKStart, writer_k_start});

        if (read_kv_heads) {
            remote_kv_read += per_core_out_kv_heads;
            remote_kv_head_start_idx = (remote_kv_head_start_idx + per_core_out_kv_heads) % per_core_in_kv_heads;
            k_start_addr = k_base_addr + remote_kv_head_start_idx * head_size;
            v_start_addr = v_base_addr + remote_kv_head_start_idx * head_size;
        }
    }

    return dynamic_args;
}

NlpCreateHeadsBoltzDeviceOperation::program_factory_t NlpCreateHeadsBoltzDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor_q;
    if (input_tensor.is_sharded()) {
        return Sharded{};
    }
    return Interleaved{};
}

}  // namespace ttnn::operations::experimental::transformer

namespace ttnn::prim {

std::tuple<Tensor, Tensor, Tensor> nlp_create_qkv_heads_boltz(
    const Tensor& input_tensor_q,
    const std::optional<Tensor>& input_tensor_kv,
    uint32_t num_q_heads,
    std::optional<uint32_t> num_kv_heads,
    uint32_t head_dim,
    bool transpose_k_heads,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<std::vector<std::optional<Tensor>>>& optional_output_tensors) {
    using OperationType = ttnn::operations::experimental::transformer::NlpCreateHeadsBoltzDeviceOperation;

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
