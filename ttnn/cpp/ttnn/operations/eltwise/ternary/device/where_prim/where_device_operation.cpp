
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/ternary/device/where_prim/where_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>

#include <tracy/Tracy.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::ternary {

WhereDeviceOperation::program_factory_t WhereDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TODO : Add all programs
    ZoneScopedN("WhereDeviceOperation::select_program_factory");
    if (operation_attributes.b_scalar.has_value() && operation_attributes.c_scalar.has_value()) {
        return BroadcastScalarsWhereProgram{};
    }

    return ElementWiseMultiCoreWhereProgram{};
}

static void validate_memory_config(
    const WhereDeviceOperation::operation_attributes_t& attributes,
    const WhereDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    const auto& output_tensor = tensor_args.output_tensor;

    bool tensor_b_sharded = false;
    if (input_tensor_b.has_value()) {
        tensor_b_sharded = input_tensor_b->memory_config().is_sharded();
        TT_FATAL(
            input_tensor_a.device() == input_tensor_b->device(),
            "Operands to eltwise ternary need to be on the same device!");
        TT_FATAL(input_tensor_b->get_layout() == Layout::TILE, "Inputs to eltwise ternary must be tilized");
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        if (tensor_b_sharded) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == input_tensor_b->memory_config().memory_layout(),
                "Error");
            TT_FATAL(input_tensor_a.shard_spec().value() == input_tensor_b->shard_spec().value(), "Error");
        }
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == attributes.memory_config.memory_layout(), "Error");
        } else {
            TT_FATAL(attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    } else if (tensor_b_sharded) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        if (attributes.memory_config.is_sharded()) {
            TT_FATAL(
                input_tensor_b->memory_config().memory_layout() == attributes.memory_config.memory_layout(), "Error");
        } else {
            TT_FATAL(attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    } else {
        TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        if (input_tensor_b.has_value()) {
            TT_FATAL((input_tensor_b->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED), "Error");
        }
        if (!attributes.memory_config.is_sharded()) {
            TT_FATAL(attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
        }
    }
}

void WhereDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& input_tensor_c = tensor_args.input_tensor_c;
    const auto& output_tensor = tensor_args.output_tensor;

    // TODO: We can check it earlier
    TT_FATAL(
        input_tensor_b.has_value() != attributes.b_scalar.has_value(), "Either the tensor b or scalar should be set");

    TT_FATAL(
        input_tensor_c.has_value() != attributes.c_scalar.has_value(), "Either the tensor c or scalar should be set");

    validate_memory_config(attributes, tensor_args);
    WhereDeviceOperation::validate_on_program_cache_hit(attributes, tensor_args);

    // TT_FATAL(input_tensor_a.get_layout() == Layout::TILE, "Input to eltwise ternary must be tilized");

    // auto program_factory = select_program_factory(attributes, tensor_args);
    // std::visit(
    //     [&attributes](auto&& program_factory) {
    //         if constexpr (std::is_same_v<decltype(program_factory), ElementWiseMultiCore>) {
    //             TT_FATAL(not attributes.activations.has_value(), "Error");
    //         }
    //     },
    //     program_factory);
}

void WhereDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& output_tensor = tensor_args.output_tensor;

    const auto& input_shape_a = input_tensor_a.get_logical_shape();

    auto batch_size_0_a = input_shape_a.rank() >= 4 ? input_shape_a[-4] : 1;
    auto batch_size_1_a = input_shape_a.rank() >= 3 ? input_shape_a[-3] : 1;
    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];

    const auto input_shape_b =
        tensor_args.input_tensor_b.has_value() ? tensor_args.input_tensor_b->get_logical_shape() : ttnn::Shape{1, 1};
    auto batch_size_0_b = input_shape_b.rank() >= 4 ? input_shape_b[-4] : 1;
    auto batch_size_1_b = input_shape_b.rank() >= 3 ? input_shape_b[-3] : 1;
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];

    const auto input_shape_c =
        tensor_args.input_tensor_c.has_value() ? tensor_args.input_tensor_c->get_logical_shape() : ttnn::Shape{1, 1};
    auto batch_size_0_c = input_shape_c.rank() >= 4 ? input_shape_c[-4] : 1;
    auto batch_size_1_c = input_shape_c.rank() >= 3 ? input_shape_c[-3] : 1;
    auto height_c = input_shape_c[-2];
    auto width_c = input_shape_c[-1];

    // TODO: Add c_tensor into account, simplify the logic

    // Input shape b must be the same as or broadcastable to input shape a
    if (batch_size_0_a != batch_size_0_b) {
        TT_ASSERT(
            batch_size_0_a > batch_size_0_b and batch_size_0_b == 1,
            "ttnn::operations::ternary::WhereDeviceOperation: batch size mismatch");
    }
    if (batch_size_1_a != batch_size_1_b) {
        TT_ASSERT(
            batch_size_1_a > batch_size_1_b and batch_size_1_b == 1,
            "ttnn::operations::ternary::WhereDeviceOperation: batch size mismatch");
    }

    TT_FATAL(
        height_a == height_b || height_a == 1 || height_b == 1,
        "ttnn::operations::ternary::WhereDeviceOperation: height mismatch");
    TT_FATAL(
        width_a == width_b || width_a == 1 || width_b == 1,
        "ttnn::operations::ternary::WhereDeviceOperation: width mismatch");
}

WhereDeviceOperation::spec_return_value_t WhereDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value()) {
        return output_tensor->get_tensor_spec();
    }

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto input_shape_a = input_tensor_a.logical_shape();
    const auto& tensor_b = tensor_args.input_tensor_b;
    const auto input_shape_b = tensor_b.has_value() ? tensor_b->logical_shape() : ttnn::Shape{};

    const auto& tensor_c = tensor_args.input_tensor_c;
    const auto input_shape_c = tensor_c.has_value() ? tensor_c->logical_shape() : ttnn::Shape{};

    // TODO: take C_tensor into account

    const int rank_a = input_shape_a.rank();
    const int rank_b = input_shape_b.rank();
    const int rank_c = input_shape_c.rank();
    const int larger_rank = std::max(std::max(rank_a, rank_b), rank_c);

    // Broadcasting Rules Overview:
    // - If the two tensors have different ranks, we virtually pad the smaller-rank tensor's shape
    //   with ones on the left (i.e., higher-order dimensions) until both shapes have the same length.
    // - For each dimension (starting from the rightmost), the sizes are compatible if:
    //     - They are equal, or
    //     - One of them is 1 (the dimension can be broadcast to match the other size).
    auto compute_broadcasted_output = [rank_a, rank_b, rank_c, larger_rank](
                                          const auto& shape_a, const auto& shape_b, const auto& shape_c) {
        SmallVector<uint32_t> output_shape(larger_rank, 1);
        for (int i = -1; i >= -larger_rank; --i) {
            auto dim_a = (i >= -rank_a) ? shape_a[i] : 1;
            auto dim_b = (i >= -rank_b) ? shape_b[i] : 1;
            auto dim_c = (i >= -rank_c) ? shape_c[i] : 1;

            // TODO: Implement all cases
            if (dim_a != 1 && dim_b != 1 && dim_c != 1) {
                TT_FATAL(dim_a == dim_b, "Incompatible dimensions {} and {}", dim_a, dim_b);
                TT_FATAL(dim_a == dim_c, "Incompatible dimensions {} and {}", dim_a, dim_c);
                output_shape[i + larger_rank] = dim_a;
            } else if (dim_a != 1 && dim_b == 1 && dim_c == 1) {
                // One of the dimension is one, calculating the other one
                output_shape[i + larger_rank] = dim_a;
            }
        }
        return ttnn::Shape(output_shape);
    };
    auto output_shape = compute_broadcasted_output(input_shape_a, input_shape_b, input_shape_c);

    auto program_factory = select_program_factory(operation_attributes, tensor_args);
    if (std::holds_alternative<ElementWiseMultiCoreWhereProgram>(program_factory)) {
        const auto& input_tensor_b = *tensor_args.input_tensor_b;
        const auto& input_tensor_c = *tensor_args.input_tensor_c;
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                shard_spec = input_tensor_a.shard_spec().value();
            } else if (input_tensor_b.memory_config().is_sharded()) {
                shard_spec = input_tensor_b.shard_spec().value();
            } else if (input_tensor_c.memory_config().is_sharded()) {
                shard_spec = input_tensor_c.shard_spec().value();
            } else {
                shard_spec = operation_attributes.memory_config.shard_spec().value();
            }
            // TODO: We put modified copy of memory_config to TensorSpec.
            // It can cause confusion to have more than one memconfig
            auto memory_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape, TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), memory_config));
        }
    } else {
        if (operation_attributes.memory_config.is_sharded()) {
            ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
            if (input_tensor_a.memory_config().is_sharded()) {
                // Derive output shard_spec based on input
                shard_spec = input_tensor_a.shard_spec().value();
            }
            auto memory_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape, TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), memory_config));
        }
    }
    return TensorSpec(
        output_shape,
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
}

WhereDeviceOperation::tensor_return_value_t WhereDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output_tensor.has_value()) {
        return *tensor_args.output_tensor;
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor_a.device());
}

tt::stl::hash::hash_t WhereDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;

    // TODO: take C_tensor into account

    auto program_factory = select_program_factory(attributes, tensor_args);
    TT_ASSERT(
        std::holds_alternative<DeviceStorage>(input_tensor_a.get_storage()),
        "Unexpected type {}",
        tt::stl::get_active_type_name_in_variant(input_tensor_a.get_storage()));

    if (input_tensor_b.has_value()) {
        TT_ASSERT(
            std::holds_alternative<DeviceStorage>(input_tensor_b->get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(input_tensor_b->get_storage()));

        return operation::hash_operation<WhereDeviceOperation>(
            attributes,
            program_factory.index(),
            input_tensor_a.dtype(),
            std::get<DeviceStorage>(input_tensor_a.storage()).memory_config(),
            input_tensor_b->dtype(),
            std::get<DeviceStorage>(input_tensor_b->storage()).memory_config());
    }

    return operation::hash_operation<WhereDeviceOperation>(
        attributes,
        program_factory.index(),
        input_tensor_a.dtype(),
        std::get<DeviceStorage>(input_tensor_a.storage()).memory_config());
}

operation::OpPerformanceModel WhereDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& output_tensor = tensor_return_value;
    // GS specific parameters
    // 80 B/cycle unpacker BW shared
    // 128 datums per cycle math, but unpacker cant keep up
    constexpr uint32_t num_cores = 9 * 12;

    uint32_t total_bytes = 0;
    std::vector<Tensor> input_tensors = {input_tensor_a};
    total_bytes += input_tensor_a.volume() * input_tensor_a.element_size();
    if (input_tensor_b.has_value()) {
        input_tensors.push_back(*input_tensor_b);
        total_bytes += input_tensor_b->volume() * input_tensor_b->element_size();
    }
    uint32_t ideal_eltwise_cycles = total_bytes / 80 / num_cores;

    // TODO: update OpPerformanceModel to work on variadic arguments
    operation::OpPerformanceModel result(input_tensors, {output_tensor}, ideal_eltwise_cycles);
#if 0
        tt::log_info(tt::LogOp, "WhereDeviceOperation PerfModel:");
        tt::log_info(tt::LogOp, "\t Data (Bytes): {}", total_bytes);
        tt::log_info(tt::LogOp, "\t ideal_eltwise_cycles: {}", ideal_eltwise_cycles);
#endif
    return result;
}

bool WhereDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

}  // namespace ttnn::operations::ternary
