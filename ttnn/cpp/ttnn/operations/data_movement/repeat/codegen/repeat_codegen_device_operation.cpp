// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_device_operation.hpp"

#include <optional>

#include <tt_stl/assert.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_supported.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::prim {

RepeatCodegenDeviceOperation::program_factory_t RepeatCodegenDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return RepeatCodegenProgramFactory{};
}

namespace {

namespace repeat_codegen = ttnn::operations::data_movement::repeat_codegen;

const MemoryConfig& output_mem_config_of(
    const RepeatCodegenParams& operation_attributes, const RepeatCodegenInputs& tensor_args) {
    return tensor_args.optional_output_tensor.has_value() ? tensor_args.optional_output_tensor->memory_config()
                                                          : operation_attributes.output_mem_config;
}

// The output checks both validators run. A preallocated output is not part of the program-cache key
// beyond its spec, so a hit can pair a cached program with a buffer that aliases the input or sits on
// another device.
void validate_output(const RepeatCodegenParams& operation_attributes, const RepeatCodegenInputs& tensor_args) {
    const Tensor& input = tensor_args.input;
    auto expected_shape = input.logical_shape();
    expected_shape[operation_attributes.rep_dim] *= operation_attributes.num_repeats;
    // The writers address the output by page id alone, so a sharded output must keep the interleaved page grid.
    TT_FATAL(
        repeat_codegen::shard_spec_is_page_identical(
            output_mem_config_of(operation_attributes, tensor_args), expected_shape, input.layout()),
        "RepeatCodegen output placement must be interleaved or a page-identical shard spec");

    if (!tensor_args.optional_output_tensor.has_value()) {
        return;
    }
    const auto& out = tensor_args.optional_output_tensor.value();
    TT_FATAL(out.storage_type() == ttnn::StorageType::DEVICE, "Repeat codegen optional output must be on device");
    TT_FATAL(out.buffer() != nullptr, "Repeat codegen optional output must be allocated in a buffer on device");
    TT_FATAL(out.logical_shape() == expected_shape, "Repeat codegen optional output shape mismatch");
    TT_FATAL(
        repeat_codegen::output_matches_input_page(input, out),
        "Repeat codegen optional output must match the input's dtype, layout and tile");
    TT_FATAL(out.device() == input.device(), "Repeat codegen optional output must be on the same device");
    TT_FATAL(out.buffer() != input.buffer(), "Repeat codegen optional output must not alias the input buffer");
}

}  // namespace

void RepeatCodegenDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input = tensor_args.input;
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to repeat need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        repeat_codegen::supported_by_codegen(
            input,
            operation_attributes.rep_dim,
            operation_attributes.num_repeats,
            output_mem_config_of(operation_attributes, tensor_args)),
        "Input is not supported by RepeatCodegen");
    validate_output(operation_attributes, tensor_args);
}

void RepeatCodegenDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // The key pins the input spec, the attributes and the CB plan; only the buffers can differ.
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    validate_output(operation_attributes, tensor_args);
}

// The default key is the attributes and the tensor specs, and neither sees what else occupies L1. A
// row-major CB is sized to the L1 left free, and Program::validate_circular_buffer_region re-checks a
// cached program's CB region against the current frontier on every enqueue, so the key also carries
// the plan: a frontier that moves without changing the plan still hits. create_output_tensors() has
// run by the time the key is computed, so this sees the frontier create_descriptor() will.
//
// A custom hash opts this op out of the canonical program-cache key, so a 64-bit collision between two
// distinct repeat specs resolves to a wrong hit rather than a rebuild.
ttsl::hash::hash_t RepeatCodegenDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    uint32_t cb_batch = 0;
    uint32_t cb_depth = 0;
    if (tensor_args.input.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        const auto cb_plan =
            rm_cb_plan_for_call(tensor_args.input, compute_output_specs(operation_attributes, tensor_args));
        if (cb_plan.has_value()) {
            cb_batch = cb_plan->batch;
            cb_depth = cb_plan->depth;
        }
    }
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<RepeatCodegenDeviceOperation>, operation_attributes, tensor_args, cb_batch, cb_depth);
}

RepeatCodegenDeviceOperation::spec_return_value_t RepeatCodegenDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }
    const auto& input = tensor_args.input;
    auto output_shape = input.logical_shape();
    output_shape[operation_attributes.rep_dim] *= operation_attributes.num_repeats;
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(),
            tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()),
            operation_attributes.output_mem_config));
}

RepeatCodegenDeviceOperation::tensor_return_value_t RepeatCodegenDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> RepeatCodegenDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = tensor_args.input;
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    return {{input_tensor}, output_tensor, ideal_dev_clock_cycles};
}

RepeatCodegenDeviceOperation::tensor_return_value_t repeat_codegen(
    const Tensor& input, const RepeatCodegenParams& params, std::optional<Tensor> optional_output_tensor) {
    // The factory indexes a 4D page map by rep_dim and divides the output page count by num_repeats.
    TT_FATAL(
        input.logical_shape().rank() == 4,
        "RepeatCodegen expects a 4D input, got rank {}",
        input.logical_shape().rank());
    TT_FATAL(params.rep_dim < 4, "RepeatCodegen rep_dim must be in [0, 3], got {}", params.rep_dim);
    TT_FATAL(params.num_repeats >= 1, "RepeatCodegen num_repeats must be at least 1, got {}", params.num_repeats);
    // The kernels split total_out_pages writes across the output buffer, so a page map that does not
    // describe this input addresses pages past its end.
    const auto page_map = derive_page_map(input, params.rep_dim, params.num_repeats);
    TT_FATAL(
        params.lower_pages == page_map.lower_pages && params.rep_dim_pages == page_map.rep_dim_pages &&
            params.total_out_pages == page_map.total_out_pages && params.stick_size == page_map.stick_size,
        "RepeatCodegen page map (lower_pages {}, rep_dim_pages {}, total_out_pages {}, stick_size {}) does not "
        "match the one derived from the input (lower_pages {}, rep_dim_pages {}, total_out_pages {}, stick_size {})",
        params.lower_pages,
        params.rep_dim_pages,
        params.total_out_pages,
        params.stick_size,
        page_map.lower_pages,
        page_map.rep_dim_pages,
        page_map.total_out_pages,
        page_map.stick_size);
    using OperationType = RepeatCodegenDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{.input = input, .optional_output_tensor = std::move(optional_output_tensor)});
}

}  // namespace ttnn::prim
