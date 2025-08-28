// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "softmax/device/softmax_types.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/core.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::normalization::softmax {

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Determine if we should use sharded multi-core program factory
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto tile_width = tensor_args.input_tensor.tensor_spec().tile().get_width();
    const auto rank = input_tensor_shape.size();

    if (operation_attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) {
        return program::SoftmaxProgramFactoryAttentionOptimized{};
    } else if (
        operation_attributes.softmax_type == SoftmaxOperationType::Softmax && operation_attributes.dim == rank - 1 &&
        rank == 4) {
        return program::SoftmaxProgramFactoryAttentionOptimized{};
    }
    return program::SoftmaxProgramFactoryGeneral{};
}

void SoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return validate_on_program_cache_miss(attributes, tensor_args);
}

void SoftmaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensors_args) {
    TT_FATAL(
        tensors_args.input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
    TT_FATAL(
        tensors_args.input_tensor.buffer() != nullptr,
        "Operands to softmax need to be allocated in buffers on device!");
    TT_FATAL((tensors_args.input_tensor.layout() == Layout::TILE), "Inputs to softmax must be tilized");
    TT_FATAL(
        tensors_args.input_tensor.dtype() == DataType::FLOAT32 ||
            tensors_args.input_tensor.dtype() == DataType::BFLOAT16 ||
            tensors_args.input_tensor.dtype() == DataType::BFLOAT8_B,
        "Input tensor must be FLOAT32, BFLOAT16, or BFLOAT8_B, got: {}",
        tensors_args.input_tensor.dtype());
    if (tensors_args.mask.has_value()) {
        auto& mask = tensors_args.mask.value();
        TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
        TT_FATAL(
            tensors_args.input_tensor.device() == mask.device(), "Input tensor and mask must be on the same device");
        if (mask.is_sharded()) {  // sharded mask
            TT_FATAL(mask.layout() == Layout::TILE, "Sharded mask must have TILE layout");
            TT_FATAL(
                mask.padded_shape() == tensors_args.input_tensor.padded_shape(),
                "Sharded mask shape must match input tensor shape");
        } else {
            if (mask.layout() == Layout::ROW_MAJOR) {
                const auto tile_width = tensors_args.input_tensor.tensor_spec().tile().get_width();
                const auto tile_height = tensors_args.input_tensor.tensor_spec().tile().get_height();
                ttnn::Shape expected_shape(
                    {mask.padded_shape()[0],
                     1,
                     tensors_args.input_tensor.padded_shape()[-1] / tile_width,
                     tile_height});
                TT_FATAL(mask.padded_shape() == expected_shape, "Non-sharded mask shape must match expected shape");
            }
            for (uint32_t i = 1; i < tensors_args.input_tensor.padded_shape().rank() - 2; i++) {
                TT_FATAL(mask.padded_shape()[i] == 1, "Non-sharded mask intermediate dimensions must be 1");
            }
            std::visit(
                [&](const auto& program_config) {
                    using ProgramConfigType = std::decay_t<decltype(program_config)>;
                    if constexpr (std::is_same_v<ProgramConfigType, SoftmaxDefaultProgramConfig>) {
                        TT_FATAL(
                            tensors_args.input_tensor.padded_shape() == tensors_args.mask.value().padded_shape(),
                            "Input and mask batch sizes must match");
                        TT_FATAL(
                            !attributes.is_scale_causal_mask_hw_dims_softmax,
                            "Scale causal mask HW dims softmax not supported in default program config");
                    } else if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                        const auto& shape = tensors_args.input_tensor.padded_shape();
                        uint32_t M = tensors_args.input_tensor.physical_volume() / shape[-1];
                        uint32_t K = shape[-1];

                        TT_FATAL(
                            M % tensors_args.input_tensor.tensor_spec().tile().get_height() == 0,
                            "M must be divisible by tile height.");
                        TT_FATAL(
                            K % tensors_args.input_tensor.tensor_spec().tile().get_width() == 0,
                            "K must be divisible by tile width.");
                        TT_FATAL(
                            program_config.block_w % program_config.subblock_w == 0,
                            "block_w must be divisible by subblock_w.");
                        TT_FATAL(
                            program_config.block_w * tensors_args.input_tensor.tensor_spec().tile().get_width() ==
                                shape[3],
                            "shard width must equal to input tensor shape[3]!");
                        TT_FATAL(attributes.inplace, "Operation must be inplace for sharded multi-core program config");
                        if (!attributes.is_scale_causal_mask_hw_dims_softmax) {
                            // grid
                            auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                            auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                            // check dims
                            TT_FATAL(
                                M * K /
                                        ((program_config.block_w * program_config.block_h) *
                                         tensors_args.input_tensor.tensor_spec().tile().get_tile_hw()) ==
                                    num_cores_r * num_cores_c,
                                "number of shards must equal to number of cores. M = {}, K = {}, block_w = {}, block_h "
                                "= {}, num_cores = {}",
                                M,
                                K,
                                program_config.block_w,
                                program_config.block_h,
                                num_cores_r * num_cores_c);
                        } else {
                            TT_FATAL(
                                attributes.is_causal_mask,
                                "Causal mask is required for scale causal mask HW dims softmax");
                            TT_FATAL(
                                mask.layout() == Layout::TILE,
                                "Mask must have TILE layout for scale causal mask HW dims softmax");
                            TT_FATAL(
                                mask.is_sharded() == false,
                                "Mask must not be sharded for scale causal mask HW dims softmax");
                            TT_FATAL(
                                tensors_args.input_tensor.layout() == Layout::TILE,
                                "Input must have TILE layout for scale causal mask HW dims softmax");
                            TT_FATAL(
                                tensors_args.input_tensor.is_sharded(),
                                "Input must be sharded for scale causal mask HW dims softmax");
                            TT_FATAL(
                                tensors_args.input_tensor.shard_spec()->orientation ==
                                    tt::tt_metal::ShardOrientation::ROW_MAJOR,
                                "Input must have ROW_MAJOR shard orientation for scale causal mask HW dims softmax");
                            TT_FATAL(
                                attributes.scale.has_value(),
                                "Scale value is required for scale causal mask HW dims softmax");
                        }
                    }
                },
                attributes.program_config);
            TT_FATAL(
                mask.padded_shape()[-2] == tensors_args.input_tensor.padded_shape()[-2],
                "Non-sharded mask second last dimension must match input tensor. Got mask: {} and input tensor: {}",
                mask.padded_shape()[-2],
                tensors_args.input_tensor.padded_shape()[-2]);
            TT_FATAL(
                mask.padded_shape()[-1] == tensors_args.input_tensor.padded_shape()[-1],
                "Non-sharded mask last dimension must match input tensor. Got mask: {} and input tensor: {}",
                mask.padded_shape()[-1],
                tensors_args.input_tensor.padded_shape()[-1]);
        }
    } else {
        TT_FATAL(not attributes.scale.has_value(), "Scale value must not be set when mask is not present");
    }
}

SoftmaxDeviceOperation::spec_return_value_t SoftmaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if ((attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) &&
        attributes.inplace) {
        return tensor_args.input_tensor.tensor_spec();
    }
    return {TensorSpec(
        tensor_args.input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            tensor_args.input_tensor.dtype(),
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            attributes.output_mem_config))};
}

SoftmaxDeviceOperation::tensor_return_value_t SoftmaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Inplace config
    if ((attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
         attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) &&
        attributes.inplace) {
        return tensor_args.input_tensor;
    }

    // Standard
    return {create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input_tensor.device())};
}

tt::tt_metal::operation::Hash SoftmaxDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return operation::hash_operation<SoftmaxDeviceOperation>(
        select_program_factory(attributes, tensor_args).index(),
        attributes.softmax_type,
        attributes.dim,
        attributes.scale,
        attributes.inplace,
        attributes.output_mem_config,
        attributes.program_config,
        attributes.is_causal_mask,
        attributes.compute_kernel_config,
        attributes.is_scale_causal_mask_hw_dims_softmax,
        attributes.numeric_stable,
        tensor_args.input_tensor.logical_shape(),
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.layout());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SoftmaxDeviceOperation::tensor_return_value_t>
SoftmaxDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, const Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    int ideal_dev_clock_cycles = data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

std::tuple<SoftmaxDeviceOperation::operation_attributes_t, SoftmaxDeviceOperation::tensor_args_t>
SoftmaxDeviceOperation::invoke(
    SoftmaxOperationType softmax_type,
    const Tensor& input_tensor,
    int8_t dim,
    const std::optional<const Tensor>& mask,
    std::optional<float> scale,
    bool inplace,
    tt::tt_metal::MemoryConfig output_mem_config,
    SoftmaxProgramConfig program_config,
    bool is_causal_mask,
    DeviceComputeKernelConfig compute_kernel_config,
    bool is_scale_causal_mask_hw_dims_softmax,
    bool numeric_stable) {
    return {
        operation_attributes_t{
            softmax_type,
            dim,
            scale,
            inplace,
            output_mem_config,
            program_config,
            is_causal_mask,
            compute_kernel_config,
            is_scale_causal_mask_hw_dims_softmax,
            numeric_stable},
        tensor_args_t{input_tensor, mask}};
}

Tensor softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    int8_t dim,
    tt::tt_metal::MemoryConfig output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);
    const auto rank = input_tensor.logical_shape().size();

    if (rank > 4) {
        // General-purpose softmax
        return ttnn::prim::softmax(
            queue_id,
            SoftmaxOperationType::Softmax,
            /*input_tensor=*/input_tensor,
            /*dim=*/dim,
            /*mask=*/std::nullopt,
            /*scale=*/std::nullopt,
            /*inplace=*/false,
            /*output_mem_config=*/output_mem_config,
            /*program_config=*/SoftmaxDefaultProgramConfig{},
            /*is_causal_mask=*/false,
            /*compute_kernel_config=*/compute_kernel_config_val,
            /*is_scale_causal_mask_hw_dims_softmax=*/false,
            /*numeric_stable=*/numeric_stable);
    }

    auto input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    if (dim == rank - 1) {
        // Input tensor formatting
        const ttnn::Shape input_pad_shape =
            ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.padded_shape());
        const ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
            .pad_shape = input_pad_shape,
            .pad_value = -std::numeric_limits<float>::infinity(),
            .target_layout = tt::tt_metal::Layout::TILE};
        auto formatted_input_tensor = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
            input_tensor,
            input_tensor.device(),
            input_format_params.pad_shape,
            input_format_params.pad_value,
            input_format_params.target_layout);
        // Attention optimized softmax
        return ttnn::prim::softmax(
            queue_id,
            SoftmaxOperationType::Softmax,
            /*input_tensor=*/formatted_input_tensor,
            /*dim=*/-1,
            /*mask=*/std::nullopt,
            /*scale=*/std::nullopt,
            /*inplace=*/false,
            /*output_mem_config=*/output_mem_config,
            /*program_config=*/SoftmaxDefaultProgramConfig{},
            /*is_causal_mask=*/false,
            /*compute_kernel_config=*/compute_kernel_config_val,
            /*is_scale_causal_mask_hw_dims_softmax=*/false,
            /*numeric_stable=*/numeric_stable);
    }
    // General-purpose softmax
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::Softmax,
        /*input_tensor=*/input_tensor_4D,
        /*dim=*/dim,
        /*mask=*/std::nullopt,
        /*scale=*/std::nullopt,
        /*inplace=*/false,
        /*output_mem_config=*/output_mem_config,
        /*program_config=*/SoftmaxDefaultProgramConfig{},
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_mask_softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    tt::tt_metal::MemoryConfig output_mem_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Input tensor formatting
    const ttnn::Shape input_pad_shape =
        ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_tensor.padded_shape());
    const ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
        .pad_shape = input_pad_shape,
        .pad_value = -std::numeric_limits<float>::infinity(),
        .target_layout = tt::tt_metal::Layout::TILE};
    auto formatted_input_tensor = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
        input_tensor,
        input_tensor.device(),
        input_format_params.pad_shape,
        input_format_params.pad_value,
        input_format_params.target_layout);

    if (mask.has_value()) {
        TT_FATAL(
            input_tensor.padded_shape()[-1] == mask.value().padded_shape()[-1],
            "Input and mask inner dimensions must match, got input: {} vs mask: {}",
            input_tensor.padded_shape()[-1],
            mask.value().padded_shape()[-1]);
        TT_FATAL(
            input_tensor.padded_shape()[0] == mask.value().padded_shape()[0],
            "Input and mask batch sizes must match, got input: {} vs mask: {}",
            input_tensor.padded_shape()[0],
            mask.value().padded_shape()[0]);
        TT_FATAL(
            mask.value().padded_shape()[-2] == 1 or
                mask.value().padded_shape()[-2] == input_tensor.tensor_spec().tile().get_height(),
            "Mask height must be 1 or input tensor tile height, got: {}",
            mask.value().padded_shape()[-2]);
        for (uint32_t i = 1; i < input_tensor.padded_shape().rank() - 2; i++) {
            TT_FATAL(
                mask.value().padded_shape()[i] == 1,
                "Mask intermediate dimension {} must be 1, got: {}",
                i,
                mask.value().padded_shape()[i]);
        }
        const ttnn::Shape mask_pad_shape =
            ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(mask.value().padded_shape());
        const ttnn::operations::experimental::auto_format::FormatParams mask_format_params = {
            .pad_shape = mask_pad_shape,
            .pad_value = -std::numeric_limits<float>::infinity(),
            .target_layout = tt::tt_metal::Layout::TILE};
        auto formatted_mask = ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor(
            mask.value(),
            input_tensor.device(),
            mask_format_params.pad_shape,
            mask_format_params.pad_value,
            mask_format_params.target_layout);

        // Operation
        return ttnn::prim::softmax(
            queue_id,
            SoftmaxOperationType::ScaleMaskSoftmax,
            /*input_tensor=*/formatted_input_tensor,
            /*dim=*/-1,
            /*mask=*/formatted_mask,
            /*scale=*/scale,
            /*inplace=*/false,
            /*output_mem_config=*/output_mem_config,
            /*program_config=*/SoftmaxDefaultProgramConfig{},
            /*is_causal_mask=*/is_causal_mask,
            /*compute_kernel_config=*/compute_kernel_config_val,
            /*is_scale_causal_mask_hw_dims_softmax=*/false,
            /*numeric_stable=*/numeric_stable);
    }

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleMaskSoftmax,
        /*input_tensor=*/formatted_input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/false,
        /*output_mem_config=*/output_mem_config,
        /*program_config=*/SoftmaxDefaultProgramConfig{},
        /*is_causal_mask=*/is_causal_mask,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    int8_t dim,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation specific checks
    TT_FATAL(
        dim == input_tensor.logical_shape().size() - 1,
        "Invalid dimension: {}. Currently softmax inplace supports dim -1.",
        dim);
    TT_FATAL(
        input_tensor.logical_shape().size() == 4,
        "Invalid tensor shape: {}. Currently softmax inplace supports 4D tensors.",
        input_tensor.logical_shape().size());

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::SoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/dim,
        /*mask=*/std::nullopt,
        /*scale=*/std::nullopt,
        /*inplace=*/true,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_mask_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleMaskSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/true,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/is_causal_mask,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/false,
        /*numeric_stable=*/numeric_stable);
}

Tensor scale_causal_mask_hw_dims_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    SoftmaxProgramConfig program_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Operation
    return ttnn::prim::softmax(
        queue_id,
        SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/-1,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/true,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/false,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/true,
        /*numeric_stable=*/numeric_stable);
}
}  // namespace ttnn::operations::normalization::softmax
