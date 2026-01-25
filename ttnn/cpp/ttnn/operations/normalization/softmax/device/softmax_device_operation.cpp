// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <utility>

#include "ttnn/device_operation.hpp"
#include "softmax_operation_types.hpp"

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
/**
 * @brief L1 memory threshold for small tensor optimizations (512KB)
 *
 * Tensors that require less than this amount of L1 memory can use
 * the "small" optimized implementations which keep all data in L1.
 */
#define L1_512KB (512 * 1024)

/**
 * @brief Check if small-width softmax optimization is available
 *
 * This function calculates the total L1 memory requirement for the small-width
 * implementation and compares it against available device L1 memory.
 *
 * @param tensor Input tensor to analyze
 * @param compute_kernel_config Compute configuration affecting memory usage
 * @return true if tensor fits in L1 memory for small-width optimization
 */
bool is_softmax_general_w_small_available(
    const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto w = tensor.logical_shape()[-1];
    int32_t Wt = (w + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tile_size(data_format);
    auto intermed_tile_size = tt::tile_size(intermed_data_format);

    // Calculate total circular buffer memory requirements
    int32_t cb_usage = 0;        // bytes
    cb_usage += Wt * tile_size;  // input buffer
    cb_usage += 1 * tile_size;   // mask buffer
    cb_usage += 1 * tile_size;   // scaler buffer

    cb_usage += Wt * tile_size;  // output buffer

    cb_usage += Wt * intermed_tile_size;  // exp(x) intermediate buffer
    cb_usage += 1 * intermed_tile_size;   // reduce intermediate buffer
    cb_usage += 1 * intermed_tile_size;   // max intermediate buffer
    cb_usage += Wt * intermed_tile_size;  // x - max intermediate buffer
    cb_usage += 1 * intermed_tile_size;   // tmp intermediate buffer

    return (tensor.device()->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
}

/**
 * @brief Check if small-height softmax optimization is available
 *
 * Similar to small-width check but for height dimension operations.
 */
bool is_softmax_general_h_small_available(
    const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config) {
    auto h = tensor.logical_shape()[-2];
    int32_t Ht = (h + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;

    auto arch = tensor.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;

    auto tile_size = tt::tile_size(data_format);
    auto intermed_tile_size = tt::tile_size(intermed_data_format);

    int32_t cb_usage = 0;        // bytes
    cb_usage += Ht * tile_size;  // input;
    cb_usage += 1 * tile_size;   // mask;
    cb_usage += 1 * tile_size;   // scaler;

    cb_usage += Ht * tile_size;  // output;

    cb_usage += Ht * intermed_tile_size;  // exp(x);
    cb_usage += 1 * intermed_tile_size;   // reduce;
    cb_usage += 1 * intermed_tile_size;   // max;
    cb_usage += Ht * intermed_tile_size;  // x - max;
    cb_usage += 1 * intermed_tile_size;   // tmp;

    return (tensor.device()->allocator()->get_base_allocator_addr(HalMemType::L1) + cb_usage <= L1_512KB);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

SoftmaxDeviceOperation::program_factory_t SoftmaxDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Determine if we should use sharded multi-core program factory
    const auto input_tensor_shape = tensor_args.input_tensor.padded_shape();
    const auto rank = input_tensor_shape.size();

    if (operation_attributes.softmax_type == SoftmaxOperationType::SoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmaxInPlace ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleMaskSoftmax ||
        operation_attributes.softmax_type == SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace) {
        return std::visit(
            [&](const auto& program_config) -> program_factory_t {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                    return SoftmaxShardedProgramFactoryAttentionOptimized{};
                } else {
                    return SoftmaxProgramFactoryAttentionOptimized{};
                }
            },
            operation_attributes.program_config);
        return SoftmaxProgramFactoryAttentionOptimized{};
    }
    if (operation_attributes.softmax_type == SoftmaxOperationType::Softmax && operation_attributes.dim == rank - 1 &&
        rank == 4) {
        return std::visit(
            [&](const auto& program_config) -> program_factory_t {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                    return SoftmaxShardedProgramFactoryAttentionOptimized{};
                } else {
                    return SoftmaxProgramFactoryAttentionOptimized{};
                }
            },
            operation_attributes.program_config);
        return SoftmaxProgramFactoryAttentionOptimized{};
    }
    if (rank - 1 == operation_attributes.dim) {
        if (CMAKE_UNIQUE_NAMESPACE::is_softmax_general_w_small_available(
                tensor_args.input_tensor, operation_attributes.compute_kernel_config)) {
            return SoftmaxProgramFactoryGeneralWSmall{};
        }
        return SoftmaxProgramFactoryGeneralWLarge{};
    }
    if (rank - 2 == operation_attributes.dim) {
        if (CMAKE_UNIQUE_NAMESPACE::is_softmax_general_h_small_available(
                tensor_args.input_tensor, operation_attributes.compute_kernel_config)) {
            return SoftmaxProgramFactoryGeneralHSmall{};
        }
        return SoftmaxProgramFactoryGeneralHLarge{};
    }
    return SoftmaxProgramFactoryGeneralCLarge{};
}

void SoftmaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
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
        const auto& mask = tensors_args.mask.value();
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
                            tensors_args.input_tensor.padded_shape()[0] == tensors_args.mask.value().padded_shape()[0],
                            "Input and mask batch sizes must match, got {} for input and {} for mask",
                            tensors_args.input_tensor.padded_shape(),
                            tensors_args.mask.value().padded_shape());
                        TT_FATAL(
                            !attributes.is_scale_causal_mask_hw_dims_softmax,
                            "Scale causal mask HW dims softmax not supported in default program config");
                    } else if constexpr (std::is_same_v<ProgramConfigType, SoftmaxShardedMultiCoreProgramConfig>) {
                        // Ensure input tensor is sharded when using sharded program config
                        TT_FATAL(
                            tensors_args.input_tensor.is_sharded() &&
                                tensors_args.input_tensor.shard_spec().has_value(),
                            "Input tensor must be sharded when using SoftmaxShardedMultiCoreProgramConfig");
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
    const operation_attributes_t& /*attributes*/, const tensor_args_t& tensor_args, const Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input_tensor;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

Tensor softmax(
    const Tensor& input_tensor,
    int8_t dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    TT_FATAL(
        input_tensor.device() != nullptr,
        "input_tensor.device() == nullptr, No device found, move input_tensor to device");
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);
    const auto rank = input_tensor.logical_shape().size();
    const auto dim_calculated = dim < 0 ? rank + dim : dim;
    if (rank > 4) {
        // General-purpose softmax
        return ttnn::prim::softmax(
            SoftmaxOperationType::Softmax,
            /*input_tensor=*/input_tensor,
            /*dim=*/dim_calculated,
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
    const auto dim_adjusted = dim < 0 ? input_tensor_4D.logical_shape().size() + dim : dim + (4 - rank);
    if (dim_adjusted == rank - 1) {
        // Input tensor formatting
        const ttnn::Shape input_pad_shape =
            ttnn::operations::data_movement::pad_to_tile_shape(input_tensor_4D.padded_shape());
        auto formatted_input_tensor = input_tensor_4D;
        if (formatted_input_tensor.layout() != Layout::TILE) {
            formatted_input_tensor = ttnn::tilize_with_val_padding(
                input_tensor_4D,
                input_pad_shape,
                -std::numeric_limits<float>::infinity(),
                input_tensor_4D.memory_config());
        }

        // Attention optimized softmax
        return ttnn::prim::softmax(
            SoftmaxOperationType::Softmax,
            /*input_tensor=*/formatted_input_tensor,
            /*dim=*/dim_adjusted,
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
        SoftmaxOperationType::Softmax,
        /*input_tensor=*/input_tensor_4D,
        /*dim=*/dim_adjusted,
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
    const Tensor& input_tensor,
    std::optional<float> scale,
    const std::optional<const Tensor>& mask,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool is_causal_mask,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    bool numeric_stable) {
    // Constants
    const auto is_fp32 = input_tensor.dtype() == DataType::FLOAT32;
    const auto compute_kernel_config_val = init_device_compute_kernel_config(
        input_tensor.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, true, is_fp32, false);

    // Input tensor formatting
    const ttnn::Shape input_pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
    auto formatted_input_tensor = input_tensor;
    if (formatted_input_tensor.layout() != Layout::TILE) {
        formatted_input_tensor = ttnn::tilize_with_val_padding(
            input_tensor, input_pad_shape, -std::numeric_limits<float>::infinity(), input_tensor.memory_config());
    }
    const auto rank = formatted_input_tensor.logical_shape().size();
    const auto dim = rank - 1;

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
            ttnn::operations::data_movement::pad_to_tile_shape(mask.value().padded_shape());
        auto formatted_mask = mask.value();
        if (formatted_mask.layout() != Layout::TILE) {
            formatted_mask = ttnn::tilize_with_val_padding(
                formatted_mask, mask_pad_shape, -std::numeric_limits<float>::infinity(), mask.value().memory_config());
        }

        // Operation
        return ttnn::prim::softmax(
            SoftmaxOperationType::ScaleMaskSoftmax,
            /*input_tensor=*/formatted_input_tensor,
            /*dim=*/dim,
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
        SoftmaxOperationType::ScaleMaskSoftmax,
        /*input_tensor=*/formatted_input_tensor,
        /*dim=*/dim,
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
    const auto rank = input_tensor.logical_shape().size();
    const auto dim = rank - 1;

    // Operation
    return ttnn::prim::softmax(
        SoftmaxOperationType::ScaleMaskSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/dim,
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
    const auto rank = input_tensor.logical_shape().size();
    const auto dim = rank - 1;

    // Operation
    return ttnn::prim::softmax(
        SoftmaxOperationType::ScaleCausalMaskHWSoftmaxInPlace,
        /*input_tensor=*/input_tensor,
        /*dim=*/dim,
        /*mask=*/mask,
        /*scale=*/scale,
        /*inplace=*/true,
        /*output_mem_config=*/input_tensor.memory_config(),
        /*program_config=*/program_config,
        /*is_causal_mask=*/true,
        /*compute_kernel_config=*/compute_kernel_config_val,
        /*is_scale_causal_mask_hw_dims_softmax=*/true,
        /*numeric_stable=*/numeric_stable);
}

Tensor softmax(
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
    return ttnn::device_operation::launch<SoftmaxDeviceOperation>(
        SoftmaxParams{
            softmax_type,
            dim,
            scale,
            inplace,
            std::move(output_mem_config),
            program_config,
            is_causal_mask,
            compute_kernel_config,
            is_scale_causal_mask_hw_dims_softmax,
            numeric_stable},
        SoftmaxInputs{input_tensor, mask});
}
}  // namespace ttnn::prim
