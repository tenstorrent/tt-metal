// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm_grid_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

GroupNormDeviceOperation::program_factory_t GroupNormDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (input.is_sharded()) {
        return GroupNormShardedProgramFactory{};
    }

    // For non-sharded: determine if we need mcast or no-mcast based on batch vs virtual rows
    const auto& program_config = std::get<GroupNormMultiCoreProgramConfig>(args.program_config);
    CoreCoord grid_size = program_config.compute_with_storage_grid_size;
    uint32_t batch = input.padded_shape()[0];
    uint32_t W = input.padded_shape()[3];
    uint32_t num_virtual_cols =
        ttnn::operations::normalization::compute_num_virtual_cols(grid_size.x, args.num_groups, W);
    TT_FATAL(
        num_virtual_cols > 0,
        "group_norm: No valid num_virtual_cols for grid_x={}, num_groups={}, W={}. "
        "Channels must be aligned to tile width and divisible by num_groups.",
        grid_size.x,
        args.num_groups,
        W);

    uint32_t num_actual_rows = grid_size.y;
    uint32_t num_virtual_rows = (grid_size.x / num_virtual_cols) * num_actual_rows;

    if (batch >= num_virtual_rows) {
        return GroupNormNoMcastProgramFactory{};
    }
    return GroupNormMcastProgramFactory{};
}

void GroupNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& negative_mask = tensor_args.negative_mask;
    const auto& reciprocals = tensor_args.reciprocals;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    TT_FATAL(a.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to groupnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
    TT_FATAL(a.padded_shape()[3] % args.num_groups == 0, "channel must be divisible by num_groups!");
    TT_FATAL(a.padded_shape()[1] == 1, "input tensor shape[1] must be 1!");
    TT_FATAL(
        (a.padded_shape()[1] * a.padded_shape()[2]) % tile_height == 0,
        "H*W ({}*{}) must be a multiple of the tile height ({})",
        a.padded_shape()[1],
        a.padded_shape()[2],
        tile_height);

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[3] == gamma.value().padded_shape()[3],
                "{} != {}",
                a.padded_shape()[3],
                gamma.value().padded_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                gamma.value().padded_shape()[2] == tile_height,
                "Gamma tensor height must equal tile height ({}), got: {}",
                tile_height,
                gamma.value().padded_shape()[2]);
        } else {
            TT_FATAL(
                gamma.value().layout() == Layout::ROW_MAJOR,
                "Gamma tensor must have ROW_MAJOR layout, got: {}",
                gamma.value().layout());
            TT_FATAL(
                (gamma.value().padded_shape()[3] == tile_width),
                "Gamma tensor inner dimension must equal tile width ({}), got: {}",
                tile_width,
                gamma.value().padded_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                gamma.value().dtype() == DataType::BFLOAT16,
                "Gamma tensor must be BFLOAT16, got: {}",
                gamma.value().dtype());
        }
        if (beta.has_value()) {
            TT_FATAL(
                gamma.value().layout() == beta.value().layout(),
                "Gamma and beta must have the same layout, got gamma: {} vs beta: {}",
                gamma.value().layout(),
                beta.value().layout());
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[3] == beta.value().padded_shape()[3],
                "Input and beta inner dimensions must match, got input: {} vs beta: {}",
                a.padded_shape()[3],
                beta.value().padded_shape()[3]);
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                beta.value().padded_shape()[2] == tile_height,
                "Beta tensor height must equal tile height ({}), got: {}",
                tile_height,
                beta.value().padded_shape()[2]);
        } else {
            TT_FATAL(
                beta.value().layout() == Layout::ROW_MAJOR,
                "Beta tensor must have ROW_MAJOR layout, got: {}",
                beta.value().layout());
            TT_FATAL(
                beta.value().padded_shape()[3] == tile_width,
                "Beta tensor inner dimension must equal tile width ({}), got: {}",
                tile_width,
                beta.value().padded_shape()[3]);
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                beta.value().dtype() == DataType::BFLOAT16,
                "Beta tensor must be BFLOAT16, got: {}",
                beta.value().dtype());
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask.value().layout() == Layout::TILE,
            "Input mask must have TILE layout, got: {}",
            input_mask.value().layout());
        TT_FATAL(
            input_mask.value().padded_shape()[1] == args.num_groups,
            "Input mask dim1 must match number of groups, got: {} vs {}",
            input_mask.value().padded_shape()[1],
            args.num_groups);
        TT_FATAL(
            input_mask.value().padded_shape()[2] == tile_height,
            "Input mask height must equal tile height ({}), got: {}",
            tile_height,
            input_mask.value().padded_shape()[2]);
        TT_FATAL(
            input_mask.value().padded_shape()[3] % tile_width == 0,
            "Input mask inner dimension must be divisible by tile width ({}), got: {}",
            tile_width,
            input_mask.value().padded_shape()[3]);
    }

    // Negative mask tensor is used to reduce the number of CB's used in the sharded version of the kernel by
    // overlapping the CB's used for tilized input and output. (The kernel is in fact row major variant, but is
    // internally tilizing RM into tilized inputs) Valid only if sharded program is used, and input and output tensors
    // are in row major layout.
    if (negative_mask.has_value()) {
        TT_FATAL(
            negative_mask.value().layout() == Layout::TILE,
            "Negative musk must be in TILE layout, but layout is {}",
            negative_mask.value().layout());
        TT_FATAL(
            negative_mask.value().padded_shape()[1] == args.num_groups,
            "Negative mask padded shape[1] must be equal to num_groups, but is {} and num_groups is {}",
            negative_mask.value().padded_shape()[1],
            args.num_groups);
        TT_FATAL(
            negative_mask.value().padded_shape()[2] == tile_height,
            "Negative mask padded shape[2] must equal tile height, but is {} and tile_height is {}",
            negative_mask.value().padded_shape()[2],
            tile_height);
        TT_FATAL(
            negative_mask.value().padded_shape()[3] % tile_width == 0,
            "Negative mask padded shape[3] must be divisible by tile_width, but is {} and tile_width is {}",
            negative_mask.value().padded_shape()[3],
            tile_width);
        TT_FATAL(a.is_sharded(), "Negative mask support is only available for sharded input tensors.");
        TT_FATAL(
            a.layout() == Layout::ROW_MAJOR,
            "If using negative mask, input tensor must be in ROW_MAJOR layout, but layout is {}",
            a.layout());
        Layout output_layout =
            std::visit([](const auto& config) -> Layout { return config.output_layout; }, args.program_config);
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR,
            "If using negative mask, output tensor must be in ROW_MAJOR layout, but layout is {}",
            output_layout);
    }

    // Reciprocals tensor validation
    if (reciprocals.has_value()) {
        TT_FATAL(args.use_welford, "Reciprocals tensor can only be provided when use_welford is True");
        TT_FATAL(
            reciprocals.value().dtype() == DataType::FLOAT32,
            "Reciprocals tensor must be FLOAT32, got: {}",
            reciprocals.value().dtype());
        TT_FATAL(reciprocals.value().storage_type() == StorageType::DEVICE, "Reciprocals tensor must be on device");
        TT_FATAL(reciprocals.value().buffer() != nullptr, "Reciprocals tensor must be allocated in buffers on device");
        TT_FATAL(a.device() == reciprocals.value().device(), "Input and reciprocals tensors must be on same device");
    }

    // For non-sharded DRAM tensors, validate that the grid produces uniform
    // multicast groups.  Non-uniform groups cause a deadlock because the sender
    // kernel waits for an exact semaphore count equal to (group_size - 1).
    if (!a.is_sharded()) {
        if (const auto* mc_config = std::get_if<GroupNormMultiCoreProgramConfig>(&args.program_config)) {
            CoreCoord grid_size = mc_config->compute_with_storage_grid_size;
            uint32_t W = a.padded_shape()[3];
            uint32_t num_batches = a.padded_shape()[0];
            uint32_t nvc = ttnn::operations::normalization::compute_num_virtual_cols(grid_size.x, args.num_groups, W);
            if (nvc > 0) {
                uint32_t num_virtual_rows = (grid_size.x / nvc) * grid_size.y;
                TT_FATAL(
                    num_virtual_rows < num_batches || num_virtual_rows % num_batches == 0,
                    "group_norm: The core grid (x={}, y={}) produces num_virtual_rows={} which is not "
                    "divisible by num_batches={}. This creates non-uniform multicast groups and will "
                    "deadlock. Use determine_expected_group_norm_dram_grid_size() with num_batches to select a valid "
                    "grid.",
                    grid_size.x,
                    grid_size.y,
                    num_virtual_rows,
                    num_batches);
            }
        }
    }
}

TensorSpec GroupNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    return std::visit(
        [&](const auto& program_config) -> spec_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return input_tensor.tensor_spec();
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }

            auto mem_config = args.output_mem_config;
            return TensorSpec(
                input_tensor.logical_shape(),
                TensorLayout::fromPaddedShape(
                    program_config.out_data_format,
                    PageConfig(program_config.output_layout),
                    mem_config,
                    input_tensor.logical_shape(),
                    input_tensor.padded_shape()));
        },
        args.program_config);
}

Tensor GroupNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    return std::visit(
        [&](const auto& program_config) -> tensor_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return input_tensor;
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }
            return create_device_tensor(compute_output_specs(args, tensor_args), input_tensor.device());
        },
        args.program_config);
}

Tensor group_norm(
    const Tensor& input,
    float eps,
    uint32_t num_groups,
    const MemoryConfig& output_mem_config,
    const GroupNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool use_welford,
    std::optional<Tensor> gamma,
    std::optional<Tensor> beta,
    std::optional<Tensor> input_mask,
    std::optional<Tensor> negative_mask,
    std::optional<Tensor> reciprocals) {
    using OperationType = GroupNormDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .eps = eps,
        .num_groups = num_groups,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .use_welford = use_welford,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input,
        .gamma = std::move(gamma),
        .beta = std::move(beta),
        .input_mask = std::move(input_mask),
        .negative_mask = std::move(negative_mask),
        .reciprocals = std::move(reciprocals)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
