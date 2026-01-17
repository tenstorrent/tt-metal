// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pool_op.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-metalium/math.hpp>
#include <utility>

/**
 * Generic pool implementation that uses the new sliding window infrastructure.
 */

namespace ttnn::operations::pool {

Pool2D::program_factory_t Pool2D::select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
    return MultiCore{};
}

void validate_pool2d(
    const Tensor& input,
    const Pool2DType pool_type,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    const MemoryConfig& out_mem_config,
    const std::optional<const int32_t> /*divisor_override*/,
    const bool return_indices,
    const Layout& output_layout) {
    // check the input tensor
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input.dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now. Tracked by issue #23338");

    TT_FATAL(input.memory_config().is_sharded(), "Input needs to be sharded");

    if (return_indices) {
        TT_FATAL(pool_type == Pool2DType::MAX_POOL2D, "Return_indices is only supported for MAX pool type");

        // generality constraints, see https://github.com/tenstorrent/tt-metal/issues/27845
        auto input_h = sliding_window_config.input_hw.first;
        auto input_w = sliding_window_config.input_hw.second;
        TT_FATAL(
            input_h * input_w <= std::numeric_limits<uint16_t>::max(),
            "Input HW {} will overflow uint16 indices max {}",
            input_h * input_w,
            std::numeric_limits<uint16_t>::max());
        auto kernel_h = sliding_window_config.window_hw.first;
        auto kernel_w = sliding_window_config.window_hw.second;
        TT_FATAL(
            kernel_h * kernel_w <= 32,
            "only kernel sizes less than or equal to 32 are supported, got {}x{}",
            kernel_h,
            kernel_w);

        TT_FATAL(output_layout == Layout::ROW_MAJOR, "Only ROW_MAJOR supported when return_indices is true");
    }

    TT_FATAL(out_mem_config.is_sharded(), "Output memory config needs to be sharded");

    const auto& input_shape = input.padded_shape();

    // check that C dimnenion is a multiple of num_shards_c for all but height sharding
    TensorMemoryLayout in_memory_layout = input.memory_config().memory_layout();
    if (in_memory_layout != TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_shards_c = sliding_window_config.num_cores_c;
        TT_FATAL(
            input_shape[3] % num_shards_c == 0,
            "For width and block sharding, input channels ({}) should be divisible by num_shards ({})",
            input_shape[3],
            num_shards_c);
    }
}

void Pool2D::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    validate_pool2d(
        tensor.input_tensor_,
        op_attr.pool_type_,
        op_attr.sliding_window_config_,
        op_attr.memory_config_,
        op_attr.divisor_override_,
        op_attr.return_indices_,
        op_attr.output_layout_);
}

void Pool2D::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    validate_pool2d(
        tensor.input_tensor_,
        op_attr.pool_type_,
        op_attr.sliding_window_config_,
        op_attr.memory_config_,
        op_attr.divisor_override_,
        op_attr.return_indices_,
        op_attr.output_layout_);
}

Pool2D::spec_return_value_t Pool2D::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensor*/) {
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const auto& out_mem_config = op_attr.memory_config_;
    const auto& output_dtype = op_attr.output_dtype_;

    uint32_t out_h = sliding_window_config.get_output_shape()[1];
    uint32_t out_w = sliding_window_config.get_output_shape()[2];
    uint32_t out_c = sliding_window_config.channels;
    uint32_t batch_size = sliding_window_config.batch_size;
    uint32_t out_nhw = batch_size * out_h * out_w;

    bool is_out_tiled = op_attr.output_layout_ == Layout::TILE;
    uint32_t tile_rows = is_out_tiled ? tt::constants::TILE_HEIGHT : 1;

    uint32_t num_cores_nhw = sliding_window_config.num_cores_nhw;
    uint32_t num_cores_c = sliding_window_config.num_cores_c;
    TT_FATAL(num_cores_nhw > 0, "num_cores_nhw must be > 0");
    TT_FATAL(num_cores_c > 0, "num_cores_c must be > 0");

    auto mem_config = out_mem_config;
    auto layout = mem_config.memory_layout();

    uint32_t out_nhw_padded = tt::round_up(out_nhw, tile_rows * num_cores_nhw);
    uint32_t out_c_padded = tt::round_up(out_c, tt::constants::TILE_WIDTH / 2);
    if (mem_config.is_sharded()) {
        if (layout == TensorMemoryLayout::WIDTH_SHARDED || layout == TensorMemoryLayout::BLOCK_SHARDED) {
            out_c_padded = tt::round_up(out_c, sliding_window_config.num_cores_c * tt::constants::TILE_WIDTH / 2);
        }
    }

    if (is_out_tiled) {
        out_c_padded = tt::round_up(out_c, tt::constants::TILE_WIDTH * sliding_window_config.num_cores_c);
        out_nhw_padded = tt::round_up(out_nhw_padded, tt::constants::TILE_HEIGHT * sliding_window_config.num_cores_nhw);
    }

    ttnn::Shape padded_output_shape({1, 1, out_nhw_padded, out_c_padded});
    ttnn::Shape output_shape({1, 1, out_nhw, out_c});

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            output_dtype, op_attr.output_layout_, mem_config, output_shape, padded_output_shape));
}

Pool2D::tensor_return_value_t Pool2D::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    auto output_spec_data = compute_output_specs(op_attr, tensor);
    if (op_attr.return_indices_) {
        // the index output spec is the same as the input spec just with a different data type
        tt::tt_metal::TensorLayout output_layout_ind(
            DataType::UINT16,
            output_spec_data.page_config(),
            output_spec_data.memory_config(),
            output_spec_data.tensor_layout().get_alignment());
        auto output_spec_ind = TensorSpec(output_spec_data.logical_shape(), output_layout_ind);
        return {
            create_device_tensor(output_spec_data, tensor.input_tensor_.device()),
            create_device_tensor(output_spec_ind, tensor.input_tensor_.device())};
    }
    return {create_device_tensor(output_spec_data, tensor.input_tensor_.device())};
}

tt::stl::hash::hash_t Pool2D::compute_program_hash(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    auto input_mem_config = tensor.input_tensor_.memory_config();
    auto in_dtype = tensor.input_tensor_.dtype();
    auto out_dtype = op_attr.output_dtype_;
    return tt::tt_metal::operation::hash_operation<Pool2D>(
        op_attr.sliding_window_config_.get_hash(),
        op_attr.pool_type_,
        op_attr.memory_config_,
        op_attr.divisor_override_,
        op_attr.count_include_pad_,
        op_attr.return_indices_,
        input_mem_config,
        in_dtype,
        out_dtype);
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Pool2D::tensor_return_value_t> Pool2D::create_op_performance_model(
    const operation_attributes_t& op_attr, const tensor_args_t& tensor, const tensor_return_value_t& outputs) {
    const auto& input = tensor.input_tensor_;
    const auto& input_shape = input.logical_shape();
    auto sliding_window_config = op_attr.sliding_window_config_;
    uint32_t batch_size = sliding_window_config.batch_size;
    uint32_t activation_h = sliding_window_config.input_hw.first;
    uint32_t activation_w = sliding_window_config.input_hw.second;
    uint32_t activation_c = input_shape[3];
    uint32_t output_channels = input_shape[3];

    uint32_t filter_h = sliding_window_config.window_hw.first;
    uint32_t filter_w = sliding_window_config.window_hw.second;
    uint32_t stride_h = sliding_window_config.stride_hw.first;
    uint32_t stride_w = sliding_window_config.stride_hw.second;
    uint32_t pad_h = sliding_window_config.get_pad_h();
    uint32_t pad_w = sliding_window_config.get_pad_w();

    // GS specific parameters
    int num_cores = 9 * 12;
    int tensix_mul_adds_per_cycle_lofi = 2048;

    // Calculate output dimensions: relevant for window/stride based OPs (conv, pool, downsample)
    int output_height = std::floor(((activation_h - filter_h + pad_h) / stride_h) + 1);
    int output_width = std::floor(((activation_w - filter_w + pad_w) / stride_w) + 1);

    // Calculate number of mul/add / compare operations
    int64_t num_mul_adds_per_elem = activation_c * filter_h * filter_w;  // 1 multiply and 1 add per element
    int64_t num_mul_adds = num_mul_adds_per_elem * output_height * output_width * output_channels * batch_size;

    int ideal_dev_clock_cycles = std::ceil((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi));

    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input}, {outputs}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::pool

namespace ttnn::prim {
std::vector<ttnn::Tensor> pool2d(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    ttnn::operations::pool::Pool2DType pool_type,
    DataType output_dtype,
    Layout output_layout,
    MemoryConfig memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    uint32_t memory_used,
    bool config_tensor_in_dram) {
    using OperationType = ttnn::operations::pool::Pool2D;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .sliding_window_config_ = sliding_window_config,
            .pool_type_ = pool_type,
            .output_dtype_ = output_dtype,
            .output_layout_ = output_layout,
            .memory_config_ = std::move(memory_config),
            .compute_kernel_config_ = compute_kernel_config,
            .count_include_pad_ = count_include_pad,
            .divisor_override_ = divisor_override,
            .return_indices_ = return_indices,
            .memory_used = memory_used,
            .config_tensor_in_dram = config_tensor_in_dram},
        OperationType::tensor_args_t{input_tensor});
}
}  // namespace ttnn::prim
