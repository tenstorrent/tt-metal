#include "conv2d_device_operation_types.hpp"
#include "conv2d_program_factory_utils.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <tt-metalium/math.hpp>

#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/constants.hpp>

#include <tt-metalium/work_split.hpp>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operations/cb_utils.hpp"

#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/shape/shape.hpp"

namespace ttnn::operations::conv::conv2d {

void post_conv2d_op_checks(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor_a = tensor_args.a;
    const auto& input_tensor_b = tensor_args.b;
    const auto& input_tensor_bias = tensor_args.bias;
    tt::tt_metal::IDevice* device = input_tensor_a.device();

    const bool has_bias = input_tensor_bias.has_value();  // operation_attributes.has_bias???

    const auto& weights_shape = input_tensor_b.padded_shape();

    const uint32_t post_op_l1_allocation_size =
        device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;

    auto actual_cb_size = calculate_total_cb_size(program);

    auto kernel_dims = std::array<uint32_t, 2>(
        {operation_attributes.sliding_window_config.window_hw.first,
         operation_attributes.sliding_window_config.window_hw.second});

    const SkipMcast skip_mcast = conv_skip_mcast(
        operation_attributes.parallelization_config, operation_attributes.memory_config.memory_layout());
    const uint32_t output_image_width = operation_attributes.sliding_window_config.get_output_shape()[2];

    const std::array<uint32_t, 2> shard_shape = input_tensor_a.shard_spec().value().shape;
    const uint32_t input_channels_padded = shard_shape[1];
    conv_op_l1_usage l1_usage = calculate_L1_usage(
        operation_attributes.compute_kernel_config,
        operation_attributes.block_config,
        operation_attributes.parallelization_config,
        weights_shape,
        operation_attributes.sliding_window_config,
        std::array<uint32_t, 2>(
            {operation_attributes.sliding_window_config.dilation_hw.first,
             operation_attributes.sliding_window_config.dilation_hw.second}),
        Conv2dConfig{
            .weights_dtype = input_tensor_b.dtype(),
            .config_tensors_in_dram = operation_attributes.config_tensors_in_dram,
            .shard_layout = operation_attributes.memory_config.memory_layout(),
            .output_layout = (operation_attributes.untilize_out ? Layout::ROW_MAJOR : Layout::TILE),
            .enable_act_double_buffer = operation_attributes.enable_act_double_buffer,
            .enable_weights_double_buffer = operation_attributes.enable_weights_double_buffer,
            .enable_activation_reuse = operation_attributes.enable_activation_reuse,
            .force_split_reader = operation_attributes.force_split_reader},
        input_tensor_a.dtype(),
        operation_attributes.dtype,
        output_image_width,
        has_bias,
        is_1d_deptwise_conv(
            operation_attributes.groups,
            operation_attributes.input_tensor_shape[3],
            operation_attributes.output_channels,
            kernel_dims[1],
            output_image_width,
            has_bias),
        input_channels_padded,
        skip_mcast.skip_activation_mcast);

    TT_FATAL(
        actual_cb_size == l1_usage.CB_allocation_size,
        "Calculated CB size {} does not match with the actual CB size {}",
        l1_usage.CB_allocation_size,
        actual_cb_size);

    // For now assume that if post_op_l1_allocation_size == 0 op is being run
    // in graph capture NO_DISPATCH mode.
    // ToDo: Device should offer an API to inform the op if it is running in NO_DISPATCH mode.
    bool is_graph_capture_no_dispathch_mode = post_op_l1_allocation_size == 0;
    TT_FATAL(
        post_op_l1_allocation_size ==
                (operation_attributes.pre_op_l1_allocation_size_bytes + l1_usage.tensor_allocation_size) ||
            is_graph_capture_no_dispathch_mode,
        "Mismatch!! L1 Allocation Pre Op =  {}, Post Op = {} Calculated Size = {}",
        operation_attributes.pre_op_l1_allocation_size_bytes,
        post_op_l1_allocation_size,
        l1_usage.tensor_allocation_size);
}

}  // namespace ttnn::operations::conv::conv2d
