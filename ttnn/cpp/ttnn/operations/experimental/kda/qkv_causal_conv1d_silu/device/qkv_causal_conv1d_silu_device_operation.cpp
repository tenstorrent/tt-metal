// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_device_operation.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

QkvCausalConv1dSiluOperation::program_factory_t QkvCausalConv1dSiluOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QkvCausalConv1dSiluProgramFactory{};
}

void QkvCausalConv1dSiluOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr std::string_view operation_name = "qkv_causal_conv1d_silu";
    kda_factory_detail::check_actual_start(in.input, in.actual_start, operation_name);
    TT_FATAL(
        in.predecessor_carry.has_value() != in.exchange_buffer.has_value(),
        "qkv_causal_conv1d_silu: pass exactly one of predecessor_carry and exchange_buffer");
    if (in.predecessor_carry.has_value()) {
        kda_factory_detail::check_allocated_device_tensor(*in.predecessor_carry, operation_name, "predecessor_carry");
        kda_factory_detail::check_same_device(in.input, *in.predecessor_carry, operation_name, "predecessor_carry");
        TT_FATAL(
            in.predecessor_carry->tensor_spec() == in.history.tensor_spec(), "qkv convolution: carries must match");
    }
    if (in.actual_end.has_value()) {
        check_actual_start(in.input, *in.actual_end, operation_name);
    }

    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::TILE, operation_name, "input");
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    check_allocated_device_tensor(in.history, operation_name, "history");
    check_layout(in.history, Layout::ROW_MAJOR, operation_name, "history");
    check_dtype(in.history, DataType::BFLOAT16, operation_name, "history");
    check_interleaved(in.history, operation_name, "history");
    check_allocated_device_tensor(in.tap0, operation_name, "tap0");
    check_layout(in.tap0, Layout::TILE, operation_name, "tap0");
    check_dtype(in.tap0, DataType::BFLOAT16, operation_name, "tap0");
    check_interleaved(in.tap0, operation_name, "tap0");
    check_allocated_device_tensor(in.tap1, operation_name, "tap1");
    check_layout(in.tap1, Layout::TILE, operation_name, "tap1");
    check_dtype(in.tap1, DataType::BFLOAT16, operation_name, "tap1");
    check_interleaved(in.tap1, operation_name, "tap1");
    check_allocated_device_tensor(in.tap2, operation_name, "tap2");
    check_layout(in.tap2, Layout::TILE, operation_name, "tap2");
    check_dtype(in.tap2, DataType::BFLOAT16, operation_name, "tap2");
    check_interleaved(in.tap2, operation_name, "tap2");
    check_allocated_device_tensor(in.tap3, operation_name, "tap3");
    check_layout(in.tap3, Layout::TILE, operation_name, "tap3");
    check_dtype(in.tap3, DataType::BFLOAT16, operation_name, "tap3");
    check_interleaved(in.tap3, operation_name, "tap3");
    check_same_device(in.input, in.history, operation_name, "history");
    check_same_device(in.input, in.tap0, operation_name, "tap0");
    check_same_device(in.input, in.tap1, operation_name, "tap1");
    check_same_device(in.input, in.tap2, operation_name, "tap2");
    check_same_device(in.input, in.tap3, operation_name, "tap3");

    TT_FATAL(
        attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be positive");
    TT_FATAL(
        attrs.q_width % tt::constants::TILE_WIDTH == 0 && attrs.k_width % tt::constants::TILE_WIDTH == 0 &&
            attrs.v_width % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be tile aligned");
    const uint64_t channels =
        static_cast<uint64_t>(attrs.q_width) + static_cast<uint64_t>(attrs.k_width) + attrs.v_width;
    TT_FATAL(attrs.channel_chunk_size > 0, "qkv_causal_conv1d_silu: channel_chunk_size must be positive");
    TT_FATAL(
        attrs.channel_chunk_size % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must be tile aligned");
    TT_FATAL(
        attrs.channel_chunk_size <= channels, "qkv_causal_conv1d_silu: channel_chunk_size must not exceed Q+K+V width");
    TT_FATAL(
        channels % attrs.channel_chunk_size == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must divide Q+K+V width exactly");
    const auto* mesh = in.input.device();
    // With the exchange, every convolution core, all of the grid but the exchange core, stages its share of the rows.
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t work_items = (attrs.sequence / tt::constants::TILE_HEIGHT) * (channels / attrs.channel_chunk_size);
    const uint32_t max_staged_ct = in.exchange_buffer.has_value()
                                       ? qkv_causal_conv1d_silu_staged_ct(
                                             channels / tt::constants::TILE_WIDTH,
                                             qkv_causal_conv1d_silu_distribution(grid, work_items, true).cores.size())
                                       : 0;
    const uint64_t required_l1_bytes = qkv_causal_conv1d_silu_l1_bytes(
        attrs.channel_chunk_size / tt::constants::TILE_WIDTH,
        qkv_conv::dest_tile_limit(attrs.compute_kernel_config.fp32_dest_acc_en),
        tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(in.input.dtype())),
        max_staged_ct);
    const uint64_t available_l1_bytes =
        mesh->l1_size_per_core() - mesh->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    TT_FATAL(
        required_l1_bytes <= available_l1_bytes,
        "qkv_causal_conv1d_silu: channel_chunk_size = {} needs {} bytes of L1 per core, only {} are available",
        attrs.channel_chunk_size,
        required_l1_bytes,
        available_l1_bytes);

    const auto& input_shape = in.input.logical_shape();
    const auto& history_shape = in.history.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == 1 && input_shape[1] == attrs.sequence &&
            input_shape[2] >= attrs.input_column_offset + channels,
        "qkv_causal_conv1d_silu: input must be [1,T,W] holding the convolution channels from input_column_offset");
    TT_FATAL(
        attrs.input_column_offset % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: input_column_offset must be tile aligned");
    if (in.exchange_buffer.has_value()) {
        const auto& buffer = *in.exchange_buffer;
        check_allocated_device_tensor(buffer, operation_name, "exchange_buffer");
        check_layout(buffer, Layout::ROW_MAJOR, operation_name, "exchange_buffer");
        check_dtype(buffer, DataType::BFLOAT16, operation_name, "exchange_buffer");
        check_interleaved(buffer, operation_name, "exchange_buffer");
        check_same_device(in.input, buffer, operation_name, "exchange_buffer");
        TT_FATAL(
            buffer.memory_config().buffer_type() == BufferType::DRAM,
            "qkv_causal_conv1d_silu: exchange_buffer must be in DRAM");
        const auto& buffer_shape = buffer.logical_shape();
        TT_FATAL(
            buffer_shape.rank() == 3 && buffer_shape[0] == 1 && buffer_shape[1] == qkv_conv::exchange_rows &&
                buffer_shape[2] == channels,
            "qkv_causal_conv1d_silu: exchange_buffer must be [1,{},Q+K+V], got {}",
            qkv_conv::exchange_rows,
            buffer_shape);
        TT_FATAL(
            std::ranges::none_of(attrs.exchange_semaphore_addresses, [](uint32_t address) { return address == 0; }),
            "qkv_causal_conv1d_silu: exchange_buffer needs its exchange semaphores");
    }
    TT_FATAL(
        history_shape.rank() == 3 && history_shape[0] == 1 && history_shape[1] == qkv_conv::history_rows &&
            history_shape[2] == channels,
        "qkv_causal_conv1d_silu: history must be [1,{},Q+K+V]",
        qkv_conv::history_rows);
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "qkv_causal_conv1d_silu: sequence must be positive and tile aligned");
    for (const auto& [tensor, name] : std::array{
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"},
             std::pair{&in.tap3, "tap3"}}) {
        TT_FATAL(
            tensor->logical_shape()[-1] == channels,
            "qkv_causal_conv1d_silu: {} last dimension must equal Q+K+V",
            name);
        TT_FATAL(
            tensor->logical_volume() == channels, "qkv_causal_conv1d_silu: {} logical volume must equal Q+K+V", name);
    }
    check_output_interleaved(attrs.output_mem_config, operation_name);
    check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(
        !attrs.compute_kernel_config.math_approx_mode,
        "qkv_causal_conv1d_silu: math_approx_mode=true is unsupported because the fused SiLU uses the precise sigmoid");
}

QkvCausalConv1dSiluOperation::spec_return_value_t QkvCausalConv1dSiluOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    spec_return_value_t specs = {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
    if (in.exchange_buffer.has_value()) {
        specs.push_back(TensorSpec(
            in.history.logical_shape(),
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG)));
    }
    return specs;
}

QkvCausalConv1dSiluOperation::tensor_return_value_t QkvCausalConv1dSiluOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    tensor_return_value_t outputs;
    for (const auto& spec : compute_output_specs(attrs, in)) {
        outputs.push_back(create_device_tensor(spec, in.input.device()));
    }
    return outputs;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<QkvCausalConv1dSiluOperation::tensor_return_value_t>
QkvCausalConv1dSiluOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    const auto& input_shape = in.input.logical_shape();
    const double width = static_cast<double>(attrs.q_width) + attrs.k_width + attrs.v_width;
    const double elements = static_cast<double>(input_shape[0]) * attrs.sequence * width;
    const KdaFpuWork work{
        .fpu_multiply_ops = 4.0 * elements,
        .fpu_add_ops = 3.0 * elements,
    };
    const std::array<const Tensor*, 6> inputs = {&in.input, &in.history, &in.tap0, &in.tap1, &in.tap2, &in.tap3};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

}  // namespace ttnn::experimental::prim
